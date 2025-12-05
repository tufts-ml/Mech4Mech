from typing import Optional
import jax.numpy as jnp
import numpy as np

from utilities.util import (
    fix__log_emissions_from_entities__at_example_boundaries,
    fix__log_emissions_from_system__at_example_boundaries,
    fix_log_entity_transitions_at_example_boundaries,
    fix_log_system_transitions_at_example_boundaries,
)
from utilities.types import (
    JaxNumpyArray1D,
    JaxNumpyArray2D,
    JaxNumpyArray3D,
    JaxNumpyArray4D,
    NumpyArray1D,
)

from model import Model 
from params import (
    ContinuousStateParameters_JAX,
    EntityTransitionParameters_MetaSwitch_JAX,
    InitializationParameters_JAX,
    SystemTransitionParameters_JAX,
)
from compute_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
    compute_hmm_posterior_summaries_JAX,
    compute_hmm_posterior_summary_JAX,
)


"""
Executes the expectation step in the CAVI training: 

How it works: 

Given the current model params (STP, ETP, CSP, IP), we compute the MODEL log probabilities -> (compute_transitions.py) and 
(compute_emissions.py). 
VES step: We then take the expectation of the MODEL log probabilities with respect to our assumed variational 
posterior q(z^{1:J}_{0:T}) computed via the VEZ step. These give new log probabilitiy potentials to feed into 
the forwards-backwards algororithm -> (compute_posterior.py) to compute the optimal posterior for q(s_{0:T})
We feed the following into a forwards-backwards algorithm. 
Transitions: MODEL system transition log probabilities 
Emissions: Expected MODEL entity transition log probabilities wrt the variational entity posterior 
Initial: pi_system 
The forwards-backwards algorithm computes the posterior. 

VEZ step: We then take the expectation of the MODEL log probabilities with respect to our variational posterior q(s_{0:T})
computed via the VES step. These give new log probabilitiy potentials to feed into 
the forwards-backwards algororithm -> (compute_posterior.py) to compute the optimal posterior for q(z^{1:J}_{0:T}).
We feed the following into forwards-backwards: 
Transitions: Expected MODEL entity transition log probabilities wrt variational system posterior 
Emissions: Expected MODEL observation log probabilities wrt the variational system posterior 
Initial: pi_entities 

Why we do this: 

We take expectations with respect to each variational posterior as a result of our variational assumption: 
q(s_{0:T},z^{1:J}_{0:T}) = q(z^{1:J}_{0:T})q(s_{0:T})
Taking derivatives of the ELBO with respect to each component yields that an optimal form of one requires 
an expectation with respec to the other. 

"""

def compute_expected_log_entity_transition_probability_matrices_wrt_entity_regimes_JAX(
    ETP: EntityTransitionParameters_MetaSwitch_JAX,
    variationally_expected_joints_for_entity_regimes: JaxNumpyArray4D,
    observations: JaxNumpyArray3D,
    model: Model,
    outside_recurrence: Optional[JaxNumpyArray2D] = None,
) -> JaxNumpyArray3D:
    """
    Compute the expected log transition probability matrices from the MODEL, where the expectations are taken with
    respect to the variational distribution over the entity-level regimes.

    Arguments:
        ETP: the entity transition parameters Psis has shape (J, L, K, D_e) and Ps has shape (J, L, K, K) 
        variationally_expected_joints_for_entity_regimes: np.array of size (T-1,J,K,K) whose (t,j,k,k')-th element gives
        the probability distribution for entity j over all pairwise options
        (z^j_{t+1}=k', z^j_t=k).
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        model: joint distribution defined in -> (model.py)
        outside_recurrence: The recurrence features (T-1, D_e) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model. 

    Returns:
        np.array of shape (T,J,L), whose (t,j,l)-th element gives the (unnormalized) (autoregressive)
        (categorical) emissions density at time t for entity j under system regime l
    """
    # `log_transition_matrices` has shape (T-1,J,L,K,K)
    T, J = np.shape(observations)[:2]
    log_transition_matrices = model.compute_log_entity_transition_probability_matrices_JAX( #Compute log MODEL probabilities
        ETP,
        T-1, 
        observations[:-1],
        inside_recurrence=model.internal_entity_recurrence_JAX,
        outside_recurrence=outside_recurrence
    )

    expected_log_transition_matrices = jnp.einsum(
        "tjlkd, tjkd -> tjl",
        log_transition_matrices,
        variationally_expected_joints_for_entity_regimes,
    )
    return expected_log_transition_matrices


def run_VES_step_JAX(
    STP: SystemTransitionParameters_JAX,
    ETP: EntityTransitionParameters_MetaSwitch_JAX,
    IP: InitializationParameters_JAX,
    observations: JaxNumpyArray3D,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    example_end_times: Optional[JaxNumpyArray1D],
    outside_system_recurrence: Optional[JaxNumpyArray2D] = None,
    outside_entity_recurrence: Optional[JaxNumpyArray3D] = None,
    mask_observations: Optional[JaxNumpyArray2D] = None,
) -> HMM_Posterior_Summary_JAX:
    """
    Purpose:
        Compute the VES step for a group dynamics model, and return results in the
        form a HMM_Posterior_Summary.
        The VES step for a group dynamics model can be
        seen as the posterior of a HMM with adjusted parameters (due to taking expectations
        over the other random variables.)  First, we construct the initial distribution
        (actually a function arg), the transitions, and the emissions.
        Then, we feed this information into a generic forward-backward algo,
        namely the `hmm_expected_states` function from Linderman's ssm repo.
        Finally, we return the HMM_Posterior_Summary.

    Arguments:
        STP: The system state parameters Upsilon has shape (L, D_s) and Pi has shape (L, L)
        ETP: the entity transition parameters Psis has shape (J, L, K, D_e) and Ps has shape (J, L, K, K)
        IP:  the initial emission parameters pi_system, pi_entities, mu_0s, Sigma_0s
        observations: np.array of shape (T,J,D) where the (t,j)-th entry is in R^D. 
        VEZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
        model: joint distribution defined in -> (model.py)
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        outside_system_recurrence: The recurrence features (T-1, D_s) are provided, which are computed from the observations
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model. 
        outside_entity_recurrence: The recurrence features (T-1, D_e) are provided, which are computed from the observations
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model.
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.
       

    Returns: 
        A posterior summary of the latent expected marginals, expected pairwise joint, and the log density of the emissions.
    """

    T, J = np.shape(observations)[:2]
    L = len(IP.pi_system)

    if mask_observations is None:
        mask_observations = np.full((T, J), True)

    # `transitions` is (T-1) x L x L
    log_transitions = model.compute_log_system_transition_probability_matrices_JAX( #Compute log MODEL probabilities
        STP,
        T - 1,
        observations=observations[:-1],
        inside_recurrence=model.internal_system_recurrence_JAX,
        outside_recurrence=outside_system_recurrence
    )

    # ` initial_log_emission_for_each_system_regime` is a float after summing over (J,K) objects
    initial_log_emission_for_each_system_regime = jnp.sum(VEZ_summaries.expected_regimes[0] * np.log(IP.pi_entities))

    # `inital_log_emission` has shape (L,)
    initial_log_emission = jnp.repeat(initial_log_emission_for_each_system_regime, L)

    # `log_emissions_for_each_entity_after_initial_time` is (T-1) x J x L.. We need to collapse the J; emissions are independent over J.
    log_emissions_for_each_entity_after_initial_time = (
        compute_expected_log_entity_transition_probability_matrices_wrt_entity_regimes_JAX( #Compute expectations 
            ETP,
            VEZ_summaries.expected_joints,
            observations,
            model,
            outside_entity_recurrence
        )
    )

    log_emissions_for_each_entity_after_initial_time_with_mask_observations = (
        log_emissions_for_each_entity_after_initial_time * mask_observations[1:][:, :, None]
    )

    # `log_emissions_after_initial_time` is (T-1) x L.
    log_emissions_after_initial_time = jnp.sum(
        log_emissions_for_each_entity_after_initial_time_with_mask_observations, axis=1
    )

    # 'log_emissions' is TxL
    log_emissions = jnp.vstack((initial_log_emission, log_emissions_after_initial_time))

    ###
    # Patch the ingredients for the E-step if there are separate events.
    ###
    log_transitions = fix_log_system_transitions_at_example_boundaries(
        log_transitions,
        IP,
        example_end_times,
    )

    log_emissions = fix__log_emissions_from_system__at_example_boundaries(
        log_emissions, VEZ_summaries.expected_regimes, IP, example_end_times
    )

    return compute_hmm_posterior_summary_JAX( #Goes on to compute forewards-backwards
        log_transitions,
        log_emissions,
        IP.pi_system,
    )

###
# VEZ Step
###


def compute_expected_log_entity_transition_probability_matrices_wrt_system_regimes_JAX(
    ETP: EntityTransitionParameters_MetaSwitch_JAX,
    VES_expected_regimes: JaxNumpyArray2D,
    observations: JaxNumpyArray3D,
    model: Model,
    outside_recurrence: Optional[JaxNumpyArray3D] = None
) -> JaxNumpyArray3D:
    """
    Purpose: Compute expected log transition probability matrices from the MODEL, where the expectations are taken with
    respect to the variational distribution over the system-level regimes.

    Arguments:
        ETP: the entity transition parameters Psis has shape (J, L, K, D_e) and Ps has shape (J, L, K, K)
        VES_expected_regimes: np.array of size (T,L) whose (t,l)-th element gives
            the probability distribution of system regime l, q(s_t=l).
            May come from a HMM_Posterior_Summary instance created by the VES step.
        observations: np.array of shape (T,J,D) where the (t,j)-th entry is in R^D
        model: joint distribution defined in -> (model.py)
        outside_recurrence: The recurrence features (T-1, D_e) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model. 

    Returns:
        np.array of size (T-1,J,K,K) whose (t,j,k,k')-th element gives
        the probability distribution for entity j over all pairwise options
        (z^j_{t+1}=k', z^j_t=k).
    """

    T, J = np.shape(observations)[:2]
    # `log_transition_matrices` has shape (T-1,J,L,K,K)
    log_transition_matrices = model.compute_log_entity_transition_probability_matrices_JAX( #Compute log probabilities 
        ETP,
        T-1,
        observations[:-1],
        inside_recurrence=model.internal_entity_recurrence_JAX,
        outside_recurrence=outside_recurrence
    )

    expected_log_transition_matrices = jnp.einsum( #Compute expectations 
        "tjlkd, tl -> tjkd",
        log_transition_matrices,
        VES_expected_regimes[1:], 
    )

    return expected_log_transition_matrices


def compute_log_entity_emissions_JAX(
    CSP: ContinuousStateParameters_JAX,
    IP: InitializationParameters_JAX,
    observations: JaxNumpyArray3D,
    model: Model,
    example_end_times: Optional[NumpyArray1D] = None,
):
    """
    Purpose: 
        Compute the log (autoregressive, switching) emissions for the observations, where we we must combine:
        initial observation emission:  x_0^j | z_0 =k
        remaining observation emission:  x_t^j | x_(t-1)^j, z_t =k
        for entity-level regimes k=1,...,K and entities j=1,...,J

        This assumes many examples N. 

    Arguments:
        CSP: the observation parameters A[j,k], b[j,k], Q[j,k]
        IP: the initial emission parameters pi_sytem, pi_entity, mu_0s, Sigma_0s
        observations: np.array of shape (T,J,D) where the (t,j)-th entry is in R^D.
        model: joint distribution defined in -> (model.py)
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.

    Returns:
        np.array of shape (T,J,K), where the (t,j,k)-th element gives the log emissions
        probability of the t-th observation (given the (t-1)-st observation)
        for the j-th entity while in the k-th entity-level regime.
        
    """
    if example_end_times is None:
        T = len(observations)
        example_end_times = np.array([-1, T])

    ### Compute log emissions assuming a single example.
    log_entity_emissions = compute_log_entity_emissions_JAX__assuming_single_example(CSP, IP, observations, model)

    ### Patch emissions if there are separate examples.
    return fix__log_emissions_from_entities__at_example_boundaries(
        log_entity_emissions, observations, IP, model, example_end_times
    )


def compute_log_entity_emissions_JAX__assuming_single_example(
    CSP: ContinuousStateParameters_JAX,
    IP: InitializationParameters_JAX,
    observations: JaxNumpyArray3D,
    model: Model,
):
    """
    Purpose: 
        Compute the log (autoregressive, switching) emissions for the observations, where we we must combine:
        initial observation emission:  x_0^j | z_0 =k
        remaining observation emission:  x_t^j | x_(t-1)^j, z_t =k
        for entity-level regimes k=1,...,K and entities j=1,...,J

        This assumes a single example n. 

    Arguments:
        CSP: the observation parameters A[j,k], b[j,k], Q[j,k]
        IP: the initial emission parameters pi_sytem, pi_entities, mu_0s, Sigma_0s
        observations : np.array of shape (T,J,D) where the (t,j)-th entry is in R^D.
        model: joint distribution defined in -> (model.py)

    Returns:
        np.array of shape (T,J,K), where the (t,j,k)-th element gives the log emissions
        probability of the t-th continuous state (given the (t-1)-st continuous state)
        for the j-th entity while in the k-th entity-level regime.
    """

    ### Initial times
    log_pdfs_init_time = model.compute_log_initial_continuous_state_emissions_JAX(
        IP,
        observations[0],
    )
    #### Remaining times
    log_pdfs_remaining_times = model.compute_log_continuous_state_emissions_after_initial_timestep_JAX( #compute log probabilities 
        CSP, observations
    )

    ### Combine them
    log_emissions = jnp.vstack((log_pdfs_init_time[None, :, :], log_pdfs_remaining_times))

    return log_emissions


def run_VEZ_step_JAX(
    CSP: ContinuousStateParameters_JAX,
    ETP: EntityTransitionParameters_MetaSwitch_JAX,
    IP: InitializationParameters_JAX,
    observations: JaxNumpyArray3D,
    VES_expected_regimes: JaxNumpyArray2D,
    model: Model,
    example_end_times: NumpyArray1D,
    outside_recurrence: Optional[JaxNumpyArray3D] = None
) -> HMM_Posterior_Summaries_JAX:
    """
    Purpose:
        Compute the VEZ step for a group dynamics model, and return results in the
        form of HMM_Posterior_Summaries for the j entities. 
        The VEZ step for a group dynamics model can be
        seen as the posterior of a HMM with adjusted parameters (due to taking expectations
        over the other random variables.)  First, we construct the initial distribution
        (actually a function arg), the transitions, and the emissions.
        Then, we feed this information into a generic forward-backward algo,
        namely the `hmm_expected_states` function from Linderman's ssm repo.
        Finally, we return the HMM_Posterior_Summaries. 

    Arguments:
        CSP: the observation parameters A[j,k], b[j,k], Q[j,k]
        ETP: the entity transition parameters Psis has shape (J, L, K, D_e) and Ps has shape (J, L, K, K)
        IP:  the initial emission parameters mu_0s, Sigma_0s
        observations : np.array of shape (T,J,D) where the (t,j)-th entry is
            in R^D.  These can be sampled for the full model (where x's are not observed)
            or observed (if we have a switching
            AR-HMM i.e. a meta-switching recurrent AR model)
        VES_expected_regimes : np.array of size (T,L) whose (t,l)-th element gives
            the probability distribution of system regime l, q(s_t=l).
            May come from a HMM_Posterior_Summary instance created by the VES step.
        model: joint distribution defined in -> (model.py)
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        outside_recurrence: The recurrence features (T-1, D_e) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model. 



    """

    # log_emissions_from_entities  has shape (T,J,K)
    log_emissions_from_entities = compute_log_entity_emissions_JAX( #compute log probabilities 
        CSP,
        IP,
        observations,
        model,
        example_end_times,
    )

    # `transitions` has shape (T-1,J,K,K)
    log_entity_transitions_expected = (
        compute_expected_log_entity_transition_probability_matrices_wrt_system_regimes_JAX( #compute expectations 
            ETP,
            VES_expected_regimes,
            observations,
            model,
            outside_recurrence
        )
    )


    log_entity_transitions_expected = fix_log_entity_transitions_at_example_boundaries(
        log_entity_transitions_expected,
        IP,
        example_end_times,
    )
    return compute_hmm_posterior_summaries_JAX( #Go compute forwards-backwards 
        log_entity_transitions_expected, log_emissions_from_entities, IP.pi_entities
    )
