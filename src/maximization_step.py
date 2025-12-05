import functools
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import jax.numpy as jnp
import numpy as np
from dynamax.utils.optimize import run_gradient_descent
from statsmodels.regression.linear_model import WLS
from statsmodels.tools.tools import add_constant


from utilities.util import (
    make_sample_weights_which_mask_the_initial_timestep_for_each_event,evaluate_log_probability_density_of_sticky_transition_matrix_up_to_constant, soften_tpm,normalize_log_potentials_by_axis_JAX,
    normalize_potentials_by_axis_JAX,
    eligible_transitions_to_next,
    get_initialization_times,
    get_non_initialization_times,
)
from utilities.types import (
    JaxNumpyArray2D,
    JaxNumpyArray3D,
    JaxNumpyArray5D,
    NumpyArray1D,
    NumpyArray3D,
    NumpyArray2D,
)

from model import Model 
from prior import SystemTransitionPrior_JAX
from params import (
    AllParameters_JAX,
    CSP_Gaussian_with_unconstrained_covariances_from_ordinary_CSP_Gaussian,
    ContinuousStateParameters_Gaussian_WithUnconstrainedCovariances_JAX,
    ContinuousStateParameters_JAX,
    ETP_MetaSwitch_with_unconstrained_tpms_from_ordinary_ETP_MetaSwitch,
    EntityTransitionParameters_MetaSwitch_JAX,
    EntityTransitionParameters_MetaSwitch_WithUnconstrainedTPMs_JAX,
    InitializationParameters_JAX,
    STP_with_unconstrained_tpms_from_ordinary_STP,
    SystemTransitionParameters_JAX,
    SystemTransitionParameters_WithUnconstrainedTPMs_JAX,
    ordinary_CSP_Gaussian_from_CSP_Gaussian_with_unconstrained_covariances,
    ordinary_ETP_MetaSwitch_from_ETP_MetaSwitch_with_unconstrained_tpms,
    ordinary_STP_from_STP_with_unconstrained_tpms,
)

from compute_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
    make_list_from_hmm_posterior_summaries,
    HMM_Posterior_Summary_NUMPY,
    HMM_Posterior_Summaries_NUMPY
)

"""
Executes the maximization step in the CAVI training. 

How it works: 

For each MODEL parameter set (STP, ETP, CSP, IP), we update with respect to the current approximate posteriors 
q(z^{1:J}_{0:T}) and q(s_{0:T}) computed in -> (compute_posterior.py). We either update via closed-form solutions 
when available or via gradient descent of the cost functions. This script includes separate computed cost functions
for the expected log-likelihoods + priors over each individual parameter set with respect to the corresponding
variational posterior. Each of these cost functions can be optimized separately as a part of optimizing the full ELBO. 

STP: Likelihood: E_q log p(s_{1:T}|...) Prior: pi_k ~ Dir(alpha * 1_K + kappa * e_k) -> Parameters estimated via Maximum Posteriori Estimation/Minimize Cross Entropy with gradient descent.  

ETP: Likelihood: E_q log p(z^{1:J}_{1:T}|...) Prior: 1 (uniform) -> Parameters estimated via Maximumum Likelihood Estimation/Minimize Cross Entropy with gradient descent.  

CSP: Likelihood: E_q log p(x^{1:J}_{1:T}|...) Prior: 1 (uniform) -> Parameters estimated via Maximumum Likelihood Estimation/Minimize Weighted Mean Squared Error with closed form. 

IP: E_q log p(s_0) + E_q log p(z_0^{1:J}|...) + E_q log p(x_0^{1:J}|...) 

Why we do it: 
Optimizing the MODEL parameters is akin to optimizing the full Evidence Lower Bound (ELBO) given the observations and 
the current assumed variational posterior. We are essentially finding the variational posterior with its assumed form 
that maximizes the ELBO (i.e. minimizes the negative ELBO). 

"""



class M_Step_Toggle_Value(Enum):

    OFF = 1
    GRADIENT_DESCENT = 2
    CLOSED_FORM_TPM = 3
    CLOSED_FORM_GAUSSIAN = 4


@dataclass
class M_Step_Toggles:
    STP: M_Step_Toggle_Value
    ETP: M_Step_Toggle_Value
    CSP: M_Step_Toggle_Value
    IP: M_Step_Toggle_Value


def M_step_toggles_from_strings(
    STP_toggle: str,
    ETP_toggle: str,
    CSP_toggle: str,
    IP_toggle: str,
) -> M_Step_Toggles:
    """
    Purpose: Describes what kind of M-step should be done for each subclass of parameters:
        gradient-descent, closed-form, or off.

        STP: Closed-form, gradient decent, or off
        ETP: Gradient decent, or off
        CSP: Closed-form, gradient decent, or off (but gradient descent doesn't work very well)
        IP: Closed-form or off
    """
    return M_Step_Toggles(
        STP=M_Step_Toggle_Value[STP_toggle.upper()],
        ETP=M_Step_Toggle_Value[ETP_toggle.upper()],
        CSP=M_Step_Toggle_Value[CSP_toggle.upper()],
        IP=M_Step_Toggle_Value[IP_toggle.upper()],
    )


def compute_variational_posterior_on_regime_triplets_JAX(
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
) -> JaxNumpyArray5D:
    """
    Purpose: Computes the pairwise marginals for the entity latents from the VARIATIONAL summary 
    under each specific system latent l. Multiples system marginals from the system posterior by entity 
    pairwise marginals from the entity posterior. 

    Arguments: 
        VES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions.  
           VES_summary.expected_regimes has shape (T,L)
        VEZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
            VEZ_summaries.expected_regimes has shape (T-1,J,K)
            VEZ_summaries.expected_joints has shape (T-1,J,K, K)

    Returns:
        np.array of shape (T-1,J,L,K,K).  The (t,j,l,k,k')-th element gives the VARIATIONAL
            probability of the the j-th entity transitioning from regime k to regime k'
            when transitioning into time t under the l-th system regime at time t-1.
            That is, it gives q(z_{t}^j = k',  z_{t-1}^j =k) q(s_{t}=l).
            This gives a probability distribution over all triplets (l,k,k').
    """

    return VES_summary.expected_regimes[1:, None, :, None, None] * VEZ_summaries.expected_joints[:, :, None, :, :]


def compute_expected_log_entity_transitions_JAX(
    observations: JaxNumpyArray3D,
    ETP: EntityTransitionParameters_MetaSwitch_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    example_end_times: NumpyArray1D,
    outside_recurrence: Optional[JaxNumpyArray3D] = None,
    mask_observations: Optional[JaxNumpyArray2D] = None,
) -> float:
    """
    Purpose: Computes the expectation of the MODEL log probabilities of the entity transitions under the variational 
    posterior probabilities q(z_{t}^j = k',  z_{t-1}^j =k) q(s_{t}=l). 

    Arguments:
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        ETP: the entity transition parameters Psis has shape (J, L, K, D_e) and Ps has shape (J, L, K, K)
        VES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions. 
        VEZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
        model: joint distribution defined in -> (model.py)
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        outside_recurrence: The recurrence features (T-1, D_e) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model.
        mask_observations: 

    Returns: 
        The sum over the MODEL log probabilities of the entity transitions mutliplied by the variational 
    posterior probabilities.
    """

    T, J = np.shape(observations)[:2]

    if mask_observations is None:
        mask_observations = np.full((T, J), True)

    variational_probs = compute_variational_posterior_on_regime_triplets_JAX(VES_summary, VEZ_summaries)
   
    # `log_transition_matrices` has shape (T-1,J,L,K,K)
    log_transition_matrices = model.compute_log_entity_transition_probability_matrices_JAX(
        ETP,
        T-1,
        observations[:-1],
        model.internal_entity_recurrence_JAX,
        outside_recurrence
    )
    log_transition_matrices_weighted = (
        log_transition_matrices
        * mask_observations[1:, :, None, None, None]
        * eligible_transitions_to_next(example_end_times)[:, None, None, None, None]
    )

    return jnp.sum(variational_probs * log_transition_matrices_weighted)


def compute_expected_log_continuous_state_dynamics_after_initial_timestep_JAX(
    CSP: ContinuousStateParameters_JAX,
    observations: JaxNumpyArray3D,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    example_end_times: NumpyArray1D,
    mask_observations: JaxNumpyArray2D,
) -> float:
   
    """
    Purpose: Computes the expectation of the MODEL log probabilities of the observation emissions under the variational 
    posterior probabilities q(z_{t}^j = k). 

    Arguments:
        CSP: the observation parameters A[j,k], b[j,k], Q[j,k]
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        VEZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
        model: joint distribution defined in -> (model.py)
        example_end_times:optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.

    Returns: 
        The sum over the MODEL log probabilities of the observation emisssions mutliplied by the variational 
    posterior marginal probabilities.
    """

    non_initialization_times = get_non_initialization_times(example_end_times)
    non_initialization_times_shifted_one_index_lower = non_initialization_times - 1

    variational_probs = VEZ_summaries.expected_regimes[non_initialization_times]
    log_continuous_state_dynamics_after_time_zero = (
        model.compute_log_continuous_state_emissions_after_initial_timestep_JAX(
            CSP,
            observations,
        )
    )
    # log_continuous_state_dynamics is (T-1,J,K)

    log_continuous_state_dynamics_weighted = (
        log_continuous_state_dynamics_after_time_zero[non_initialization_times_shifted_one_index_lower]
        * mask_observations[non_initialization_times, :, None]
    )
    return jnp.sum(variational_probs * log_continuous_state_dynamics_weighted)


def compute_expected_log_system_transitions_JAX(
    STP: SystemTransitionParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    model: Model,
    example_end_times: NumpyArray1D,
    outside_recurrence: Optional[JaxNumpyArray2D],
    observations: Optional[JaxNumpyArray3D],
) -> float:
    """
    Purpose: Computes the expectation of the MODEL log probabilities of the system transitions under the variational 
    posterior probabilities q(s_{t}=l', s_(t-1)=l). 

    Arguments:
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        ETP: the entity transition parameters Psis has shape (J, L, K, D_e) and Ps has shape (J, L, K, K)
        VES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions. 
        VEZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
        model: joint distribution defined in -> (model.py)
        example_end_times:optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        outside_recurrence: The recurrence features (T-1, D_s) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model.
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.

    Returns: 
        The sum over the MODEL log probabilities of the system transitions mutliplied by the variational 
    posterior pairwise marginal probabilities.
    """
   
    
    # `variational_probs` has shape (T-1,L,L); entry (t,l,l') gives q(s_{t+1}=l', s_t=1)
    variational_probs = VES_summary.expected_joints
    T_minus_1 = np.shape(variational_probs)[0]

    # ` log_transition_matrices` has shape (T-1,L,L)
    log_transition_matrices = model.compute_log_system_transition_probability_matrices_JAX(
        STP,
        T_minus_1,
        observations=observations[:-1],
        inside_recurrence=model.internal_system_recurrence_JAX,
        outside_recurrence=outside_recurrence
    )
    return jnp.sum(
        variational_probs * log_transition_matrices * eligible_transitions_to_next(example_end_times)[:, None, None]
    )

###
# Compute costs for optimization 
###


def compute_cost_for_entity_transition_parameters_JAX(
    ETP: EntityTransitionParameters_MetaSwitch_JAX,
    observations: JaxNumpyArray3D,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    example_end_times: NumpyArray1D,
    outside_recurrence: Optional[JaxNumpyArray3D] = None,
    mask_observations: Optional[JaxNumpyArray2D] = None,
) -> float:
    """
    Purpose: Computes the cost function for the entity transition parameters, which is the negative 
    expected log likelihood + log prior over the ETP parameters. The expected log likelihood is
    the MODEL log probabilities with respect to the variational posterior q(z_{t}^j = k',  z_{t-1}^j =k) q(s_{t}=l).

    Arguments:
        ETP: the entity transition parameters Psis has shape (J, L, K, D_e) and Ps has shape (J, L, K, K)
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        VES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions. 
        VEZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
        model: joint distribution defined in -> (model.py)
        example_end_times:optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        outside_recurrence: The recurrence features (T-1, D_e) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model.
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.

    Returns: 
        The cost (i.e. energy) for the MODEL likelihood and prior distribution under the variational posterior. 

    """
    T, J = np.shape(observations)[:2]
    if mask_observations is None:
        mask_observations = np.full((T, J), True)

    expected_log_transitions = compute_expected_log_entity_transitions_JAX(
        observations,
        ETP,
        VES_summary,
        VEZ_summaries,
        model,
        example_end_times,
        outside_recurrence,
        mask_observations,
    )
    log_prior = 0.0  # Prior 
    energy = expected_log_transitions + log_prior

    return -energy / jnp.sum(observations)


def compute_cost_for_entity_transition_parameters_with_unconstrained_tpms_JAX(
    ETP_WUC: EntityTransitionParameters_MetaSwitch_WithUnconstrainedTPMs_JAX,
    observations: JaxNumpyArray3D,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    example_end_times: NumpyArray1D,
    outside_recurrence: Optional[JaxNumpyArray3D] = None,
    mask_observations: Optional[JaxNumpyArray2D] = None,
) -> float:
    """
    Purpose: Computes the cost function for the entity transition parameters, which is the negative 
    expected log likelihood + log prior over the ETP parameters - this is just the UNCONSTRAINED version. 
    The expected log likelihood is the MODEL log probabilities with respect to the variational posterior q(z_{t}^j = k',  z_{t-1}^j =k) q(s_{t}=l).

    Arguments:
        ETP_WUC: Unconstrained ETP params
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        VES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions. 
        VEZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
        model: joint distribution defined in -> (model.py)
        example_end_times:optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        outside_recurrence: The recurrence features (T-1, D_e) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model.
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.

    Returns: 
        The cost (i.e. energy) for the MODEL likelihood and prior distribution under the variational posterior. 
    """
    ETP = ordinary_ETP_MetaSwitch_from_ETP_MetaSwitch_with_unconstrained_tpms(ETP_WUC)
    return compute_cost_for_entity_transition_parameters_JAX(
        ETP,
        observations,
        VES_summary,
        VEZ_summaries,
        model,
        example_end_times,
        outside_recurrence,
        mask_observations=mask_observations,
    )


def compute_cost_for_system_transition_parameters_JAX(
    STP: SystemTransitionParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    system_transition_prior: Optional[SystemTransitionPrior_JAX],
    model: Model,
    example_end_times: NumpyArray1D,
    outside_recurrence: Optional[JaxNumpyArray2D],
    observations: Optional[JaxNumpyArray3D],
) -> float:

    """
    Purpose: Computes the cost function for the system transition parameters, which is the negative 
    expected log likelihood + log prior over the STP parameters. The expected log likelihood is the MODEL log probabilities with respect 
    to the variational posterior q(s_{t}=l', s_{t-1}=l).

    Arguments:
        STP: The system state parameters Upsilon has shape (L, D_s) and Pi has shape (L, L)
        VES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions. 
        system_transition_prior: Dirichlet distribution over the categorical parameters in -> (prior.py)
        model: joint distribution defined in -> (model.py)
        example_end_times:optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        outside_recurrence: The recurrence features (T-1, D_s) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model.
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D

    Returns: 
        The cost (i.e. energy) for the MODEL likelihood and prior distribution under the variational posterior. 
    """
    expected_log_transitions = compute_expected_log_system_transitions_JAX(
        STP,
        VES_summary,
        model,
        example_end_times,
        outside_recurrence,
        observations,
    )
    if system_transition_prior is not None:
        log_prior = evaluate_log_probability_density_of_sticky_transition_matrix_up_to_constant(
            normalize_log_potentials_by_axis_JAX(STP.Pi, axis=1),
            system_transition_prior.alpha,
            system_transition_prior.kappa,
        )
    else:
        log_prior = 0.0
    energy_non_constant = expected_log_transitions + log_prior
    T = jnp.shape(VES_summary.expected_regimes)[0]
    return -energy_non_constant / T


def compute_cost_for_system_transition_parameters_with_unconstrained_tpms_JAX(
    STP_WUC: SystemTransitionParameters_WithUnconstrainedTPMs_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    system_transition_prior: Optional[SystemTransitionPrior_JAX],
    model: Model,
    example_end_times: NumpyArray1D,
    outside_recurrence: Optional[JaxNumpyArray2D],
    observations: Optional[JaxNumpyArray3D],
) -> float:
    """
    Purpose: Computes the cost function for the system transition parameters, which is the negative 
    expected log likelihood + log prior over the STP parameters - this is just the UNCONSTRAINED version. 
    The expected log likelihood is the MODEL log probabilities with respect 
    to the variational posterior q(s_{t}=l', s_{t-1}=l).

    Arguments:
        STP_WUC: 
        VES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions. 
        system_transition_prior: Dirichlet distribution over the categorical parameters in -> (prior.py)
        model: joint distribution defined in -> (model.py)
        example_end_times:optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        outside_recurrence: The recurrence features (T-1, D_s) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model.
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D  

    Returns: 
        The cost (i.e. energy) for the MODEL likelihood and prior distribution under the variational posterior. 
    """
    STP = ordinary_STP_from_STP_with_unconstrained_tpms(STP_WUC)
    return compute_cost_for_system_transition_parameters_JAX(
        STP,
        VES_summary,
        system_transition_prior,
        model,
        example_end_times,
        outside_recurrence,
        observations,
    )


def compute_cost_for_continuous_state_parameters_after_initial_timestep_JAX(
    CSP: ContinuousStateParameters_JAX,
    observations: JaxNumpyArray3D,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    example_end_times: NumpyArray1D,
    mask_observations: Optional[JaxNumpyArray2D] = None,
) -> float:
    """
    Purpose: Computes the cost function for the observation emissions, which is the negative 
    expected log likelihood + log prior over the CSP parameters. 
    The expected log likelihood is the MODEL log probabilities with respect 
    to the variational posterior q(z_{t}^j = k).

    Arguments:
        CSP: 
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D 
        VEZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
        system_transition_prior: Dirichlet distribution over the categorical parameters in -> (prior.py)
        model: joint distribution defined in -> (model.py)
        example_end_times:optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.
   
    Returns: 
        The cost (i.e. energy) for the MODEL likelihood and prior distribution under the variational posterior. 
    """

    T, J = np.shape(observations)[:2]
    if mask_observations is None:
        mask_observations = np.full((T, J), True)

    expected_log_state_dynamics = compute_expected_log_continuous_state_dynamics_after_initial_timestep_JAX(
        CSP,
        observations,
        VEZ_summaries,
        model,
        example_end_times,
        mask_observations,
    )
    log_prior = 0.0
    energy = expected_log_state_dynamics + log_prior
    return -energy / jnp.sum(mask_observations)


def compute_cost_for_continuous_state_parameters_with_unconstrained_covariances_after_initial_timestep_JAX(
    CSP_WUC: ContinuousStateParameters_Gaussian_WithUnconstrainedCovariances_JAX,
    observations: JaxNumpyArray3D,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    example_end_times: NumpyArray1D,
    mask_observations: JaxNumpyArray2D,
) -> float:
    """
    Purpose: Computes the cost function for the observation emissions, which is the negative 
        expected log likelihood + log prior over the CSP parameters - this is just the UNCONSTRAINED version. 
        The expected log likelihood is the MODEL log probabilities with respect 
        to the variational posterior q(z_{t}^j = k).

    Arguments:
        CSP_WUC: 
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        VEZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
        system_transition_prior: Dirichlet distribution over the categorical parameters in -> (prior.py)
        model: joint distribution defined in -> (model.py)
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.
   
    Returns: 
        The cost (i.e. energy) for the MODEL likelihood and prior distribution under the variational posterior. 
    """
    CSP = ordinary_CSP_Gaussian_from_CSP_Gaussian_with_unconstrained_covariances(CSP_WUC)
    return compute_cost_for_continuous_state_parameters_after_initial_timestep_JAX(
        CSP,
        observations,
        VEZ_summaries,
        model,
        example_end_times,
        mask_observations,
    )

###
# Run M-steps
###

def run_M_step_for_CSP_in_closed_form__Gaussian_case(
    VEZ_expected_regimes: JaxNumpyArray3D,
    observations: JaxNumpyArray3D,
    example_end_times: NumpyArray1D,
    mask_observations: Optional[JaxNumpyArray2D] = None,
) -> ContinuousStateParameters_JAX:
    """
    Purpose: The M-step for CSP for this model is just the solution for a vector auto-regression (VAR) model
        with weights given by the variational posterior marginal entity probabilities. 

        x_t^j | x_{t-1}^j, z_t^j=k ~ N(A_j^k x_{t-1}^j + b_j^k, Q_j^k)

        To get the parameters for the (j,k)-th entity and entity-regime,
        we weight each sample by the q(z_t^j=k) and compute least squares between: x_t^j and x_{t-1}^j + b_j^k
        across all time steps. These parameters are shared across time-steps, but different for each entity and regime. 

     Arguments:
        VEZ_expected_regimes: has shape (T,J,K) array
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.

    Returns: 
        The CSP parameters. 
    """
    ### Upfront computations
    D = np.shape(observations)[2]
    T, J, K = np.shape(VEZ_expected_regimes)

    ### Make sample weights (as a combo of `mask_observations`` and `example_end_times`).  Shape is (T,J)
    sample_weights = make_sample_weights_which_mask_the_initial_timestep_for_each_event(
        observations,
        example_end_times,
        mask_observations,
    )

    As = np.zeros((J, K, D, D))
    bs = np.zeros((J, K, D))
    Qs = np.zeros((J, K, D, D))

    MIN_SUM_WEIGHTS_TO_UPDATE_PARAMS = 0.5
    for j in range(J):
        xs = np.asarray(observations[:, j, :])
        for k in range(K):
            response_weights = np.asarray(VEZ_expected_regimes[:, j, k] * sample_weights[:, j])[1:]
            sum_of_response_weights = np.sum(response_weights)
            if sum_of_response_weights >= MIN_SUM_WEIGHTS_TO_UPDATE_PARAMS:
                responses = xs[1:]
                predictors = add_constant(xs[:-1], prepend=False)
                wls_model = WLS(responses, predictors, hasconst=True, weights=response_weights)
                results = wls_model.fit()
                # WLS returns parameters where the d-th column gives the weights for predicting d-th element of response vector.
                # So we need to transpose to get a state transition matrix
                As[j, k] = results.params[:-1].T
                bs[j, k] = results.params[-1]
                residuals = results.resid
                # CONFIRM: I need a weighted estimate of covariance if I already used weights to create the wls model.
                Qs[j, k] = np.cov(residuals.T, aweights=response_weights)
            else:
                print(
                    f"\tState {k} for entity {j} has a summed response weights of {sum_of_response_weights:.02f} "
                    "which is insufficient for updating the CSP parameters."
                )

    return ContinuousStateParameters_JAX(jnp.asarray(As), jnp.asarray(bs), jnp.asarray(Qs))


def run_M_step_for_ETP_via_gradient_descent(
    ETP: EntityTransitionParameters_MetaSwitch_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    observations: NumpyArray3D,
    iteration: int,
    num_M_step_iters: int,
    model: Model,
    example_end_times: NumpyArray1D,
    outside_recurrence: Optional[JaxNumpyArray3D] = None,
    mask_observations: Optional[JaxNumpyArray2D] = None,
    verbose: bool = True,
) -> EntityTransitionParameters_MetaSwitch_JAX:
    """
    Purpose: The M-step for ETP with gradient descent optimized on the cost function (likelihood + prior) on the 
    ETP parameters. 

     Arguments:
        ETP: the entity transition parameters Psis has shape (J, L, K, D_e) and Ps has shape (J, L, K, K)
        VES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions. 
        VEZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        iteration: Current iteration for the entire E-M CAVI training 
        num_M_step_iters: number of iterations for optimization (e.g. gradient descent)
        model: joint distribution defined in -> (model.py)
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        outside_recurrence: The recurrence features (T-1, D_e) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model.
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.
        verbose:  True boolean if we want to print loss statements during training 

    Returns: 
        The UNCONSTRAINED ETP parameters. 
    """
    T, J = np.shape(observations)[:2]
    if mask_observations is None:
        mask_observations = np.full((T, J), True)

    ### Do gradient descent on unconstrained parameters.
    ETP_WUC = ETP_MetaSwitch_with_unconstrained_tpms_from_ordinary_ETP_MetaSwitch(ETP)

    cost_function_ETP = functools.partial(
        compute_cost_for_entity_transition_parameters_with_unconstrained_tpms_JAX,
        observations=observations,
        VES_summary=VES_summary,
        VEZ_summaries=VEZ_summaries,
        model= model,
        example_end_times=example_end_times,
        outside_recurrence=outside_recurrence, 
        mask_observations=mask_observations,
    )
    
    optimizer_state_for_entity_transitions = None
    (
        ETP_WUC_new,
        optimizer_state_for_entity_transitions,
        losses_for_entity_transitions,
    ) = run_gradient_descent(
        cost_function_ETP,
        ETP_WUC,
        optimizer_state=optimizer_state_for_entity_transitions,
        num_mstep_iters=num_M_step_iters,
    )

    if verbose:
        print(
            f"For iteration {iteration+1} of the M-step with entity transitions, First 5 Losses are {losses_for_entity_transitions[:5]}. Last 5 losses are {losses_for_entity_transitions[-5:]}"
        )

    return ordinary_ETP_MetaSwitch_from_ETP_MetaSwitch_with_unconstrained_tpms(ETP_WUC_new)


def run_M_step_for_ETP(
    all_params: AllParameters_JAX,
    M_step_toggles_ETP: M_Step_Toggle_Value,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    observations: NumpyArray3D,
    iteration: int,
    num_M_step_iters: int,
    model: Model,
    example_end_times: NumpyArray1D,
    outside_recurrence: Optional[JaxNumpyArray3D] = None,
    mask_observations: Optional[JaxNumpyArray2D] = None,
    verbose: bool = True,
) -> AllParameters_JAX:

    """
    Purpose: Exectutes the M-step for the ETP params based on the setting (e.g. closed_form, gradient_descent)

     Arguments:
        all_params: STP, ETP, CSP, IP
        M_step_toggles_ETP: Toggle value for the optimization setting (e.g. closed_form, gradient_descent)
        VES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions. 
        VEZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        iteration: Current iteration for the entire E-M CAVI training 
        num_M_step_iters: number of iterations for optimization (e.g. gradient descent)
        model: joint distribution defined in -> (model.py)
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        outside_recurrence: The recurrence features (T-1, D_e) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model. 
        verbose:  True boolean if we want to print loss statements during training 

    Returns: 
        All parameters with ETP updated. 
    """
    if M_step_toggles_ETP == M_Step_Toggle_Value.OFF:
        print("Skipping M-step for ETP, as requested.")
        return all_params
    elif M_step_toggles_ETP == M_Step_Toggle_Value.CLOSED_FORM_TPM:
        raise ValueError("Closed-form solution to the M-step for Entity Transition parameters is not available.")
    elif M_step_toggles_ETP == M_Step_Toggle_Value.GRADIENT_DESCENT:
        ### Do gradient descent on unconstrained parameters.
        ETP_new = run_M_step_for_ETP_via_gradient_descent(
            all_params.ETP,
            VES_summary,
            VEZ_summaries,
            observations,
            iteration,
            num_M_step_iters,
            model,
            example_end_times,
            outside_recurrence,
            mask_observations,
            verbose,
        )
    else:
        raise ValueError("I do not know what to do with ETP for the M-step.")

    all_params = AllParameters_JAX(all_params.STP, ETP_new, all_params.CSP, all_params.IP)

    return all_params

def compute_STP_closed_form_M_step(
    posterior_summary: HMM_Posterior_Summary_NUMPY,
    mask_observations: Optional[NumpyArray2D] = None,
    example_end_times: Optional[NumpyArray1D] = None,
) -> NumpyArray2D:
    """
    Purpose: Computes the closed form update for the MODEL parameters of the system transition probabilities
    given the posterior summary (e.g. expected marginals and joints).

    Arguments: 
        posterior_summary: An array containing the latent expected marginals and expected joints, and the log prob
        density for the emissions. 
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.

    Returns:
        Array of shape (L,L) which are the updated MODEL transition probability parameters. 
    """
    T, K = np.shape(posterior_summary.expected_regimes)[:2]

    if mask_observations is None:
        mask_observations = np.full((T), True)

    if example_end_times is None:
        example_end_times = np.array([-1, T])

    # Compute tpm
    tpm_empirical = np.zeros((K, K))
    for k in range(K):
        for k_prime in range(K):
            tpm_empirical[k, k_prime] = np.sum(
                posterior_summary.expected_joints[:, k, k_prime]
                * mask_observations[1:]
                * eligible_transitions_to_next(example_end_times),
                axis=0,
            ) / np.sum(
                posterior_summary.expected_regimes[:-1, k]
                * mask_observations[1:]
                * eligible_transitions_to_next(example_end_times),
                axis=0,
            )

    return soften_tpm(tpm_empirical)

def compute_ETP_closed_form_M_step_on_posterior_summaries(
    posterior_summaries: HMM_Posterior_Summaries_NUMPY,
    mask_observations: Optional[NumpyArray2D] = None,
    example_end_times: Optional[NumpyArray1D] = None,
) -> NumpyArray3D:
    """
    Purpose: Computes the closed form update for the MODEL parameters of the entity transition probabilities
    given the posterior summary (e.g. expected marginals and joints).
    Arguments:
        posterior_summary: An array containing the latent expected marginals and expected joints, and the log prob
        density for the emissions. This function is only used for a bottom initialization of the ETP parameters. 
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.

    Returns:
        Array of shape (J,K,K), whose j-th entry is a tpm
    """

    T, J, K = np.shape(posterior_summaries.expected_regimes)

    if mask_observations is None:
        mask_observations = np.full((T, J), True)

    if example_end_times is None:
        example_end_times = np.array([-1, T])

    posterior_summaries_list = make_list_from_hmm_posterior_summaries(posterior_summaries)

    tpms = [None] * J
    for j in range(J):
        tpms[j] = compute_STP_closed_form_M_step(
            posterior_summaries_list[j], mask_observations[:, j], example_end_times
        )

    return np.array(tpms)



def run_M_step_for_STP_in_closed_form(
    STP: SystemTransitionParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    example_end_times: NumpyArray1D,
) -> SystemTransitionParameters_JAX:

    """
    Purpose: Exectutes the M-step for the STP params in closed form. 

     Arguments:
        STP: The system state parameters Upsilon has shape (L, D_s) and Pi has shape (L, L)
        VES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions. 
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.

    Returns: 
        The STP parameters. 
    """

    warnings.warn("Running closed-form M-step for STP.  Note that this ignores the prior specification.")
    STP_gives_a_TPM = not STP.Upsilon.any()
    if not STP_gives_a_TPM:
        warnings.warn("Using closed-form M step for STP even though STP does not give a TPM!!!")
    exp_Pi = compute_STP_closed_form_M_step(VES_summary, example_end_times=example_end_times)
    Pi_new = jnp.asarray(np.log(exp_Pi))
    return SystemTransitionParameters_JAX(STP.Upsilon, Pi_new)


def run_M_step_for_STP_via_gradient_descent(
    STP: SystemTransitionParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    system_transition_prior: Optional[SystemTransitionPrior_JAX],
    iteration: int,
    num_M_step_iters: int,
    model: Model,
    example_end_times: NumpyArray1D,
    outside_recurrence: Optional[JaxNumpyArray2D],
    observations: Optional[JaxNumpyArray3D],
    verbose: bool = True,
) -> SystemTransitionParameters_JAX:
    
    """
    Purpose: The M-step for STP with gradient descent optimized on the cost function (likelihood + prior) on the 
    STP parameters. 

     Arguments:
        STP: The system state parameters Upsilon has shape (L, D_s) and Pi has shape (L, L)
        VES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions. 
        system_transition_prior: Dirichlet distribution over the categorical parameters in -> (prior.py)
        iteration: Current iteration for the entire E-M CAVI training 
        num_M_step_iters: number of iterations for optimization (e.g. gradient descent)
        model: joint distribution defined in -> (model.py)
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        outside_recurrence: The recurrence features (T-1, D_s) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model.
        observations:  np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        verbose: True boolean if we want to print loss statements during training 

    Returns: 
        The UNCONSTRAINED STP parameters.
    """

    STP_WUC = STP_with_unconstrained_tpms_from_ordinary_STP(STP)
    cost_function_STP = functools.partial(
        compute_cost_for_system_transition_parameters_with_unconstrained_tpms_JAX,
        VES_summary=VES_summary,
        system_transition_prior=system_transition_prior,
        model=model,
        example_end_times=example_end_times,
        outside_recurrence=outside_recurrence,
        observations=observations,
    )

    
    optimizer_state_for_system_transitions = None
    (
        STP_WUC_new,
        optimizer_state_for_system_transitions,
        losses_for_system_transitions,
    ) = run_gradient_descent(
        cost_function_STP,
        STP_WUC,
        optimizer_state=optimizer_state_for_system_transitions,
        num_mstep_iters=num_M_step_iters,
    )

    STP_new = ordinary_STP_from_STP_with_unconstrained_tpms(STP_WUC_new)

    if verbose:
        print(
            f"For iteration {iteration+1} of the M-step with system transitions, First 5 Losses are {losses_for_system_transitions[:5]}. Last 5 losses are {losses_for_system_transitions[-5:]}"
        )
    return STP_new


def run_M_step_for_STP(
    all_params: AllParameters_JAX,
    M_step_toggles_STP: M_Step_Toggle_Value,
    VES_summary: HMM_Posterior_Summary_JAX,
    system_transition_prior: Optional[SystemTransitionPrior_JAX],
    iteration: int,
    num_M_step_iters: int,
    model: Model,
    example_end_times: NumpyArray1D,
    outside_recurrence: Optional[JaxNumpyArray2D],
    observations: Optional[JaxNumpyArray3D],
    verbose: bool = True,
) -> AllParameters_JAX:

    """
    Purpose: Exectutes the M-step for the STP params based on the setting (e.g. closed_form, gradient_descent)

     Arguments:
        all_params: STP, ETP, CSP, IP
        M_step_toggles_STP: Toggle value for the optimization setting (e.g. closed_form, gradient_descent)
        VES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions. 
        system_transition_prior: Dirichlet distribution over the categorical parameters in -> (prior.py)
        iteration: Current iteration for the entire E-M CAVI training 
        num_M_step_iters: number of iterations for optimization (e.g. gradient descent)
        model: joint distribution defined in -> (model.py)
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        outside_recurrence: The recurrence features (T-1, D_s) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model.
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D 
        verbose:  True boolean if we want to print loss statements during training 

    Returns: 
        All parameters with STP updated. 
    """
    if M_step_toggles_STP == M_Step_Toggle_Value.OFF:
        if verbose:
            print("Skipping M-step for STP, as requested.")
        return all_params
    elif M_step_toggles_STP == M_Step_Toggle_Value.CLOSED_FORM_TPM:
        STP_new = run_M_step_for_STP_in_closed_form(all_params.STP, VES_summary, example_end_times)
    elif M_step_toggles_STP == M_Step_Toggle_Value.GRADIENT_DESCENT:
        STP_new = run_M_step_for_STP_via_gradient_descent(
            all_params.STP,
            VES_summary,
            system_transition_prior,
            iteration,
            num_M_step_iters,
            model,
            example_end_times,
            outside_recurrence,
            observations,
            verbose,
        )
    else:
        raise ValueError(
            "I don't understand the specification for how to do the M-step with system transition parameters."
        )

    all_params = AllParameters_JAX(STP_new, all_params.ETP, all_params.CSP,all_params.IP)
    return all_params


def run_M_step_for_CSP(
    all_params: AllParameters_JAX,
    M_step_toggles_CSP: M_Step_Toggle_Value,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    observations: NumpyArray3D,
    iteration: int,
    num_M_step_iters: int,
    model: Model,
    example_end_times: NumpyArray1D,
    mask_observations: Optional[JaxNumpyArray2D],
) -> AllParameters_JAX:
    """
    Purpose: Exectutes the M-step for the CSP params based on the setting (e.g. closed_form, gradient_descent)

     Arguments:
        all_params: STP, ETP, CSP, IP
        M_step_toggles_STP: Toggle value for the optimization setting (e.g. closed_form, gradient_descent)
        VEZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
        observations:  np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        iteration: Current iteration for the entire E-M CAVI training 
        num_M_step_iters: number of iterations for optimization (e.g. gradient descent)
        model: joint distribution defined in -> (model.py)
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.
    
    Returns: 
        All parameters with CSP updated. 
    """

    if M_step_toggles_CSP == M_Step_Toggle_Value.OFF:
        print("Skipping M-step for CSP, as requested.")
        return all_params
    elif M_step_toggles_CSP == M_Step_Toggle_Value.CLOSED_FORM_GAUSSIAN:
        CSP_new = run_M_step_for_CSP_in_closed_form__Gaussian_case(
            VEZ_summaries.expected_regimes,
            observations,
            example_end_times,
            mask_observations,
        )
    elif M_step_toggles_CSP == M_Step_Toggle_Value.GRADIENT_DESCENT:
        warnings.warn(
            f"Learning the CSP parameters by gradient descent.  Performance seems to be worse than with the closed-form approach. "
            f"We should do the analogue to ETP, STP steps which used tensorflow.probability to convert simplex-valued parameters to unconstrained rep and back."
            f"that is, we should rely upon tensorflow.probability to convert covariance parameters to unconstrained representation and back."
        )

        CSP_WUC = CSP_Gaussian_with_unconstrained_covariances_from_ordinary_CSP_Gaussian(all_params.CSP)

        cost_function_CSP = functools.partial(
            compute_cost_for_continuous_state_parameters_with_unconstrained_covariances_after_initial_timestep_JAX,
            observations=observations,
            VEZ_summaries=VEZ_summaries,
            model=model,
            example_end_times=example_end_times,
            mask_observations=mask_observations,
        )

        optimizer_state_for_state_dynamics = None
        (
            CSP_WUC_new,
            optimizer_state_for_state_dynamics,
            losses_for_state_dynamics,
        ) = run_gradient_descent(
            cost_function_CSP,
            CSP_WUC,
            optimizer_state=optimizer_state_for_state_dynamics,
            num_mstep_iters=num_M_step_iters,
        )

        CSP_new = ordinary_CSP_Gaussian_from_CSP_Gaussian_with_unconstrained_covariances(CSP_WUC_new)

        print(
            f"For iteration {iteration+1} of the M-step with continuous state dynamics, First 5 Losses are {losses_for_state_dynamics[:5]}. Last 5 losses are {losses_for_state_dynamics[-5:]}"
        )
    else:
        raise ValueError(
            "I don't understand the specification for how to do the M-step with continuous state parameters."
        )

    all_params = AllParameters_JAX(all_params.STP, all_params.ETP, CSP_new, all_params.IP)
    return all_params


def run_M_step_for_IP_in_closed_form__Gaussian_case(
    IP: InitializationParameters_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    observations: JaxNumpyArray3D,
    example_end_times: NumpyArray1D,
) -> InitializationParameters_JAX:
    """
    Purpose: Exectutes the M-step for the IP params in the closed form Gaussian case. 

     Arguments:
        IP: the initial emission parameters pi_system, pi_entities, mu_0s, Sigma_0s
        VEZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
        VES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions. 
        observations:  np.array of shape (T,J,D) where the (t,j)-th entry isin R^D 
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
    
    Returns: 
        IP parameters. 
    """
 

    init_times = get_initialization_times(example_end_times)

    EPSILON = 1e-3
    # These are set to be the values that minimize the cross-entropy, plus some noise
    expected_system_regime_init_probs = jnp.mean(VES_summary.expected_regimes[init_times], axis=0)
    pi_system = normalize_potentials_by_axis_JAX(expected_system_regime_init_probs + EPSILON, axis=0)

    expected_entity_regime_init_probs = jnp.mean(VEZ_summaries.expected_regimes[init_times], axis=0)
    pi_entities = normalize_potentials_by_axis_JAX(expected_entity_regime_init_probs + EPSILON, axis=1)

    J, K = jnp.shape(pi_entities)

    # set mu_0s to be equal to observed x's.
    empirical_continuous_state_init_means = jnp.mean(observations[init_times], axis=0)  # (J,D)
    # TODO: We are assuming that the initial means are identical across the K regimes.  No reason for this.
    # Take the (expected-regime-)weighted mean above instead of the arithmetic mean.
    mu_0s = jnp.tile(empirical_continuous_state_init_means[:, None, :], (1, K, 1))

    empirical_continuous_state_init_vars = jnp.var(observations[init_times], axis=0)  # (J,D)
    CUTOFF_NUM_OF_INIT_EXAMPLES_TO_USE_ML_ESTIMATE_OF_INIT_VARIANCES = 5
    if len(init_times) < CUTOFF_NUM_OF_INIT_EXAMPLES_TO_USE_ML_ESTIMATE_OF_INIT_VARIANCES:
        # if len(init_idxs)=1, keep Sigma_0s to tbe the same as initialized... not clear how to learn these
        # although could do a Bayesian update (of the prior) even with only one observation.
        Sigma_0s = IP.Sigma_0s

    else:
        D = np.shape(IP.Sigma_0s)[-1]
        Sigma_0s = np.zeros((J, K, D, D))
        # TODO: Vectorize this
        for j in range(J):
            cov_empirical_across_examples = empirical_continuous_state_init_vars[j] * np.eye(D)
            for k in range(K):
                # TODO: We are currently forcing the init covs to be diagonal.  There's no reason for this at all -
                # just implementational haste.  Go back and do it correctly
                #
                # TODO: We are assuming that the initial covs are identical across the K regimes.  No reason for this.
                # Take the (expected-regime-)weighted mean above, instead of the arithmetic mean.
                Sigma_0s[j, k] = cov_empirical_across_examples
    return InitializationParameters_JAX(pi_system, pi_entities, mu_0s, jnp.array(Sigma_0s))


def run_M_step_for_IP(
    IP: InitializationParameters_JAX,
    M_step_toggles_IP: M_Step_Toggle_Value,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    observations: NumpyArray3D,
    example_end_times: NumpyArray1D,
) -> InitializationParameters_JAX:
  
    """
    Purpose: Exectutes the M-step for the IP params based on the setting (e.g. closed_form, gradient_descent)

     Arguments:
        IP: the initial emission parameters pi_system, pi_entities, mu_0s, Sigma_0s
        M_step_toggles_IP: Toggle value for the optimization setting (e.g. closed_form, gradient_descent)
        VES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions. 
        VEZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
        observations:  np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
    
    Returns: 
        All parameters with CSP updated. 
    """
    if M_step_toggles_IP == M_Step_Toggle_Value.OFF:
        print("Skipping M-step for IP, as requested.")
        return IP
    elif M_step_toggles_IP == M_Step_Toggle_Value.CLOSED_FORM_GAUSSIAN:
        IP_new = run_M_step_for_IP_in_closed_form__Gaussian_case(
            IP,
            VEZ_summaries,
            VES_summary,
            observations,
            example_end_times,
        )
    elif M_step_toggles_IP == M_Step_Toggle_Value.GRADIENT_DESCENT:
        raise ValueError(f"Learning the IP parameters by gradient descent is not currently supported.  Try closed-form")
    else:
        raise ValueError(
            "I don't understand the specification for how to do the M-step with continuous state parameters."
        )

    return IP_new
