import functools
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np
import jax.numpy as jnp
import jax_dataclasses as jdc

from utilities.util import (evaluate_log_probability_density_of_sticky_transition_matrix_up_to_constant, normalize_log_potentials_by_axis_JAX,eligible_transitions_to_next,
    get_initialization_times,
    get_non_initialization_times)
from utilities.types import (
    JaxNumpyArray1D,
    JaxNumpyArray2D,
    JaxNumpyArray3D,
    JaxNumpyArray5D,
    NumpyArray1D,
    NumpyArray2D,
)

from model import Model 
from prior import SystemTransitionPrior_JAX
from params import (
    AllParameters_JAX,
    SystemTransitionParameters_JAX,
    EntityTransitionParameters_MetaSwitch_JAX,
    ContinuousStateParameters_JAX,
    InitializationParameters_JAX,
)
from compute_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
)

"""
Computes the Evidence Lower Bound (ELBO). Not used directly for training, but for monitoring training. 
"""

def calc_elbo(
    all_params: AllParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    system_transition_prior: Optional[SystemTransitionPrior_JAX],
    model: Model,
    observations: JaxNumpyArray3D,
    example_end_times: NumpyArray1D,
    mask_observations: Optional[JaxNumpyArray2D] = None,
    outside_system_recurrence: Optional[JaxNumpyArray2D] = None,
    outside_entity_recurrence: Optional[JaxNumpyArray3D] = None,
    return_dict: bool = False,
) -> float:

    """
    Purpose: Compute the ELBO from the data likelihood term, the prior term, and the entropy terms of the approximate 
    posteriors q(s_{0:T}) and q(z_{0:T}^{1:J})). 

    ELBO = E[log(p(x_{0:T}^{1:J}, z_{0:T}^{1:J}, s_{0:T} | theta))] + E[log(q(z_{0:T}^{1:J}, s_{0:T}))] + log(P(theta)),
    where the expectation E is with respect to q(z_{0:T}^{1:J}, s_{0:T}). 

    Arguments:
        all_params: STP, ETP, CSP, IP
        VES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions.
        VEZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
        system_transition_prior: Dirichlet distribution over the categorical parameters in -> (prior.py)
        model: joint distribution defined in -> (model.py)
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.
        outside_system_recurrence: The recurrence features (T-1, D_s) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model. 
        outside_entity_recurrence: The recurrence features (T-1, D_e) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model. 
        return_dict: False boolean if we don't want to return the dictionary with all of the ELBO component values 
    Returns:
        ELBO computation. 
    """

    elbo_init_dict = calc_energy__init_sys_init_entity_init_data_lik(
        all_params.IP, VES_summary, VEZ_summaries,
        model, observations, example_end_times, mask_observations, True)
    Elogp_s1toT = calc_energy__system_trans(
        all_params.STP, VES_summary,
        model, observations, example_end_times, outside_system_recurrence)
    Elogp_s0toT = Elogp_s1toT + elbo_init_dict['Elogp_s0']
    assert Elogp_s0toT < 1e-7 # expected logpmf of discrete should be negative

    Elogp_z1toT = calc_energy__entity_trans(
        all_params.ETP, VES_summary, VEZ_summaries,
        model, observations, example_end_times, outside_entity_recurrence, mask_observations)
    Elogp_z0toT = Elogp_z1toT + elbo_init_dict['Elogp_z0']
    assert Elogp_z0toT < 1e-7 # expected logpmf of discrete should be negative

    Elogp_x1toT = calc_energy__data_likelihood(
        all_params.CSP, VEZ_summaries,
        model, observations, example_end_times, mask_observations)
    Elogp_x0toT = Elogp_x1toT + elbo_init_dict['Elogp_x0']

    if system_transition_prior is not None:
        logpdf_prior_STP = evaluate_log_probability_density_of_sticky_transition_matrix_up_to_constant(
            normalize_log_potentials_by_axis_JAX(all_params.STP.Pi, axis=1),
            system_transition_prior.alpha,
            system_transition_prior.kappa,
        )
    else:
        logpdf_prior_STP = 0.0

    energy = Elogp_s0toT + Elogp_z0toT + Elogp_x0toT
    entropy_qs = jnp.sum(VES_summary.entropy) # ensures cast to jax
    entropy_qz = jnp.sum(VEZ_summaries.entropies)
    elbo = energy + entropy_qz + entropy_qs + logpdf_prior_STP
    if return_dict:
        return {
            'elbo':elbo,
            'energy':energy, 'entropy':entropy_qs + entropy_qz,
            'logpdf_prior_STP':logpdf_prior_STP,
            'entropy_qs': entropy_qs, 'entropy_qz':entropy_qz,
            'Elogp_s0toT':Elogp_s0toT, 'Elogp_z0toT':Elogp_z0toT,
            'Elogp_x0toT':Elogp_x0toT}
    return elbo

def calc_energy__init_sys_init_entity_init_data_lik(
    IP: InitializationParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    observations: JaxNumpyArray3D,
    example_end_times: NumpyArray1D,
    mask_observations: Optional[JaxNumpyArray2D] = None,
    return_dict: bool = False,
) -> float:

    """
    Purpose: Compute the component of the ELBO that involves the initial states with initial parameters IP. It sums over: 

    ELBO = E[log(p(s_0)] + E[log(p(z_0^{1:J})] +  E[log(p(x_0^{1:J})],
    where the expectation E is with respect to q(z_{0:T}^{1:J}, s_{0:T}). This is summed for all examples. 

    Arguments:
        IP: the initial emission parameters pi_sytem, pi_entities, mu_0s, Sigma_0s
        VES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions.
        VEZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
        model: joint distribution defined in -> (model.py)
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.
        return_dict: False boolean if we don't want to return the dictionary with all of the ELBO component values 
    Returns:
        ELBO computation for the initial states with initial parameters.  
    """

    init_times_V = get_initialization_times(example_end_times)
    V = len(init_times_V)
    J, K = np.shape(IP.pi_entities)

    Es_VL = VES_summary.expected_regimes[init_times_V]
    Ns_L = jnp.sum(Es_VL, axis=0)  # shape (L,)
    logprob_s0_1L = jnp.log(IP.pi_system)[None,:]
    Elogp_s0 = jnp.sum(Ns_L * logprob_s0_1L)

    Ez_TJK = VEZ_summaries.expected_regimes    
    Elogp_z0 = 0.0
    for j in range(J):
        Ezj_VK = Ez_TJK[init_times_V, j]
        Nj_K = jnp.sum(Ezj_VK, axis=0)  # shape (K,)
        Elogp_z0 += jnp.sum(Nj_K * jnp.log(IP.pi_entities[j]))

    Elogp_x0 = 0.0
    for t_init in init_times_V:
        logpdf_x_t_JK = model.compute_log_initial_continuous_state_emissions_JAX(
            IP, observations[t_init]
         )
        Ez_JK = Ez_TJK[t_init]
        Elogp_x0 += jnp.sum(Ez_JK * logpdf_x_t_JK)

    if return_dict:
        return {'Elogp_s0':Elogp_s0, 'Elogp_z0':Elogp_z0, 'Elogp_x0':Elogp_x0}
    return Elogp_s0 + Elogp_z0 + Elogp_x0    


def calc_energy__system_trans(
    STP: SystemTransitionParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    model: Model,
    observations: Optional[JaxNumpyArray3D],
    example_end_times: NumpyArray1D,
    outside_recurrence: Optional[JaxNumpyArray2D] = None
) -> float:
    """ 
    Purpose: Compute the component of the ELBO that involves the system transitions with the STP parameters. It computes: 

    E[log p(s_{0:T} | x_{0:T}^{1:J}, theta)],
    where the expectation E is with respect to q(z_{0:T}^{1:J}, s_{0:T}). This is summed for all examples. 

    Arguments:
        STP: The system state parameters Upsilon has shape (L, D_s) and Pi has shape (L, L)
        VES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions.
        model: joint distribution defined in -> (model.py)
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        outside_recurrence: The recurrence features (T-1, D_s) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model. 
    Returns:
        ELBO computation for the system states with STP parameters.  

    """
    # s_ULL has shape (T-1,L,L)
    # s_ULL[t,l,l'] := q(s_{t+1}=l', s_t=l)
    s_ULL = VES_summary.expected_joints
    T_minus_1 = np.shape(s_ULL)[0]

    # `log_transition_matrices` has shape (T-1,L,L)
    log_trans_prob_ULL = model.compute_log_system_transition_probability_matrices_JAX(
        STP,
        T_minus_1,
        observations=observations[:-1],
        inside_recurrence=model.internal_system_recurrence_JAX,
        outside_recurrence=outside_recurrence
    )
    # Create binary mask of which tsteps are eligible
    elig_bmask_U = eligible_transitions_to_next(example_end_times)
    s_MLL = s_ULL[elig_bmask_U]
    log_trans_MLL = log_trans_prob_ULL[elig_bmask_U]
    return jnp.sum(s_MLL * log_trans_MLL)


def calc_energy__entity_trans(
    ETP: EntityTransitionParameters_MetaSwitch_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    observations: JaxNumpyArray3D,
    example_end_times: NumpyArray1D,
    outside_recurrence: Optional[JaxNumpyArray3D] = None,
    mask_observations: Optional[JaxNumpyArray2D] = None,
) -> float:
    """ 
    
    Purpose: Compute the component of the ELBO that involves the entity transitions with the ETP parameters. It computes: 

    E[log p(z_{0:T}^{1:J} | s_{0:T}^{1:J}, x_{0:T}^{1:J}, theta)],
    where the expectation E is with respect to q(z_{0:T}^{1:J}, s_{0:T}). This is summed for all examples. 

    Arguments:
        ETP: the entity transition parameters Psis has shape (J, L, K, D_e) and Ps has shape (J, L, K, K)
        VES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions.
        VEZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
        model: joint distribution defined in -> (model.py)
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended. 
        outside_recurrence: The recurrence features (T-1, D_e) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model. 
    Returns:
        ELBO computation for the entity states with ETP parameters.  
    """
    T, J, D = np.shape(observations)
    U = T - 1
    # Compute E[ s_t+1=l, z_t = k, z_t+1 = k']
    Eszz_UJLKK = calc_prob_of_regime_triplets_at_adjacent_times_JAX(
        VES_summary, VEZ_summaries)
    # `log_transition_matrices` has shape (T-1,J,L,K,K)
    log_trans_UJLKK = model.compute_log_entity_transition_probability_matrices_JAX(
        ETP,
        T-1,
        observations[:-1],
        model.internal_entity_recurrence_JAX,
        outside_recurrence
    )
    # Mask out sequence boundaries
    elig_bmask_U = eligible_transitions_to_next(example_end_times)
    elig_bmask_U1111 = elig_bmask_U[:, None, None, None, None]
    # Also ignore parts of any entity sequence with no data
    mask_UJ111 = mask_observations[1:, :, None, None, None]
    mask_UJ111 = jnp.logical_and(elig_bmask_U1111, mask_UJ111)
    return jnp.sum(Eszz_UJLKK * (log_trans_UJLKK * mask_UJ111))
    


def calc_energy__data_likelihood(
    CSP: ContinuousStateParameters_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    observations: JaxNumpyArray3D,
    example_end_times: NumpyArray1D,
    mask_observations: Optional[JaxNumpyArray2D] = None,
) -> float:
    """ 

    Purpose: Compute the component of the ELBO that involves the observation emissions transitions with the CSP parameters. 
    It computes: 

    E[log p(x_{0:T}^{1:J} | s_{0:T}^{1:J},z_{0:T}^{1:J}, theta)],
    where the expectation E is with respect to q(z_{0:T}^{1:J}, s_{0:T}). This is summed for all examples. 

    Arguments:
        CSP: the observation parameters A[j,k], b[j,k], Q[j,k]
        VES_summary: contains the posterior summary for the system latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over emissions.
        VEZ_summaries: contains the posterior summary for the entity latent marginals and pairwise marginals
            given the entire observation sequence, and the probability density over the observations. 
        model: joint distribution defined in -> (model.py)
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended. 
    Returns:
        ELBO computation for the observations with CSP parameters.  
    """
    T, J, D = np.shape(observations)
    
    non_init_times_V = get_non_initialization_times(example_end_times)
    non_init_times_lower_index_V = non_init_times_V - 1
    V = len(non_init_times_V)

    # Compute probability of each non-init time t assigned to each state k
    Ez_VJK = VEZ_summaries.expected_regimes[non_init_times_V]

    # Compute likelihood of each data obs under each state k
    # We compute the AR likelihood of t=1 given t=0, t=2 given t=1, etc.
    # Thus, indices need to be shifted down by one
    logpdf_UJK = model.compute_log_continuous_state_emissions_after_initial_timestep_JAX(
        CSP, observations)
    logpdf_VJK = logpdf_UJK[non_init_times_lower_index_V]

    # Ignore parts of any entity sequence with no data
    mask_VJ1 = mask_observations[non_init_times_V, :, None]
    return jnp.sum(Ez_VJK * (mask_VJ1 * logpdf_VJK))


def calc_prob_of_regime_triplets_at_adjacent_times_JAX(
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    assert_valid_output: bool = True, # turn off to go fast
) -> JaxNumpyArray5D:
    """
    Purpose: Multiplies the system posterior marginals by the entity posterior pairwise marginals 
    to obtain the entity transition distributions under the system state l. 

    Arguments: 
        VES_summary:
        VEZ_summaries: 
        assert_valid_output: boolean to check the matrix has valid KxK rows that sum to 1 for each l. 

    Returns:
        q_UJLKK : np.array of shape (T-1,J,L,K,K).  
            entry (t,j,l,k,k') := q( z^j_t = k, z^j_{t+1} = k', s^j_{t+1} = l)
            Probability of j-th entity transition, when time goes from t to t+1,
            transitioning from its regime k to k' under system state l
    """
    sys_r_U1L11 = VES_summary.expected_regimes[1:, None, :, None, None]
    ent_s_UJ1KK = VEZ_summaries.expected_joints[:, :, None, :, :]
    q_UJLKK = sys_r_U1L11 * ent_s_UJ1KK
    if assert_valid_output:
        assert np.allclose(jnp.sum(q_UJLKK,axis=(2,3,4)), 1.)
    return q_UJLKK
