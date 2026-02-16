

import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import multivariate_normal as mvn_JAX
from scipy.stats import multivariate_normal as mvn
from typing import Optional

from utilities.types import (
    JaxNumpyArray2D,
    JaxNumpyArray3D,
    NumpyArray3D,
)

from utilities.util import sample_emissions_for_transition_types

from params import (
    ContinuousStateParameters_JAX,
    InitializationParameters_JAX,
)



"""
Functions to compute the log emission probability densities within the factorized joint distribution for the HSRDM. 
Given the current parameters of the gaussian distributions and the observations, the log probability densities are computed. 

Includes the assumptions about all emission dynamics. 
- All observations are assumed to be distributed via multi-variate Gaussian distributions 
- All entities and entity regimes have distributions with their own parameters 
- For each entity in regime k, auto-regressive means and full-rank covariances are computed 
"""


def compute_log_continuous_state_emissions_after_initial_timestep_JAX(
    CSP: ContinuousStateParameters_JAX,
    observations: JaxNumpyArray3D,
    one_hot_evidence: Optional[NumpyArray3D] = None,
    save_dir: Optional[str] = None,
    iteration: Optional[int] = None, 
    table_save: Optional[bool] = False,
) -> JaxNumpyArray3D:
    """
    Purpose: Compute the log (autoregressive, switching) emissions for the observations, where we have
        x_t^j ~ N( A[j,k] @ x_{t-1}^j + b[j,k] , Q[j,k])
        for entity-level regimes k=1,...,K and entities j=1,...,J
        LITERALLY computes the log Gaussian PDF of x_t^j for each A[j,k], b[j,k], Q[j,k].

    Arguments:
        CSP: the observation parameters A[j,k], b[j,k], Q[j,k]
        observations: array of shape (T,J,D) where the (t,j)-th entry is
        in R^D.
        obervations: np.array of shape (T,J,D) where the (t,j)-th entry is in R^D.
        one_hot_evidence: One-hot evidence labels of shape (T, J, C).
            Class 0 is assumed to mean "no evidence".
        save_dir: directory for saving the csv table
        iteration: training iteration; only necessary when saving the csv table
        table_save: Flag for if True, save the CSV table 

    Returns:
        array of shape (T-1,J,K), where the (t,j,k)-th element gives the log emissions
        probability of the (t+1)-st observation (given the (t)-th observation)
        for the j-th entity while in the k-th entity-level regime.

    """
    T = len(observations)
    K = np.shape(CSP.As)[1]

    means_after_initial_timestep = jnp.einsum("jkde,tje->tjkd", CSP.As, observations[:-1]) 
    means_after_initial_timestep += CSP.bs[None, :, :, :]
    covs_after_initial_timestep = jnp.tile(CSP.Qs, (T - 1, 1, 1, 1, 1))  
    observations_after_initial_timestep_axes_poorly_ordered = jnp.tile(
        observations[1:], (K, 1, 1, 1)
    )  
    observations_after_initial_timestep = jnp.moveaxis(
        observations_after_initial_timestep_axes_poorly_ordered,
        [0, 1, 2],
        [2, 0, 1],
    )  
    log_pdfs_after_initial_timestep = mvn_JAX.logpdf(
        observations_after_initial_timestep,
        means_after_initial_timestep,
        covs_after_initial_timestep,
    )

    K_SPECIALS = jnp.array([0, 1])  # <-- your two special states

    silent_observation = observations[0][1]
    all_silent_obs = jnp.all(
        observations[1:] == silent_observation[None, None, :],
        axis=-1
    )
   # all_silent_obs is (T-1, J) because you used observations[1:]
    all_silent_obs_TJK = all_silent_obs[..., None]   # (T-1, J, 1)
    non_silent_TJK = ~all_silent_obs_TJK             # (T-1, J, 1)

    K = log_pdfs_after_initial_timestep.shape[-1]

    is_special = jnp.isin(jnp.arange(K), K_SPECIALS)     # (K,)
    is_non_special = ~is_special                         # (K,)

    is_special = is_special[None, None, :]               # (1, 1, K)
    is_non_special = is_non_special[None, None, :]       # (1, 1, K)

    BIG_NEG = -1e12
    LOG_BOOST = 1e6

    # # 1) If silent => punish NON-special states
    # log_pdfs_after_initial_timestep = jnp.where(
    #     all_silent_obs_TJK & is_non_special,
    #     BIG_NEG,
    #     log_pdfs_after_initial_timestep
    # )

# # 2) If non-silent => punish SPECIAL states
#     log_pdfs_after_initial_timestep = jnp.where(
#         non_silent_TJK & is_special,
#         BIG_NEG,
#         log_pdfs_after_initial_timestep
#     )

    # # Enforce k=2, k=3 to be equal at time t+1 where a speaker at t+1 has no evidence in the utterance of mechanism 
    # #but there was evidence from the speaker at time t. 
    # if one_hot_evidence is not None and log_pdfs_after_initial_timestep.shape[-1] >= 4:
    #     K2, K3 = 2, 3
    #     CONST_LOG_EMIT = 0.0

    #     # evidence_present[t,j] = True iff any evidence class > 0 (exclude class 0)
    #     evidence_present = jnp.any(one_hot_evidence[..., 1:] > 0, axis=-1)  # (T, J)

    #     # non-silent at time t
    #     silent_observation = observations[0][1]
    #     is_silent_tj = jnp.all(observations == silent_observation[None, None, :], axis=-1)  # (T, J)
    #     non_silent_tj = ~is_silent_tj  # (T, J)

    #     J = evidence_present.shape[1]

    #     # Identify the (assumed unique) speaking speaker at each time t
    #     prev_spk = jnp.argmax(non_silent_tj[:-1].astype(jnp.int32), axis=1)  # (T-1,)
    #     next_spk = jnp.argmax(non_silent_tj[1:].astype(jnp.int32), axis=1)   # (T-1,)

    #     # Gather evidence/no-evidence for those speakers
    #     prev_has_evidence = jnp.take_along_axis(evidence_present[:-1], prev_spk[:, None], axis=1).squeeze(1)  # (T-1,)
    #     next_has_evidence = jnp.take_along_axis(evidence_present[1:],  next_spk[:, None], axis=1).squeeze(1)  # (T-1,)

    #     # Also ensure they are actually non-silent (turn exists)
    #     prev_is_speaking = jnp.take_along_axis(non_silent_tj[:-1], prev_spk[:, None], axis=1).squeeze(1)      # (T-1,)
    #     next_is_speaking = jnp.take_along_axis(non_silent_tj[1:],  next_spk[:, None], axis=1).squeeze(1)      # (T-1,)

    #     # Condition per transition t -> t+1:
    #     # evidence at time t from prev speaker AND no evidence at time t+1 from next speaker,
    #     # and they must be different speakers
    #     cond_t = (
    #         prev_is_speaking
    #         & next_is_speaking
    #         & prev_has_evidence
    #         & (~next_has_evidence)
    #         & (prev_spk != next_spk)
    #     )  # (T-1,)

    #     # Apply ONLY to the next-turn speaker at time t+1 (emission index t)
    #     mask_tj = cond_t[:, None] & (jnp.arange(J)[None, :] == next_spk[:, None])  # (T-1, J)

    #     # Set k=2 and k=3 emissions to a constant where mask is true
    #     log_pdfs_after_initial_timestep = log_pdfs_after_initial_timestep.at[..., K2].set(
    #         jnp.where(mask_tj, CONST_LOG_EMIT, log_pdfs_after_initial_timestep[..., K2])
    #     )
    #     log_pdfs_after_initial_timestep = log_pdfs_after_initial_timestep.at[..., K3].set(
    #         jnp.where(mask_tj, CONST_LOG_EMIT, log_pdfs_after_initial_timestep[..., K3])
    #     )

    # Enforce k=2, k=3 to be equal at time t+1 where a speaker at t+1 has evidence in the utterance of mechanism 
    #but there was no evidence from the speaker at time t. 
    # if one_hot_evidence is not None and log_pdfs_after_initial_timestep.shape[-1] >= 4:
    #     K2, K3 = 2, 3
    #     T, J, C = one_hot_evidence.shape

    #     # mech evidence: any class beyond index 0 is 1
    #     mech_tj = jnp.any(one_hot_evidence[..., 1:] > 0, axis=-1)   # (T, J)
    #     no_ev_tj = one_hot_evidence[..., 0] > 0                     # (T, J)

    #     # You already computed non_silent_tj as (T, J)
    #     # non_silent_tj = ~is_silent_tj

    #     # Identify the (assumed unique) speaker for each time t
    #     prev_spk = jnp.argmax(non_silent_tj[:-1].astype(jnp.int32), axis=1)  # (T-1,)
    #     next_spk = jnp.argmax(non_silent_tj[1:].astype(jnp.int32), axis=1)   # (T-1,)

    #     # Gather booleans for the identified speakers
    #     prev_no_ev = jnp.take_along_axis(no_ev_tj[:-1], prev_spk[:, None], axis=1).squeeze(1)          # (T-1,)
    #     prev_non_silent = jnp.take_along_axis(non_silent_tj[:-1], prev_spk[:, None], axis=1).squeeze(1)# (T-1,)

    #     next_mech = jnp.take_along_axis(mech_tj[1:], next_spk[:, None], axis=1).squeeze(1)             # (T-1,)
    #     next_non_silent = jnp.take_along_axis(non_silent_tj[1:], next_spk[:, None], axis=1).squeeze(1) # (T-1,)

    #     # Condition per transition t -> t+1
    #     cond_t = prev_no_ev & prev_non_silent & next_mech & next_non_silent & (prev_spk != next_spk)   # (T-1,)

    #     # Expand to (T-1, J): apply only to the NEXT speaker at time t+1
    #     mask_tj = cond_t[:, None] & (jnp.arange(J)[None, :] == next_spk[:, None])  # (T-1, J)

    #     # Enforce equality between k=2 and k=3 at those (t,j) locations
    #     # Option A: copy k=2 into k=3
    #     log_pdfs_after_initial_timestep = log_pdfs_after_initial_timestep.at[..., K3].set(
    #         jnp.where(mask_tj,
    #                 log_pdfs_after_initial_timestep[..., K2],
    #                 log_pdfs_after_initial_timestep[..., K3])
    #     )

    OVERWRITE_FOR_NANS_IN_LOG_EMISSIONS = -1e12
    log_pdfs_after_initial_timestep = jnp.nan_to_num(
        log_pdfs_after_initial_timestep, nan=OVERWRITE_FOR_NANS_IN_LOG_EMISSIONS
    )


    log_pdfs_after_initial_timestep = jnp.where(
        non_silent_TJK & is_non_special,
        log_pdfs_after_initial_timestep + LOG_BOOST,
        log_pdfs_after_initial_timestep
    )


# Save to a CSV table if boolean flag 
    if table_save == True: 
        sample_emissions_for_transition_types(observations = observations, one_hot_evidence = one_hot_evidence, log_emissions = log_pdfs_after_initial_timestep, iteration = iteration, save_dir = save_dir)

    return log_pdfs_after_initial_timestep


def compute_log_initial_continuous_state_emissions_JAX(
    IP: InitializationParameters_JAX,
    initial_observations: JaxNumpyArray2D,
) -> JaxNumpyArray2D:
    """
    Purpose: Computes the log (autoregressive, switching) emissions for the observations at the INITIAL timestep.
        x_0^j ~ N( mu_0[j,k], Sigma_0[j,k] )
        for entity-level regimes k=1,...,K and entities j=1,...,J
        LITERALLY computes the log Gaussian PDF of x_0^j for each A[j,k], b[j,k], Q[j,k].

    Arguments:
        IP: the initial emission parameters pi_system, pi_entities, mu_0s, Sigma_0s
        initial_observations : np.array of shape (J,D) where the (j)-th entry is in R^D

    Returns:
        np.array of shape (J,K), where the (j,k)-th element gives the log emissions
        probability of the initial observations
        for the j-th entity while in the k-th entity-level regime.
    """

    means_init_time, covs_init_time = IP.mu_0s, IP.Sigma_0s
    log_pdfs_init_time = mvn_JAX.logpdf(initial_observations[:, None, :], means_init_time, covs_init_time)

    K_SPECIAL = 0 #Make it such that for each silent observation and each state that is not k=0, the probability is very small 
    silent_observation = initial_observations[1]
    is_silent = jnp.all(initial_observations == silent_observation, axis=-1)
    K = log_pdfs_init_time.shape[1]
    is_non_special_state = (jnp.arange(K) != K_SPECIAL)  # (K,)
    
    mask_JK = is_silent[:, None] & is_non_special_state[None, :]

    BIG_NEG = -1e10

    log_pdfs_init_time = jnp.where(
    mask_JK,
    BIG_NEG,                 # kill log-probabilities
    log_pdfs_init_time       # leave everything else unchanged
)

    return log_pdfs_init_time
