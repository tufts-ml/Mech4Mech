

import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import multivariate_normal as mvn_JAX
from scipy.stats import multivariate_normal as mvn

from utilities.types import (
    JaxNumpyArray2D,
    JaxNumpyArray3D,
)

from params import (
    ContinuousStateParameters_JAX,
    InitializationParameters_JAX,
)



"""
Functions to compute the emission probability matrices of the HSRDM. 
"""


def compute_log_continuous_state_emissions_after_initial_timestep_JAX(
    CSP: ContinuousStateParameters_JAX,
    continuous_states: JaxNumpyArray3D,
) -> JaxNumpyArray3D:
    """
    Compute the log (autoregressive, switching) emissions for the continuous states, where we have
        x_t^j ~ N( A[j,k] @ x_{t-1}^j + b[j,k] , Q[j,k] )
    for entity-level regimes k=1,...,K and entities j=1,...,J

    Note that we do NOT include the initial state
        x_0^j ~ N( mu_0[j,k], Sigma_0[j,k] )
    which is computed elsewhere.

    Arguments:
        continuous_states : array of shape (T,J,D) where the (t,j)-th entry is
            in R^D

    Returns:
        array of shape (T-1,J,K), where the (t,j,k)-th element gives the log emissions
        probability of the (t+1)-st continuous state (given the (t)-th continuous state)
        for the j-th entity while in the k-th entity-level regime.

    Notation:
        T: number of timesteps
        J: number of entities
        L: number of system-level regimes
        K: number of entity-level regimes
        D: dimension of continuous states
    """
    T = len(continuous_states)
    K = np.shape(CSP.As)[1]

    #### Remaining times
    # We have x_t^j ~ N(A[j,k] @ x_{t-1}^j + b[j,k], Q[j,k])
    # TODO: DO I need to tile the covs and the continuous states?
    means_after_initial_timestep = jnp.einsum("jkde,tje->tjkd", CSP.As, continuous_states[:-1])  # (T-1,J,K,D)
    means_after_initial_timestep += CSP.bs[None, :, :, :]
    covs_after_initial_timestep = jnp.tile(CSP.Qs, (T - 1, 1, 1, 1, 1))  # (T-1,J,K,D, D)
    continuous_states_after_initial_timestep_axes_poorly_ordered = jnp.tile(
        continuous_states[1:], (K, 1, 1, 1)
    )  # (K,T-1,J,D)
    continuous_states_after_initial_timestep = jnp.moveaxis(
        continuous_states_after_initial_timestep_axes_poorly_ordered,
        [0, 1, 2],
        [2, 0, 1],
    )  
    log_pdfs_after_initial_timestep = mvn_JAX.logpdf(
        continuous_states_after_initial_timestep,
        means_after_initial_timestep,
        covs_after_initial_timestep,
    )

    OVERWRITE_FOR_NANS_IN_LOG_EMISSIONS = -1e12
    log_pdfs_after_initial_timestep = jnp.nan_to_num(
        log_pdfs_after_initial_timestep, nan=OVERWRITE_FOR_NANS_IN_LOG_EMISSIONS
    )

    return log_pdfs_after_initial_timestep


def compute_log_initial_continuous_state_emissions_JAX(
    IP: InitializationParameters_JAX,
    initial_continuous_states: JaxNumpyArray2D,
) -> JaxNumpyArray2D:
    """
    Compute the log (autoregressive, switching) emissions for the continuous states at the INITIAL timestep
        x_0^j ~ N( mu_0[j,k], Sigma_0[j,k] )
    for entity-level regimes k=1,...,K and entities j=1,...,J

    Arguments:
        initial_continuous_states : np.array of shape (J,D) where the (j)-th entry is
            in R^D

    Returns:
        np.array of shape (J,K), where the (j,k)-th element gives the log emissions
        probability of the initial continuous state
        for the j-th entity while in the k-th entity-level regime.

    Notation:
        T: number of timesteps
        J: number of entities
        L: number of system-level regimes
        K: number of entity-level regimes
        D: dimension of continuous states
    """

    means_init_time, covs_init_time = IP.mu_0s, IP.Sigma_0s
    log_pdfs_init_time = mvn_JAX.logpdf(initial_continuous_states[:, None, :], means_init_time, covs_init_time)

    return log_pdfs_init_time
