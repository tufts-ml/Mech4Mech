

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

    OVERWRITE_FOR_NANS_IN_LOG_EMISSIONS = -1e12
    log_pdfs_after_initial_timestep = jnp.nan_to_num(
        log_pdfs_after_initial_timestep, nan=OVERWRITE_FOR_NANS_IN_LOG_EMISSIONS
    )

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

    return log_pdfs_init_time
