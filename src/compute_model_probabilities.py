import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import multivariate_normal as mvn_JAX
from scipy.stats import multivariate_normal as mvn

from params import (
    ContinuousStateParameters,
    ContinuousStateParameters_JAX,
    EntityTransitionParameters,
    EntityTransitionParameters_JAX,
    InitializationParameters,
    InitializationParameters_JAX,
    SystemTransitionParameters,
    SystemTransitionParameters_JAX,
)
from utilities.types import (
    JaxNumpyArray2D,
    JaxNumpyArray3D,
    JaxNumpyArray5D,
    NumpyArray3D,
    NumpyArray5D,
)
from utilities.util import (
    normalize_log_potentials,
    normalize_log_potentials_by_axis_JAX,
)


# We import from autograd instead of doing `import numpy as np`
# so that we can use use the functions here when doing
# numerical optimization procedures on parameters.


"""
Gives the system transition, entity transitions, 
dynamics, emission, and init functions for the model.

We include both JAX and NUMPY versions.
    * JAX is for AD, and also speed (since it's vectorized)
    * NUMPY is for readability.

We can compare the two in unit tests.
"""

# TODO: For Model 1,  the log system transition probability matrices
# can have recurrent feedback from the entities.  So to generalize
# the model factors, we'll have to update the `compute_log_system_transition_probability_matrices`
# function accordingly.

# TODO: For Model 2a, this function needs to use covariates and Upsilon!.


def compute_log_system_transition_probability_matrices_JAX(
    STP: SystemTransitionParameters_JAX,
    T_minus_1: int,
    system_covariates: Optional[jnp.array] = None,
    x_prevs: Optional[JaxNumpyArray3D] = None,
    system_recurrence_transformation: Callable = None,
):
    """
    Compute log system transition probability matrices.

    These are time varying, but only if at least one of the following conditions are true:
        * system-level covariates exist (in which case the function signature needs to be updated).
            The covariate effect is governed by the parameters in STP.Upsilon
        * there is recurrent feedback from the previous entities (1:J) via x_prev[t-1], as in Model 2a.
            The recurrence effect is also governed by the parameters in STP.Upsilon

    Arguments:
        T_minus_1: The number of timesteps minus 1.  This is used instead of T because the initial
            system regime probabilities are governed by the initial parameters.
        x_prevs: np.array of shape (T-1,J,D) where the (t,j)-th entry is
            in R^D.  These are the previous continuous_states
        system_covariates_prevs: An optional array of shape (T-1, D_s).

    Returns:
        np.array of shape (T-1,L,L).  The (t,l,l')-th element gives the probability of transitioning
            from regime l to regime l' when transitioning into time t+1.

    Notation:
        T: number of timesteps
        L: number of system-level regimes
    """

    if system_covariates is not None and np.prod(np.shape(system_covariates)) != 0:
        raise NotImplementedError("Currently assuming no covariates for skip-level (x-to-s) recurrence.")

    if system_recurrence_transformation is not None and x_prevs is not None:
        system_recurrence_transformation__with_no_covariates = functools.partial(
            system_recurrence_transformation, system_covariates=None
        )

        ### Flatten (T-1,J,D) array to (T-1,JD), where we scroll through j's first, and then d's.
        x_prevs_transposed = jnp.transpose(x_prevs, (0, 2, 1))
        x_prevs_flattened = jax.lax.reshape(
            x_prevs_transposed, (x_prevs_transposed.shape[0], x_prevs_transposed.shape[1] * x_prevs_transposed.shape[2])
        )

        ### Contruct transformation of the above, mapping each (JD,) array to a transformed (D_s,) array.
        # To this, we pre-multiply by the parameter weight matrix Upsilon, which has shape (L, D_s,)
        # In other words, the contribution here biases each of the L destinations differently
        # depending on the values of the (D_s, ) vectors of transfomed skip-level recurrent inputs.

        # x_prevs are (T-1,J,D)... first we flattened to (T-1,JD). Then x_prevs_tildes should be (T-1, D_s)
        x_prevs_tildes = jnp.apply_along_axis(
            system_recurrence_transformation__with_no_covariates,
            1,
            x_prevs_flattened,
        )
        if x_prevs_tildes.ndim == 1:
            x_prevs_tildes = x_prevs_tildes[:, None]

        bias_from_system_recurrence_and_covariates = jnp.einsum("lm,tm->tl", STP.Upsilon, x_prevs_tildes)  # (T-1, L)
    else:
        L = np.shape(STP.Upsilon)[0]
        bias_from_system_recurrence_and_covariates = jnp.zeros((T_minus_1, L))

    # Pi: has shape (L, L)
    log_potentials = (
        bias_from_system_recurrence_and_covariates[:, None, :] + STP.Pi[None, :, :]
    )  # (T-1, None, L) + (None,L,L) = (T-1, L,L)
    return normalize_log_potentials_by_axis_JAX(log_potentials, axis=2)

def compute_log_entity_transition_probability_matrices_JAX(
    ETP_JAX: EntityTransitionParameters_JAX,
    x_prevs: JaxNumpyArray3D,
    transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX: Callable = None,
) -> JaxNumpyArray5D:
    """
    Compute log entity transition probability matrices.

    Arguments:
        ETP_JAX:
            See `EntityTransitionParameters` class definition for more details.
        x_prevs : jnp.array of shape (T-1,J,D) where the (t,j)-th entry is in R^D
            for t=1,...,T-1.   If `sample` is an instance of the `Sample` class, this
            object can be obtained by doing sample.xs[:-1], which gives all the x's except
            the one at the final timestep.
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix: transform R^D -> R^D
            of the continuous state vector before pre-multiplying by the the recurrence matrix.

    Returns:
        jnp.array of shape (T-1,J,L,K,K).  The (t,j,l,k,k')-th element gives the probability of
            the j-th entity transitioning from regime k to regime k'
            when transitioning into time t+1 under the l-th system regime at time t+1.
            That is, it gives P(z_{t+1}^j = k' | z_t^j =k, s_{t+1}=l).
            for t=1,...,T-1.

    Notation:
        T: number of timesteps
        L: number of system-level regimes
        K: number of entity-level regimes
        D: dimension of continuous states
    """
    if transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX is None:
        transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX = lambda x: x
    # TODO: Add covariates
    x_prev_tildes = jnp.apply_along_axis(
        transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX,
        2,
        x_prevs,
    )
    bias_from_recurrence = jnp.einsum("jlkd,tjd->tjkl", ETP_JAX.Psis, x_prev_tildes)  # (T-1, J, K, L)
    bias_from_recurrence_reordered_axes = jnp.moveaxis(bias_from_recurrence, [2, 3], [3, 2])  # (T-1, J, L, K)
    log_potentials = (
        bias_from_recurrence_reordered_axes[:, :, :, None, :] + ETP_JAX.Ps[None, :, :, :, :]
    )  # (T-1, J, L, None, K) + (1,J,L, K,K ) = (T-1, J, L, K, K)
    return normalize_log_potentials_by_axis_JAX(log_potentials, axis=4)


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
