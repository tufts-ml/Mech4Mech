import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import multivariate_normal as mvn_JAX
from scipy.stats import multivariate_normal as mvn

from utilities.util import (
    normalize_log_potentials_by_axis_JAX,
)
from utilities.types import (
    JaxNumpyArray3D,
    JaxNumpyArray5D,
)

from params import (
    EntityTransitionParameters_JAX,
    SystemTransitionParameters_JAX,
)

"""
Functions to compute the tranition probability matrices of the HSRDM. 
"""



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
