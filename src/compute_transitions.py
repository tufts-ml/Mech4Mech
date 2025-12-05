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
    EntityTransitionParameters_MetaSwitch_JAX,
    SystemTransitionParameters_JAX,
)

"""
Functions to compute the log transition probabilities for the system and entity latents within the 
factorized joint distribution for the HSRDM. 

Given the current parameters of the log transition probability matrix, the recurrence outputs,
and the observations, the MODEL log transition probabilities are computed. 

Includes the assumptions about all transition dynamics. 
- All transition distributions are categorical with recurrence via some function over the observations
- Markov assumption for all transitions 
"""


def compute_log_system_transition_probability_matrices_JAX(
    STP: SystemTransitionParameters_JAX,
    T_minus_1: int,
    observations: Optional[JaxNumpyArray3D] = None,
    inside_recurrence: Callable = None,
    outside_recurrence: Optional[JaxNumpyArray3D] = None
):
    """
    Purpose: Compute log system transition probability matrices: s_t | s_(t-1), x^(1:J)_(t-1) for each s_t = l, s_(t-1) = l'

    LITERALLY adds the log transition parameters to the recurrence outputs and then normalizes for the distribution. 

    Arguments:
        STP: The system state parameters Upsilon (L, D_s)  and Pi (L, L)
        T_minus_1: The number of timesteps minus 1.  This is used instead of T because the initial
            system regime probabilities are governed by the initial parameters.
        observations: np.array of shape (T-1,J,D) where the (t,j)-th entry is in R^D. 
        inside_recurrence: The transformation function for the recurrent feedback (functions found in recurrence.py). 
            These functions are called inside the JAX tracer of the continuous states/observations, and are computed on the states as
            tracer objects. Thus, these functions can only work with JAX tracer objects. 
        outside_recurrence: The recurrence features (T-1, D_s) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model. 

    Returns:
        np.array of shape (T-1,L,L).  The (t,l,l')-th element gives the probability of transitioning
            from regime l to regime l' when transitioning into time t+1.
    """

    if inside_recurrence is not None and observations is not None:

        ### Flatten (T-1,J,D) array to (T-1,JD), where we scroll through j's first, and then d's.
        observations_transposed = jnp.transpose(observations, (0, 2, 1))
        observations_flattened = jax.numpy.reshape(
            observations_transposed, (observations_transposed.shape[0], observations_transposed.shape[1] * observations_transposed.shape[2])
        )

        # observations are (T-1,J,D)... first we flattened to (T-1,JD). Then observations_tildes should be (T-1, D_s)
        observations_tildes = jnp.apply_along_axis(
            inside_recurrence,
            1,
            observations_flattened,
        )
        if observations_tildes.ndim == 1:
            observations_tildes = observations_tildes[:, None]

        bias_from_system_recurrence = jnp.einsum("lm,tm->tl", STP.Upsilon, observations_tildes)  # (T-1, L)

    elif outside_recurrence is not None: 
        bias_from_system_recurrence= jnp.einsum("lm,tm->tl", STP.Upsilon, outside_recurrence)
    else:
        L = np.shape(STP.Upsilon)[0]
        bias_from_system_recurrence = jnp.zeros((T_minus_1, L))

    # Pi: has shape (L, L)
    log_potentials = (
        bias_from_system_recurrence[:, None, :] + STP.Pi[None, :, :]
    )  
    return normalize_log_potentials_by_axis_JAX(log_potentials, axis=2)

def compute_log_entity_transition_probability_matrices_JAX(
    ETP_JAX: EntityTransitionParameters_MetaSwitch_JAX,
    T_minus_1: int,
    observations: JaxNumpyArray3D,
    inside_recurrence: Callable = None,
    outside_recurrence: Optional[JaxNumpyArray3D] = None
) -> JaxNumpyArray5D:
    """
    Purpose: Compute log entity transition probability matrices: z^j_t | z^j_(t-1), x^(j)_(t-1), s_t for each z^j_t = k, z_(t-1) = k'

    Arguments:
        ETP_JAX: the entity transition parameters Psis has shape (J, L, K, D_e) and Ps has shape (J, L, K, K)
        T_minus_1: The number of timesteps minus 1.  This is used instead of T because the initial
            system regime probabilities are governed by the initial parameters.
        observations: np.array of shape (T-1,J,D) where the (t,j)-th entry is in R^D.  
       inside_recurrence: The transformation function for the recurrent feedback (functions found in recurrence.py). 
            These functions are called inside the JAX tracer of the continuous states/observations, and are computed on the states as
            tracer objects. Thus, these functions can only work with JAX tracer objects. 
        outside_recurrence: The recurrence features (T-1, J, D_e) are provided, which are computed from the observations/continuous states
            outside of the JAX tracer environment. This is useful for when the recurrence function is a pre-trained pytorch model. 

    Returns:
        jnp.array of shape (T-1,J,L,K,K).  The (t,j,l,k,k')-th element gives the probability of
            the j-th entity transitioning from regime k to regime k'
            when transitioning into time t+1 under the l-th system regime at time t+1.
            That is, it gives P(z_{t+1}^j = k' | z_t^j =k, s_{t+1}=l).
            for t=1,...,T-1.
    """
    if inside_recurrence is None and outside_recurrence is None:
        K = np.shape(ETP.Psis)[2]
        bias_from_recurrence = jnp.zeros((T_minus_1, K))

    elif inside_recurrence is not None: 
        x_prev_tildes = jnp.apply_along_axis(
            inside_recurrence,
            2,
            observations,
        )     
        bias_from_recurrence = jnp.einsum("jlkd,tjd->tjkl", ETP_JAX.Psis, x_prev_tildes)  # (T-1, J, K, L)

    elif outside_recurrence is not None: 
        bias_from_recurrence = jnp.einsum("jlkd,tjd->tjkl", ETP_JAX.Psis, outside_recurrence)

    bias_from_recurrence_reordered_axes = jnp.moveaxis(bias_from_recurrence, [2, 3], [3, 2])  # (T-1, J, L, K)
    log_potentials = (
        bias_from_recurrence_reordered_axes[:, :, :, None, :] + ETP_JAX.Ps[None, :, :, :, :]
    )  # (T-1, J, L, None, K) + (1,J,L, K,K ) = (T-1, J, L, K, K)
    return normalize_log_potentials_by_axis_JAX(log_potentials, axis=4)
