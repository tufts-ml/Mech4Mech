import jax_dataclasses as jdc
import jax.numpy as jnp
import jax.random as jr
import numpy as np

"""
Defines the priors for the transition parameters in the probabilistic models. 
"""

@jdc.pytree_dataclass
class SystemTransitionPrior_JAX:
    """
    Purpose: This prior is for the transition probability matrix parameters Pi (L x L) for system and Ps (K x K) for entity.
    In particular, the prior has independent Dirichlet priors on each of the L or K rows in the matrix (each row sums to 1). 
    where each Dirichlet is ALMOST symmmetric, except that self-transitions are upweighted aka "sticky".

    Attributes: 
        alpha: controls base concentration for transition. 
        kappa: controls how much boost for self-transition (i.e. controls stickyness). 

    Return: the parameters of a "sticky" Dirichlet prior distribution.
    """

    alpha: float
    kappa: float


