import datetime
import warnings
from typing import List, Tuple
import os
import copy
from typing import Optional
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from pathlib import Path
import torch
import tensorflow_probability.substrates.jax.bijectors as tfb
from jax.scipy.special import logsumexp as logsumexp_JAX
from scipy.special import logsumexp

from utilities.types import NumpyArray1D, NumpyArray2D, JaxNumpyArray1D, JaxNumpyArray2D, JaxNumpyArray3D, JaxNumpyArray4D
from model import Model 

"""
Utility functions used throughout the repository. 
"""


###
# Functions to normalize arrays for arrays/matrices (e.g. to simplex, log normalization, matrix normalization by mean and std)
###

def normalize_log_potentials(log_potentials: NumpyArray2D) -> NumpyArray2D:
    """
    Purpose: Normalize the log potentials so that each row of the KxK matrix sums to 1. 

    Arguments:
        log_potentials:  A KxK matrix whose (k,k')-th entry gives the UNNORMALIZED log probability
            of transitioning from state k to state k'

    Returns:
        A KxK matrix whose (k,k')-th entry gives the log probability
            of transitioning from state k to state k'
    """
    log_normalizer = logsumexp(log_potentials, axis=1)
    return log_potentials - log_normalizer[:, None]


def normalize_log_potentials_by_axis(log_potentials: np.array, axis: int) -> np.array:
    """
    Purpose: Normalize the log potentials so that each row of the KxK matrix sums to 1. This
    functions specifies the axis of the matrix to normalize. 

    Arguments:
        log_potentials: A KxK matrix whose (k,k')-th entry gives the UNNORMALIZED log probability
            of transitioning from state k to state k'
        axis: Intended axis to normalize the values. 

    Returns:
        A KxK matrix whose (k,k')-th entry gives the log probability
            of transitioning from state k to state k'
    """
    log_normalizer = logsumexp(log_potentials, axis)
    return log_potentials - np.expand_dims(log_normalizer, axis)


def normalize_log_potentials_by_axis_JAX(log_potentials: jnp.array, axis: int) -> jnp.array:

    """
    Purpose: Normalize the log potentials so that each row of the KxK matrix sums to 1. This
    functions specifies the axis of the matrix to normalize. JAX version. 

    Arguments:
        log_potentials: A KxK matrix whose (k,k')-th entry gives the UNNORMALIZED log probability
            of transitioning from state k to state k'
        axis: Intended axis to normalize the values. 

    Returns:
        A KxK matrix whose (k,k')-th entry gives the log probability
        of transitioning from state k to state k'. Returns a JAX object. 
    """
    log_normalizer = logsumexp_JAX(log_potentials, axis)
    return log_potentials - jnp.expand_dims(log_normalizer, axis)


def normalize_potentials_by_axis(potentials: np.array, axis: int) -> np.array:
    """
    Purpose: Normalize the potentials (NOT originally LOG) so that each row of the KxK matrix sums to 1. This
    functions specifies the axis of the matrix to normalize. 

    Arguments:
        potentials: A KxK matrix whose (k,k')-th entry gives the UNNORMALIZED probability potentials 
        of transitioning from state k to state k'
        axis: Intended axis to normalize the values. 

    Returns:
        A KxK matrix whose (k,k')-th entry gives the log probability
        of transitioning from state k to state k'. 
    """
    log_probs = normalize_log_potentials_by_axis(np.log(potentials), axis)
    return np.exp(log_probs)


def normalize_potentials_by_axis_JAX(potentials: jnp.array, axis: int) -> jnp.array:
    """
    Purpose: Normalize the potentials (NOT originally LOG) so that each row of the KxK matrix sums to 1. This
    functions specifies the axis of the matrix to normalize. Returns a JAX object. 

    Arguments:
        potentials: A KxK matrix whose (k,k')-th entry gives the UNNORMALIZED probability potentials
        of transitioning from state k to state k'
        axis: Intended axis to normalize the values. 

    Returns:
        A KxK matrix whose (k,k')-th entry gives the log probability
        of transitioning from state k to state k'. Returns a JAX object. 
    """
    log_probs = normalize_log_potentials_by_axis_JAX(jnp.log(potentials), axis)
    return jnp.exp(log_probs)


def normalize_matrix_by_mean_and_std_of_columns(arr: NumpyArray2D) -> NumpyArray2D:
    """
    Purpose: Standardize the data by the mean and the standard deviation of the columns. 

    Arguments:
        arr: 2-D array of data (rows, cols)

    Returns:
        A new data array with a subtracted mean of each feature in a column and divided by the std. 
    """

    if arr.ndim != 2:
        raise ValueError
    return (arr - np.mean(arr, 0)) / np.std(arr, 0)

def inv_softplus(x)-> float:
    """
    Purpose: Implements softplus function on the input. 

    Arguments:
        x: scalar input

    Returns:
        Softplus transformation of the scalar input x. 
    """
    return x + torch.log(-torch.expm1(-x))
        

###
# Functions to generate random objects
###


def generate_random_covariance_matrix(dim, var=1.0):
    """
    Purpose: Generates a random covariance matrix. 

    Arguments:
        dim: scalar size of the DxD covariance matrix 
        var: variance of the covariance matrix 

    Returns:
        The covariance matrix. 
    """
    A = np.random.randn(dim, dim) * np.sqrt(var)
    return np.dot(A, A.transpose())


def make_2d_rotation_matrix(theta):
    """
    Purpose: Generates a 2D rotation matrix parameterized by theta. 

    Arguments:
        theta: rotation angle 

    Returns:
        The 2D rotation matrix.  
    """
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def make_2d_rotation_JAX(theta):
    """
    Purpose: Generates a 2D rotation matrix parameterized by theta. JAX instead of NUMPY. 

    Arguments:
        theta: rotation angle 

    Returns:
        The 2D rotation matrix. A JAX object. 
    """
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])


def random_rotation(n, theta=None):
    """
    Purpose: Generate a random rotation matrix of dimension n
        From Linderman's state space modeling repo
        Reference:
            https://github.com/lindermanlab/ssm/blob/master/ssm/util.py#L73-L86

    Arguments:
        theta: rotation angle 
        n: dimension for the rotation

    Returns:
        The n-D rotation matrix. 

    """
    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * np.pi * np.random.rand()

    if n == 1:
        return np.random.rand() * np.eye(1)

    rot = make_2d_rotation_matrix(theta)
    out = np.eye(n)
    out[:2, :2] = rot
    q = np.linalg.qr(np.random.randn(n, n))[0]
    return q.dot(out).dot(q.T)


###
# Functions for manipulating transition probability matrices (TPMs)
###


def make_fixed_sticky_tpm(self_transition_prob: float, num_states: int) -> np.array:
    """
    Purpose: Generates a matrix that prioritizes self-transitions (high values on the diagonals)

    Arguments:
        self_transition_prob: probability of remaining in the current state at the next time-step
        num_states: number of regimes (e.g. L for system states)

    Returns:
        Sticky transition probability matrix.  
    """
    if num_states == 1:
        warnings.warn("Sticky tpm has only 1 state; ignoring self transition prob and creating a `1` matrix.")
        return np.array([[1]])
    external_transition_prob = (1.0 - self_transition_prob) / (num_states - 1)
    return np.eye(num_states) * self_transition_prob + (1.0 - np.eye(num_states)) * external_transition_prob


def make_fixed_sticky_tpm_JAX(self_transition_prob: float, num_states: int) -> jnp.array:
    """
    Purpose: Generates a matrix that prioritizes self-transitions (high values on the diagonals).
    Returns a JAX obejct. 

    Arguments:
        self_transition_prob: probability of remaining in the current state at the next time-step
        num_states: number of regimes (e.g. L for system states)

    Returns:
        Sticky transition probability matrix. JAX object. 
    """
    if num_states == 1:
        warnings.warn("Sticky tpm has only 1 state; ignoring self transition prob and creating a `1` matrix.")
        return jnp.array([[1]])
    external_transition_prob = (1.0 - self_transition_prob) / (num_states - 1)
    return jnp.eye(num_states) * self_transition_prob + (1.0 - jnp.eye(num_states)) * external_transition_prob


def sample_sticky_transition_matrix(K: int, alpha: float, kappa: float, seed: int) -> np.array:
    """
    Purpose: Construct a transition matrix (source x destination) by independently sampling each
    row (which gives probabilities of transitioning into each destination) from a Dirichlet
    disitribution.

    Each Dirichlet is ALMOST symmmetric, except that self-transitions are upweighted, i.e.
        pi_k ~ Dir(alpha * 1_K + kappa * e_k)

    In other words, the Dirichlet on the k-th row is given by (alpha, ..., alpha, alpha+kappa, alpha...,alpha)
    with kappa added to the k-th location, in order to encourage self-transitions.

    Arguments: 
        K: the number of states in the categorical distribution 
        alpha: controls base concentration for transition. 
        kappa: controls how much boost for self-transition (i.e. controls stickyness). 
        seed: for fixed randomness 

    Returns: KxK probability transition matrix sampled from a Dirichlet distribution 
    """
    key = jr.PRNGKey(seed)
    Ps = np.zeros((K, K))
    for k in range(K):
        dirichlet_param = np.ones(K) * alpha
        dirichlet_param[k] += kappa
        Ps[k] = jr.dirichlet(key, dirichlet_param)
        key, _ = jr.split(key)
    return Ps


def evaluate_log_probability_density_of_sticky_transition_matrix_up_to_constant(
    log_P: jnp.array,
    alpha: float,
    kappa: float,
) -> float:
    """
    Purpose: Compute the log probability density of the sticky transition matrix. 

    Arguments:
        log_P: the logarithm of the transition matrix, P, which we want to
            evaluate against a sticky tpm distribution, in particular one
            with independent Dirichlet priors on each of the k=1,...,K rows
            where each Dirichlet is ALMOST symmmetric, except that self-transitions are upweighted, i.e.
            pi_k ~ Dir(alpha * 1_K + kappa * e_k)
        alpha: controls base concentration for transition. 
        kappa: controls how much boost for self-transition (i.e. controls stickyness). 

    Returns: log probability density. 
   """
    K = len(log_P)

    lp = 0
    for k in range(K):
        dirichlet_param = alpha * jnp.ones(K) + kappa * (jnp.arange(K) == k)
        lp += jnp.dot((dirichlet_param - 1), log_P[k])
    return lp


def soften_tpm(tpm_orig: NumpyArray2D) -> NumpyArray2D:
    """
    Purpose: By "softening" a tpm, we mean to bound its entries away from exact 1's or 0's
    by mixing it with a very small amount of a uniform tpm. The motivation is to prevent numerical issues after taking the logarithm.

    Arguments:
        tpm_orig: the original transition probability matrix KxK
        alpha: controls base concentration for transition. 
        kappa: controls how much boost for self-transition (i.e. controls stickyness). 

    Returns: the softened KxK transition probability matrix 
    """

    K = np.shape(tpm_orig)[0]

    # Add in a small bit of a uniform distribution to bound away from exact ones and zeros.
    # A better approach is to use a Dirichlet prior and take the posterior.
    P_UNIFORM = 0.001
    tpm_uniform = np.ones((K, K)) / K

    return P_UNIFORM * tpm_uniform + (1 - P_UNIFORM) * tpm_orig


def unconstrained_tpm_from_tpm(tpm_or_tpms: jnp.array) -> jnp.array:
    """
    Purpose: Represents a tpm over K destinations in unconstrained space R^{K-1} through a bijection
    If X is the unconstrained representation of a pmf and Y is the pmf, we can write
        Y = g(X) = exp([X 0]) / sum(exp([X 0])).
    i.e., we identify the softmax by forcing the last potential to be 0.
    See https://github.com/tensorflow/probability/blob/v0.19.0/tensorflow_probability/python/bijectors/softmax_centered.py#LL34C30-L34C72.

    Arguments:
        tpm_or_tpms: A jnp.array whose last two axes give a  tpm (a transition probability matrix), of size (K,K),
            whose (k,k')-th entry gives the probability of transitioning FROM the k-th state
            to the k'-th state, and where each k-th row is constrained to live on the simplex.

            So for instance, `tpm_or_tpms` could have shape (J,L,K,K),
            where tpm_or_tpms[j,l] gives a KxK tpm.

    Returns:
        A (...,K,K-1) matrix whose values in the last axis are unconstrained reals; there is a bijection betweeen
        the set of (K-1)-vectors and the simplex with K entries.

    """
    if (tpm_or_tpms == 0).any() or (tpm_or_tpms == 1).any():
        raise ValueError(
            "Transition probability matrices cannot be converted to an unconstrained representation if any entry is exactly 0 or 1."
        )

    softmax_bijector = tfb.SoftmaxCentered()
    unconstrained_tpm_list = []
    for row in tpm_or_tpms:
        row_identified = softmax_bijector.inverse(row)
        unconstrained_tpm_list.append(row_identified)
    return jnp.asarray(unconstrained_tpm_list)


def tpm_from_unconstrained_tpm(
    unconstrained_tpm_or_unconstrained_tpms: jnp.array,
) -> jnp.array:
    """
    Purpose: The inverse of `unconstrained_tpm_from_tpm`

    Arguments:
        unconstrained_tpm_or_unconstrained_tpms: A (...,K,K-1) matrix whose values in the last axis are unconstrained reals; there is a bijection betweeen
            the set of (K-1)-vectors and the simplex with K entries.
            
    Returns:
        A jnp.array whose last two axes give a  tpm (a transition probability matrix), of size (K,K),
            whose (k,k')-th entry gives the probability of transitioning FROM the k-th state
            to the k'-th state, and where each k-th row is constrained to live on the simplex.

    """
    softmax_bijector = tfb.SoftmaxCentered()
    tpm_list = []
    for row in unconstrained_tpm_or_unconstrained_tpms:
        row_on_simplex = softmax_bijector.forward(row)
        tpm_list.append(row_on_simplex)
    return jnp.asarray(tpm_list)


###
# Functions for setting up the individual sequence example regimes from the set of overall data sequences. 
###


def convert_list_of_regime_id_and_num_timesteps_to_regime_sequence(
    list_of_regime_id_and_num_timesteps: List[Tuple[int, int]]
) -> NumpyArray1D:

    """
    Purpose: Convert a tuple of regime ids and time-steps to a single list of a regime sequence. 

    Arguments:
        list_of_regime_id_and_num_timesteps: List of tuples(regime id, time-step) 
            
    Returns:
        Regime sequence. 
    """

    regime_sequence = []
    for k, T_slice in list_of_regime_id_and_num_timesteps:
        segment = [k] * T_slice
        regime_sequence.extend(segment)
    return regime_sequence

def make_sample_weights_which_mask_the_initial_timestep_for_each_event(
    observations: JaxNumpyArray3D,
    example_end_times: NumpyArray1D,
    mask_observations: Optional[JaxNumpyArray2D] = None,
) -> JaxNumpyArray2D:
    """
    Purpose: Takes in  the observations, the end times of the event, and a matrix of booleans for each (t,j) pair 
    to inform the model which ones we want masked. If all True, the function just ensures that the returned boolean matrix for each (t,j) pair has a false 
    for each sequence starting point. 

    Arguments:
        observations:  np.array of shape (T-1,J,D) where the (t,j)-th entry is in R^D. 
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.

    Returns: 
        A boolean matrix of True/False for observations we wanted masked. 
    """
    T, J, D = np.shape(observations)

    if mask_observations is None:
        mask_observations = np.full((T, J), True)

    if example_end_times is None:
        T = len(observations)
        example_end_times = np.array([-1, T])

    sample_weights = copy.deepcopy(mask_observations)
    for event_end_idx in example_end_times[:-1]:
        event_start_idx = event_end_idx + 1
        sample_weights[event_start_idx, :] = False
    return sample_weights

def example_end_times_are_proper(example_end_times: NumpyArray1D, T: int) -> Optional[bool]:
    """
    Purpose: Check the valid form of example_end_times. 

    Arguments: 
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.

    Return: 
        A boolean True if the example_end_times are in a valid form. 

    """
    """example_end_times should look like [-1, <bunch of times giving ends of all segments besides the last one>, T]"""
    return example_end_times[0] == -1 and example_end_times[-1] == T


def only_one_example(example_end_times: Optional[NumpyArray1D], T: int) -> Optional[bool]:

    """
    Purpose: Check if there is only ONE example 

    Arguments: 
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.

    Return: 
        A boolean True if there is only ONE example. 

    """
    return (example_end_times is None) or (len(example_end_times) == 2 and (example_end_times == [-1, T]))


def get_initialization_times(example_end_times: NumpyArray1D) -> NumpyArray1D:
    """
   Purpose: Generate a list of the end times for each example. 

    Arguments: 
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
    
    Return: 
        A list of end times for each example up to N
    """
    return np.array(example_end_times[:-1]) + 1


def get_non_initialization_times(example_end_times: NumpyArray1D) -> NumpyArray1D:
    """
    Purpose: Generate a list of the NON end times for each example. 

    Arguments: 
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
    
    Return: 
        A list of all times that are NOT the end time for each example. 
    """
    
    T = example_end_times[-1]
    initialization_times = get_initialization_times(example_end_times)
    return np.array([i for i in range(T) if i not in initialization_times])


###
# Functions for knowing and fixing the modeling probability distributions at sequence example boundaries. 
###


def eligible_transitions_to_next(example_end_times: NumpyArray1D) -> NumpyArray1D:
    """
    Purpose: Generates a numpy array of booleans telling whether each timestep in (0,...,T-1) should be selected when performing
        an inference operation on transitions. If example_end_times=[-1,4,10], this function returns
        array([ True,  True,  True,  True, False,  True,  True,  True,  True]).
        The value at index 4 is False because the transition from 4 to 5 is not eligible for doing inference.
        (It crosses an example boundary.)

    Arguments: 
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
    
    Returns:
        The boolean array. 

    """
    T = example_end_times[-1]
    return np.isin(np.arange(T - 1), example_end_times[1:-1], invert=True)


def fix_log_system_transitions_at_example_boundaries(
    log_system_transitions: JaxNumpyArray4D,
    IP,
    example_end_times: NumpyArray1D,
) -> JaxNumpyArray4D:
    """
    Purpose: Ensure that the log system transitions cannot go past the time segments at each example boundary. 
   
    Arguments:
        log_system_transitions: has shape (T-1, L, L)
        IP: the initial emission parameters pi_sytem, pi_entities, mu_0s, Sigma_0s
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
    
    Return: Fixed log system transition array that has a shape (T-1, L, L) 
    """
    L = np.shape(log_system_transitions)[2]

    log_system_transitions_fixed = np.array(log_system_transitions)


    log_transitions_to_destinations_per_init_dist = np.tile(np.log(IP.pi_system), (L, 1))
    for end_time in example_end_times[1:-1]:
        log_system_transitions_fixed[end_time] = log_transitions_to_destinations_per_init_dist
    return jnp.array(log_system_transitions_fixed)


def fix_log_entity_transitions_at_example_boundaries(
    log_entity_transitions: JaxNumpyArray4D,
    IP,
    example_end_times: NumpyArray1D,
) -> JaxNumpyArray4D:
    """
    Purpose: Ensure that the log entity transitions cannot go past the time segments at each example boundary. 
  
    Arguments:
        log_entity_transitions: has shape (T-1, J, K, K)
        IP: the initial emission parameters pi_sytem, pi_entities, mu_0s, Sigma_0s
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
    
    Return: Fixed log entity transition array that has a shape (T-1, J, K, K)
    """
    _, J, K, _ = np.shape(log_entity_transitions)

    log_entity_transitions_fixed = np.array(log_entity_transitions)

    for j in range(J):
        log_transitions_to_destinations_per_init_dist = np.tile(np.log(IP.pi_entities[j]), (K, 1))
        for end_time in example_end_times[1:-1]:
            log_entity_transitions_fixed[end_time, j] = log_transitions_to_destinations_per_init_dist
    return jnp.array(log_entity_transitions_fixed)


def fix__log_emissions_from_system__at_example_boundaries(
    log_emissions_from_system: JaxNumpyArray2D,
    VEZ_expected_regimes: JaxNumpyArray3D,
    IP,
    example_end_times: NumpyArray1D,
) -> JaxNumpyArray3D:
    """   
    Purpose: Ensure that the log emissions (i.e. the current log probs of the regime)
    of the system cannot go past the time segments at each example boundary. 
  
    Arguments:
        log_emissions_from_system: has shape (T, L)
        VEZ_expected_regimes: posterior summary of maringals has shape (T,J,K)
        IP: the initial emission parameters pi_sytem, pi_entities, mu_0s, Sigma_0s
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
    
    Return: Fixed log emissions from system array that has a shape (T,L)
    """
    L = np.shape(log_emissions_from_system)[1]

    log_emissions_from_system_fixed = np.array(log_emissions_from_system)

    # Reconstruct the initial log emissions from system

    # `iinitial_log_emissions_from_system` has shape (L,) and is obtained by summing over (J,K) objects
    initial_log_emission_for_each_system_regime = jnp.sum(VEZ_expected_regimes * np.log(IP.pi_entities))
    initial_log_emissions_from_system = jnp.repeat(initial_log_emission_for_each_system_regime, L)

    for end_time in example_end_times[1:-1]:
        log_emissions_from_system_fixed[end_time + 1] = initial_log_emissions_from_system

    return jnp.array(log_emissions_from_system_fixed)


def fix__log_emissions_from_entities__at_example_boundaries(
    log_emissions_from_entities: JaxNumpyArray3D,
    observations: JaxNumpyArray3D,
    IP,
    model: Model,
    example_end_times: NumpyArray1D,
) -> JaxNumpyArray3D:
    """
    Purpose: Ensure that the log emissions (i.e. the current log probs of the regime)
    of the entities cannot go past the time segments at each example boundary. 
  
    Arguments:
        log_emissions_from_entities: has shape (T, J, K).  These are the log emissions associated to an entity
            transition function. 
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        VEZ_expected_regimes: posterior summary of maringals has shape (T,J,K)
        IP: the initial emission parameters pi_sytem, pi_entities, mu_0s, Sigma_0s
        model: joint distribution defined in -> (model.py)
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
 
    Return: Fixed log emissions from entity array that has a shape (T,J, K)
    """
  
    log_emissions_from_entities_fixed = np.array(log_emissions_from_entities)

    for end_time in example_end_times[1:-1]:
        log_emissions_from_entities_fixed[end_time + 1] = model.compute_log_initial_continuous_state_emissions_JAX(
            IP, observations[end_time + 1]
        )

    return jnp.array(log_emissions_from_entities_fixed)


###
# Functions for covariance computations
###


def cholesky_nzvals_from_covariance_JAX(Sigma: JaxNumpyArray2D) -> JaxNumpyArray1D:
    """
    Purpose: Returns the non-zero values of the Cholesky factor (which is lower triangular)
    of a covariance matrix - i.e. if Sigma = LL^T, we return the nzvals of L.

    Arguments:
        Sigma: A covariance matrix.

    Returns:
        A 1d array giving the non-zero values of the Cholesky factor (which is lower triangular)
        of a covariance matrix.

    """
    L = jnp.linalg.cholesky(Sigma)
    return L[jnp.tril_indices_from(L)]


def covariance_from_cholesky_nzvals_JAX(
    cholesky_nzvals: JaxNumpyArray1D,
) -> JaxNumpyArray2D:
    """
    Purpose: Returns the covariance matrix from the non-zero values of the Cholesky factor (which is lower triangular).

    Arguments:
        cholesky_nzvals: A 1d array giving the non-zero values of L, the Cholesky factor (which is lower triangular)
        of a covariance matrix.

    Returns:
        A covariance matrix, LL^T
    """

    D = _compute_dim_of_lower_triangular_matrix_from_number_of_nzvals(len(cholesky_nzvals))
    idxs = np.tril_indices(D)
    L_reconstructed = jnp.zeros((D, D), dtype=cholesky_nzvals.dtype).at[idxs].set(cholesky_nzvals)
    return L_reconstructed @ L_reconstructed.T


def _compute_dim_of_lower_triangular_matrix_from_number_of_nzvals(n: int) -> int:
    """
    Purpose: Compute the dimension of a square lower triangular matrix
        if there are `n` nonzero values. For a lower triangular (square) matrix with D rows and columns,
        the number of nonzero values is N=D(D+1)/2. Thus, given N, we can solve for D via the quadratic formula:
        D^2 + D - 2N = 0 gives, taking the positive square root
            D = -1 + sqrt(1+8N)
                ---------------
                    2
    Arguments: 
        n: number of non-zero values in the matrix 

    Return: matrix dimension 
    """
    return int((-1 + np.sqrt(1 + 8 * n)) / 2)


###
# Functions for general array/matrix computations
###

def compute_cartesian_product_of_two_1d_arrays(x_vals: NumpyArray1D, y_vals: NumpyArray1D):
    """
    Purpose: Compute cartesian product of 2 1-D arrays 
    
    Arguments: 
        x_vals: x values 
        y_vals: y values 

    Return: cartesian product. 
    """
    
    x_for_grid, y_for_grid = np.meshgrid(x_vals, y_vals)
    return np.column_stack((x_for_grid.ravel(), y_for_grid.ravel()))  # has shape (len(x)*len(y), D=2)


###
# Functions for manipulating lists
###
def construct_a_new_list_after_removing_multiple_items(orig_list: List, indices_to_remove: List[int]) -> List:
    """
    Purpose: Constructs a new list after removing multiple items from the original list
    
    Arguments: 
        orig_list: list of items
        indices_to_remove: list of indices to remove

    Return: New list without iterms 
    """
    return [item for index, item in enumerate(orig_list) if index not in indices_to_remove]


def are_lists_identical(list_of_lists):
    """
    Purpose: Checks if two lists in a list are identifical
    
    Arguments: 
        list_of_lists): list containing two lists to check 

    Return: Boolean True if two lists are identical 
    """
    
    # Check if the list_of_lists is empty
    if not list_of_lists:
        return True  # Empty lists are considered identical

    # Compare each list to the first list
    first_list = list_of_lists[0]
    for other_list in list_of_lists[1:]:
        if first_list != other_list:
            return False

    return True


def flatten_list_of_lists(list_of_lists):
    """
    Purpose: Flattens lists in a larger to list to all items in a single list 
    
    Arguments: 
        list_of_lists: list containing multiple lists 

    Return: A new list of all items as single elements of that one list 
    """
    return [item for sublist in list_of_lists for item in sublist]

def segment_list(data, indexes):

    """
    Purpose: Creates a list with segmented data at desired indices given data 
    
    Arguments: 
        data: list of elements 
        index: list of indices to segment the data into different lists

    Return: List of lists (segments) containing the data cut-off at the desired indices  
    """
    indexes = sorted(indexes)
    
    indexes = indexes + [len(data)]
    
    segments = []
    start = 0
    
    for idx in indexes:
        segments.append(data[start:idx])
        start = idx
    
    return segments

def find_indices(lst, target):
    """
    Purpose: Find the index of the target value in a list
    
    Arguments: 
        lst: list of elements 
        target: value one wants to know the index of

    Return: Index of the target value in the list 
    """
    """Return the indices where the target integer appears in the list."""
    return [i for i, x in enumerate(lst) if x == target]


###
# Functions to express and save model runs 
###


def get_current_datetime_as_string():
    """
    Purpose: Get the date and time as a string

    Return: String with current date and time  
    """
    return datetime.datetime.now().strftime("%m-%d-%Y_%Hh%Mm%Ss")


def ensure_dir(directory):
    """
    Purpose: Makes sure directory exists before saving to it.

    Arguments: 
        directory: A string naming the directory on the local machine where we will save stuff.
    """
    # alternative: os.makedirs(directory, exist_ok=True)
    if not os.path.isdir(directory):
        os.makedirs(directory)


def prepare_run_directories(run_description: str):
    """
    Purpose: Creates a new run directory structure of the form:
      results/unsupervised_inference/{run_description}/
          plots/
          artifacts/
    
    Arguments: 
        run_description: string of the model run description

    Returns:
        run_dir (Path): Path to the main run directory
        plots_dir (Path): Path to save plots
        artifacts_dir (Path): Path to save model files, configs, etc.
    """

    # Repo root
    repo_root = Path(__file__).resolve().parents[1]

    # Put everything inside results/unsupervised_inference
    base_dir = repo_root / "results" / "unsupervised_inference"

    # Make run folder name unique using timestamp
    run_folder_name = f"{run_description}"

    run_dir = base_dir / run_folder_name
    plots_dir = run_dir / "plots"
    artifacts_dir = run_dir / "artifacts"

    # Create them
    plots_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, plots_dir, artifacts_dir

