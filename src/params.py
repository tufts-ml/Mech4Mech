import os
import pickle
import warnings
from dataclasses import dataclass
from typing import Union
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from jax import vmap

from utilities.util import (
    normalize_log_potentials_by_axis_JAX,
    tpm_from_unconstrained_tpm,
    unconstrained_tpm_from_tpm,
    cholesky_nzvals_from_covariance_JAX,
    covariance_from_cholesky_nzvals_JAX,
)
from utilities.types import (
    JaxNumpyArray1D,
    JaxNumpyArray2D,
    JaxNumpyArray3D,
    JaxNumpyArray4D,
    NumpyArray1D,
    NumpyArray2D,
    NumpyArray3D,
    NumpyArray4D,
)

from model import Model


"""
Defines all MODEL parameters that are used to compute the log probability distributions in -> (compute_emissions.py) and 
(compute_transitions.py), and then trained via the maximization step -> (maximization_step.py). 
Both Numpy and JAX param versions are available. 

Notation:
    T: number of time-steps 
    J: number of entities
    K: number of entity-level regimes
    L: number of system-level regimes
    D: dimensionality of the continuous observation x
    D_s: dimensionality of contribution to the system level transitions from recurrence
    D_e: dimensionality of the contribution to the entity level transitions from recurrence
    MVN: multi-variate normal
    x_t^j: observations
    s_t = l: system latent regimes 
    z_t^j = k: entity latent regimes 
    q(s_{0:T},z_{0:T}^{1:J}): approximate latent posterior distribution 
"""


@dataclass
class SystemTransitionParameters:
    """
    Purpose: These are individual categorical probabilities from l to l',
        and they are shared across all entities and time points. 
   
     Attributes:
        Upsilon: has shape (L, D_s):
            Weights the contribution to system transitions from recurrence. 
        Pi: has shape (L, L).
            Log of a LxL transition probability matrix (this is the log of the original transition matrix, as noted in the paper).
    """

    Upsilon: NumpyArray2D
    Pi: NumpyArray2D


@jdc.pytree_dataclass
class SystemTransitionParameters_JAX:
    """
    Purpose: These are individual categorical probabilities from l to l',
        and they are shared across all entities and time points. 
   
   Attributes:
        Upsilon: has shape (L, D_s):
            Weights the contribution to system transitions from recurrence
        Pi: has shape (L, L).
            Log of a LxL transition probability matrix (this is the log of the original transition matrix, as noted in the paper)
    """

    Upsilon: JaxNumpyArray2D
    Pi: JaxNumpyArray2D


@dataclass
class EntityTransitionParameters:
    """
    Purpose: These are individual categorical probabilities for each entity j transitioning from k to k',
        and they are shared across all time points. 

    Attributes:
        Psis : has shape (J, L, K, D_e)
            Each Psis[j] gives recurrence weights from the observations,
            after we've transformed the observations from dim D to dim D_e.
            The L dimension is to switch between different matrices with shape (K,D_e).
        Ps : has shape (J, L, K, K)
            Each Ps[j,l] is the log of a KxK transition probability matrix for the system state l and the entity j 
    """

    Psis: NumpyArray4D
    Ps: NumpyArray4D


@jdc.pytree_dataclass
class EntityTransitionParameters_MetaSwitch_JAX:
    """
    Purpose: These are individual categorical probabilities for each entity j transitioning from k to k',
        and they are shared across all time points. 

    Attributes:
        Psis : has shape (J, L, K, D_e)
            Each Psis[j] gives recurrence weights from the observations,
            after we've transformed the observations from dim D to dim D_e.
            The L dimension is to switch between different matrices with shape (K,D_e).
        Ps : has shape (J, L, K, K)
            Each Ps[j,l] is the log of a KxK transition probability matrix
    """

    Psis: JaxNumpyArray4D
    Ps: JaxNumpyArray4D


@dataclass
class ContinuousStateParameters:
    """
    Purpose: These are individual Gaussian density function params for each entity j and for each latent entity regime k,
        and they are shared across all time points. 

    Attributes:
        As : has shape (J, K, D, D) 
        bs : has shape (J, K, D) for the current entity 
        Qs : has shape (J, K, D,D) for the covariance matrix 
            Each Qs[j] is a covariance matrix

    """

    As: NumpyArray4D
    bs: NumpyArray3D
    Qs: NumpyArray4D


@jdc.pytree_dataclass
class ContinuousStateParameters_JAX:
    """
    Purpose: These are individual Gaussian density function params for each entity j and for each latent entity regime k,
        and they are shared across all time points. 

    Attributes:
        As : has shape (J, K, D, D)
        bs : has shape (J, K, D)
        Qs : has shape (J, K, D,D)
            Each Qs[j] is a covariance matrix
    """

    As: JaxNumpyArray4D
    bs: JaxNumpyArray3D
    Qs: JaxNumpyArray4D


@dataclass
class InitializationParameters:
    """
    Purpose: These are initial categorical probabilities for system regimes from l to l',
        for entity regimes k to k', as well as the initial gaussian density function parameters.
        These parameters, other than pi_system, are individual for each entity. 

    Attributes:
        pi_system : has shape (L,)
            Lives on the simplex
        pi_entities : has shape (J, K)
            Each pi_entities[j] lives on the simplex.
        mu_0s : has shape (J,K,D)
            Mean of MVN density on initial continuous state 
        Sigma_0s : has shape (J,K,D,D)
            Covariance of MVN density on initial continuous state 
    """

    pi_system: NumpyArray1D
    pi_entities: NumpyArray2D
    mu_0s: NumpyArray3D
    Sigma_0s: NumpyArray4D


@jdc.pytree_dataclass
class InitializationParameters_JAX:
    """
    Purpose: These are initial categorical probabilities for system regimes from l to l',
        for entity regimes k to k', as well as the initial gaussian density function parameters.
        These parameters, other than pi_system, are individual for each entity. 

    Attributes:
        pi_system : has shape (L,)
            Lives on the simplex
        pi_entities : has shape (J, K)
            Each pi_entities[j] lives on the simplex.
        mu_0s : has shape (J,K,D)
            Mean of MVN density on initial continuous state x0
        Sigma_0s : has shape (J,K,D,D)
            Covariance of MVN density on initial continuous state x0
    """

    pi_system: JaxNumpyArray1D
    pi_entities: JaxNumpyArray2D
    mu_0s: JaxNumpyArray3D
    Sigma_0s: JaxNumpyArray4D


@dataclass
class AllParameters:
    STP: SystemTransitionParameters
    ETP: EntityTransitionParameters
    CSP: ContinuousStateParameters
    IP: InitializationParameters


@jdc.pytree_dataclass
class AllParameters_JAX:
    STP: SystemTransitionParameters_JAX
    ETP: EntityTransitionParameters_MetaSwitch_JAX
    CSP: ContinuousStateParameters_JAX
    IP: InitializationParameters_JAX


###
# Dims
###


@dataclass
class Dims:

    J: int
    K: int
    L: int
    D: int
    D_e: int
    D_s: int


def dims_from_params(all_params: AllParameters) -> Dims:
    return Dims(
        J=np.shape(all_params.ETP.Psis)[0],
        K=np.shape(all_params.ETP.Psis)[2],
        L=np.shape(all_params.ETP.Psis)[1],
        D=np.shape(all_params.CSP.As)[2],
        D_e=np.shape(all_params.ETP.Psis)[3],
        D_s=np.shape(all_params.STP.Upsilon)[1],
    )


def get_dim_of_internal_entity_recurrence(D: int, model: Model):
    """
    Purpose: Return the internal contribution to the entity level transitions from recurrence. 
    The external contribution needs to be defined elsewhere. 

    Arguments: 
        D: dimensionality of the continuous observation x
        model: joint distribution defined in -> (model.py)

    Return: dimensionality of the internal contribution to the entity level transitions from recurrence.
    """
    if model.internal_entity_recurrence_JAX is None:
        return 0
    else:
        transformed_dim = (
            model.internal_entity_recurrence_JAX(
                jnp.zeros(D)
            )
        )
        return np.shape(transformed_dim)[0]


def get_dim_of_internal_system_recurrence(D: int, J: int, model: Model):
    """
    Purpose: Return the internal contribution to the system level transitions from recurrence. 
    The external contribution needs to be defined elsewhere. 

    Arguments: 
        D: dimensionality of the continuous observation x
        J: number of entities
        model: joint distribution defined in -> (model.py)

    Return: dimensionality of the internal contribution to the system level transitions from recurrence.
    """
    if (
        model.internal_system_recurrence_JAX
        is None
    ):
        return 0
    else:
        transformed_dim = (
            model.internal_system_recurrence_JAX(
                jnp.zeros(D * J)
            )
        )
        return np.shape(transformed_dim)[0]


###
# Conversions: Jax -> Numpy
###


def numpyify_param_group(param_group_instance, New_Param_Group_Class):
    """
    Purpose: Takes a parameter group, e.g. `InitializationParameters_JAX`, and constructs
        a corresponding numpy parameter group,  e.g. `InitializationParameters`,
        which has the same attributes but as np.arrays instead of jnp.arrays.

    Arguments: 
        param_group_instance: param group (e.g. ETP, STP, CSP ,IP)
        New_Param_Group_Class: param class (e.g. InitailzationParemeters or InitializationParameters_JAX)

    Returns: A parameter group with numpy instead of JAX objects. 

    """
    dict_of_numpy_arrays = {}
    for attr, value in vars(param_group_instance).items():
        dict_of_numpy_arrays[attr] = np.asarray(value)

    return New_Param_Group_Class(**dict_of_numpy_arrays)


def jaxify_param_group(param_group_instance, New_Param_Group_Class):
    """
    Purpose: Takes a parameter group, e.g. `InitializationParameters`, and constructs
        a corresponding jax parameter group,  e.g. `InitializationParameters_JAX`,
        which has the same attributes but as jnp.arrays instead of np.arrays.

    Arguments: 
        param_group_instance: param group (e.g. ETP, STP, CSP ,IP)
        New_Param_Group_Class: param class (e.g. InitailzationParemeters or InitializationParameters_JAX)

    Returns: A parameter group with JAX instead of numpy objects. 

    """
    for attr, value in vars(param_group_instance).items():
        setattr(param_group_instance, attr, jnp.asarray(value))

    return New_Param_Group_Class(**vars(param_group_instance))


def jax_params_from_params(all_params: AllParameters) -> AllParameters_JAX:
    """
    Purpose: Takes numpy params and makes them into JAX param groups. 

    Arguments: 
        all_params: STP, ETP, CSP, IP

    Returns: All parameters in the JAX group form.  
    """

    if "AllParameters_JAX" in str(type(all_params)):
        warnings.warn("Parameters ALREADY have Jax typing.")
        return all_params

    return AllParameters_JAX(
        jaxify_param_group(all_params.STP, SystemTransitionParameters_JAX),
        jaxify_param_group(all_params.ETP, EntityTransitionParameters_MetaSwitch_JAX),
        jaxify_param_group(all_params.CSP, ContinuousStateParameters_JAX),
        jaxify_param_group(all_params.IP, InitializationParameters_JAX),
    )


def numpy_params_from_params(all_params: AllParameters_JAX) -> AllParameters:
    """
    Purpose: Takes JAX params and makes them into numpy param groups. 

    Arguments: 
        all_params: STP, ETP, CSP, IP

    Returns: All parameters in the numpy group form.  
    """

    return AllParameters_JAX(
        numpyify_param_group(all_params.STP, SystemTransitionParameters),
        numpyify_param_group(all_params.ETP, EntityTransitionParameters_MetaSwitch),
        numpyify_param_group(all_params.CSP, ContinuousStateParameters_Gaussian),
        numpyify_param_group(all_params.IP, InitializationParameters_Gaussian),
    )


###
# Conversions for M-step: Unconstrained representations for transition probability matrices
###


@jdc.pytree_dataclass
class SystemTransitionParameters_WithUnconstrainedTPMs_JAX:
    """
    Purpose: 
        Create UNCONSTRAINED STP parameters that can be used for optimizing with gradient descent. 
        These parameters do not have to be bounded on the simplex and instead and can unconstrained 
        during training and then mapped back to a contrained valid form. 
    Attributes:
        Upsilon: has shape (L, D_s):
            Weights the contribution to system transitions from recurrence. 
        PiTilde_Unconstrained: has shape (L, L-1).
            Whereas Pi is the log of a LxL transition probability matrix
            (so PiTilde := jnp.exp(Pi) is a LxL transition probability matrix whose rows
            are constrained to live on the simplex with L entries). we here represent
            each row with only L-1 UNCONSTRAINED entries.  This makes for easier optimization;
            we don't have to worry about respecting parameter constraints
            (as if we worked with jnp.exp(Pi)),  nor do we have to worry about overparametrized representations
            (as if we worked with Pi).
    """

    Upsilon: JaxNumpyArray2D
    PiTilde_Unconstrained: JaxNumpyArray2D


def STP_with_unconstrained_tpms_from_ordinary_STP(
    STP: SystemTransitionParameters_JAX,
) -> SystemTransitionParameters_WithUnconstrainedTPMs_JAX:

    """
    Purpose: Transforms constrained into unconstrained STP parameters for optimization. 

    Arguments:
        STP: The system state parameters Upsilon has shape (L, D_s) and Pi has shape (L, L)

    Returns: 
        Uncontrained STP parameters.
    """
    PiTilde_Unconstrained = unconstrained_tpm_from_tpm(jnp.exp(STP.Pi))
    return SystemTransitionParameters_WithUnconstrainedTPMs_JAX(STP.Upsilon, PiTilde_Unconstrained)


def ordinary_STP_from_STP_with_unconstrained_tpms(
    STP_WUC: SystemTransitionParameters_WithUnconstrainedTPMs_JAX,
) -> SystemTransitionParameters_JAX:
    """
    Purpose: Transforms unconstrained into constrained STP parameters for optimization. 

    Arguments:
        STP_WUC: Constrained version of STP parameters 

    Returns: 
        Contrained STP parameters.
    """
    PiTilde = tpm_from_unconstrained_tpm(STP_WUC.PiTilde_Unconstrained)
    return SystemTransitionParameters_JAX(STP_WUC.Upsilon, jnp.log(PiTilde))


@jdc.pytree_dataclass
class EntityTransitionParameters_MetaSwitch_WithUnconstrainedTPMs_JAX:
    """
    Purpose: Create UNCONSTRAINED ETP parameters that can be used for optimizing with gradient descent. 
        These parameters do not have to be bounded on the simplex and instead and can unconstrained 
        during training and then mapped back to a contrained valid form. 

    Attributes:
        Psis : has shape (J, L, K, D_e)
            Each Psis[j] gives recurrence weights from the observations,
            after we've transformed the observations from dim D to dim D_e.
            The L dimension is to switch between different matrices with shape (K,D_e).
        PTildes_Unconstrained : has shape (J, L, K, K-1)
            Whereas each Ps[j,l] is the log of a LxL transition probability matrix
            (so if PTildes := jnp.exp(Ps), then each PTildes[j,l] is a KxK transition probability matrix whose rows
            are constrained to live on the simplex with K entries), we here represent
            each row with only K-1 UNCONSTRAINED entries.  This makes for easier optimization;
            we don't have to worry about respecting parameter constraints
            (as if we worked with jnp.exp(Ps)),  nor do we have to worry about overparametrized representations
            (as if we worked with Ps).

    """

    Psis: JaxNumpyArray4D
    PTildes_Unconstrained: JaxNumpyArray4D


def ETP_MetaSwitch_with_unconstrained_tpms_from_ordinary_ETP_MetaSwitch(
    ETP: EntityTransitionParameters_MetaSwitch_JAX,
) -> EntityTransitionParameters_MetaSwitch_WithUnconstrainedTPMs_JAX:
    """
    Purpose: Transforms constrained into unconstrained ETP parameters for optimization. 

    Arguments:
        ETP: the entity transition parameters Psis has shape (J, L, K, D_e) and Ps has shape (J, L, K, K)

    Returns: 
        Uncontrained ETP parameters.
    """
    PTildes_Unconstrained = unconstrained_tpm_from_tpm(jnp.exp(ETP.Ps))
    return EntityTransitionParameters_MetaSwitch_WithUnconstrainedTPMs_JAX(ETP.Psis, PTildes_Unconstrained)


def ordinary_ETP_MetaSwitch_from_ETP_MetaSwitch_with_unconstrained_tpms(
    ETP_WUC: EntityTransitionParameters_MetaSwitch_WithUnconstrainedTPMs_JAX,
) -> EntityTransitionParameters_MetaSwitch_JAX:
    """
    Purpose: Transforms unconstrained into constrained ETP parameters for optimization. 

    Arguments:
        ETP_WUC:  contrained version of ETP params

    Returns: 
        Contrained ETP parameters.
    """
    PTildes = tpm_from_unconstrained_tpm(ETP_WUC.PTildes_Unconstrained)
    return EntityTransitionParameters_MetaSwitch_JAX(ETP_WUC.Psis, jnp.log(PTildes))


# @jit
def cholesky_nzvals_from_covariances_with_two_mapping_axes_JAX(batched):

    """
    Purpose: Compute the cholesky non-zero values from the covariance parameters matrix. 
         This is for the gaussian parameters. 

    Arguments:
        batched: All Gaussian covariance parameters 

    Returns: 
        Map to the non-zero cholseky factors from the covariance parameters.
    """

    return vmap(vmap(cholesky_nzvals_from_covariance_JAX))(batched)


# @jit
def covariance_from_cholesky_nzvals_with_two_mapping_axes_JAX(batched):
    """
    Purpose: Compute the covariance parameter values from the cholesky non-zero values.
        This is for the gaussian parameters.  

    Arguments:
        batched: All Gaussian covariance parameters 

    Returns: 
        Map to the covariance parameters from the non-zero cholseky factors.
    """
    return vmap(vmap(covariance_from_cholesky_nzvals_JAX))(batched)


@jdc.pytree_dataclass
class ContinuousStateParameters_Gaussian_WithUnconstrainedCovariances_JAX:
    """
    Purpose: Create UNCONSTRAINED CSP parameters that can be used for optimizing with gradient descent. 
        These parameters do not have to be positive semi-definite (as valid covariance matrices do) and can unconstrained 
        during training and then mapped back to a contrained valid form. 

    Attributes:
        As : has shape (J, K, D, D)
        bs : has shape (J, K, D)
        Cholesky_nzvals : has shape (J, K, R)
            Each (j,k)-th entry gives R values which gives the non-zero values
            of a lower tringular Cholesky factor from which one can
            obtain a valid covariance matrix.
    """

    As: JaxNumpyArray4D
    bs: JaxNumpyArray3D
    cholesky_nzvals: JaxNumpyArray4D


def CSP_Gaussian_with_unconstrained_covariances_from_ordinary_CSP_Gaussian(
    CSP: ContinuousStateParameters_JAX,
) -> ContinuousStateParameters_Gaussian_WithUnconstrainedCovariances_JAX:

    """
    Purpose: Transforms constrained into unconstrained CSP parameters for optimization. 

    Arguments:
        CSP: the observation parameters A[j,k], b[j,k], Q[j,k]

    Returns: 
        Uncontrained CSP parameters.
    """

    cholesky_nzvals = cholesky_nzvals_from_covariances_with_two_mapping_axes_JAX(CSP.Qs)
    return ContinuousStateParameters_Gaussian_WithUnconstrainedCovariances_JAX(CSP.As, CSP.bs, cholesky_nzvals)


def ordinary_CSP_Gaussian_from_CSP_Gaussian_with_unconstrained_covariances(
    CSP_WUC: ContinuousStateParameters_Gaussian_WithUnconstrainedCovariances_JAX,
) -> ContinuousStateParameters_JAX:

    """
    Purpose: Transforms unconstrained into constrained CSP parameters for optimization. 

    Arguments:
        CSP_WUC: contrained version of CSP params

    Returns: 
        Contrained CSP parameters.
    """
    Qs = covariance_from_cholesky_nzvals_with_two_mapping_axes_JAX(CSP_WUC.cholesky_nzvals)
    return ContinuousStateParameters_JAX(CSP_WUC.As, CSP_WUC.bs, Qs)


def normalize_log_tpms_within_parameter_group(param_group, name_of_param_to_log_normalize: str, axis: int):
    """
    Purpose: Takes a parameter group, e.g. `SystemTransitionParameters`, and normalizes (on a log scale)
        a parameter within that group. Useful for ensuring valid categorical probability distributions. 

    Arguments: 
        param_group: a single parameter grouping (e.g. STP, ETP, CSP, IP)
        name_of_param_to_log_normalize: a single param within that group (e.g. Pi)
        axis: The axis for which to normalize 

    Returns:   
        New param group that is normalized. 

    """
    object_to_log_normalize = param_group.__dict__[name_of_param_to_log_normalize]
    log_normalized_object = normalize_log_potentials_by_axis_JAX(object_to_log_normalize, axis)
    dict_for_new_param_group = param_group.__dict__
    dict_for_new_param_group[name_of_param_to_log_normalize] = log_normalized_object
    return type(param_group)(**dict_for_new_param_group)


def save_params(params: AllParameters_JAX, save_dir: str, basename_prefix: str = ""):
    """
    Purpose: Save the parameters into a file. 
    
    Arguments: 
        params: all parameters (STP, ETP, CSP, IP)
        save_dir: str for the path to save the file
        basename_prefix:  str for the file name

    """
    filepath = os.path.join(save_dir, f"{basename_prefix}_params.pkl")
    with open(filepath, "wb") as file:
        pickle.dump(params, file)


def load_params(filepath: str):
    """
    Purpose: load params from a file 
    
    Arguments: 
        filepath: path to file that has params saved 
    """
    with open(filepath, "rb") as file:
        params = pickle.load(file)
    return params
