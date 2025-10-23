import pandas as pd
import numpy as np
from typing import List, Union
from ssm.util import find_permutation

from utilities.types import(
    NumpyArray1D,
    NumpyArray2D,
    NumpyArray3D,
    JaxNumpyArray1D, 
    JaxNumpyArray2D,
)
from params import AllParameters, dims_from_params
from compute_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summaries_NUMPY,
    HMM_Posterior_Summary,
)

"""
Computes metrics for the entire HSRDM: latent state classification. 
"""


def compute_regime_labeling_accuracy(
    estimated_regime_seq: Union[NumpyArray2D, JaxNumpyArray2D],
    true_regime_seq: Union[NumpyArray1D, JaxNumpyArray1D],
) -> float:
    """
    Due to label-switching, need to find first find the permutation that best matches the truth
    in order to compute accuracy.
    """
    # Convert types in case we have jax arrays with int32s.  This is necessary because
    # `find_permutation` does a type checking and assumes the type is int, not int32.
    estimated_regime_seq = np.asarray(estimated_regime_seq, dtype=int)
    true_regime_seq = np.asarray(true_regime_seq, dtype=int)

    # Rk: The `find_permutation` function requires numpy arrays
    perm_of_estimated = find_permutation(
        np.array(estimated_regime_seq),
        np.array(true_regime_seq),
    )
    estimated_regime_seq_with_aligned_labels = np.array(
        [perm_of_estimated[x] for x in estimated_regime_seq]
    )
    pct_correct_regimes = np.mean(true_regime_seq == estimated_regime_seq_with_aligned_labels)
    return pct_correct_regimes


def get_aligned_estimate(
    estimated_regime_seq: Union[NumpyArray2D, JaxNumpyArray2D],
    true_regime_seq: Union[NumpyArray1D, JaxNumpyArray1D],
) -> float:
    """
    Due to label-switching, need to find first find the permutation that best matches the truth
    in order to compute accuracy. This function returns the labels
    """
    # Convert types in case we have jax arrays with int32s.  This is necessary because
    # `find_permutation` does a type checking and assumes the type is int, not int32.
    estimated_regime_seq = np.asarray(estimated_regime_seq, dtype=int)
    true_regime_seq = np.asarray(true_regime_seq, dtype=int)

    # Rk: The `find_permutation` function requires numpy arrays
    perm_of_estimated = find_permutation(
        np.array(estimated_regime_seq),
        np.array(true_regime_seq),
    )
    estimated_regime_seq_with_aligned_labels = np.array(
        [perm_of_estimated[x] for x in estimated_regime_seq]
    )
    return estimated_regime_seq_with_aligned_labels




