import pandas as pd
import numpy as np
from typing import List, Union
from ssm.util import find_permutation
from scipy.stats import spearmanr

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
Computes metrics for the entire HSRDM: latent state probabilistic dynamics. 
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

def compute_monotonicity_correlation(
    estimated_regime_probabilities,
    evidence_strength,
):
    """
    Purpose: Computes the Spearman rank correlation between estimated regime probabilities
        and evidence strength across all time points and all entities. Both inputs should be 2D arrays of shape (T, J).
        We flatten over time and entities and compute ONE global Spearman rho. Includes all entities (even those who did not speak), 
        as long as they have defined probability and evidence values. NaNs are dropped pairwise.

    Arguments: 
        estimated_regime_probabilities : (T, n_entities)
            Estimated regime probabilities (e.g., for a specific latent regime).
        evidence_strength : (T, n_entities)
            Evidence strengths (e.g., integers 1–8, where 1 is none and 8 is max).

    Returns: 
        rho : float
            Spearman rank correlation coefficient in [-1, 1].
            Close to 1 indicates a strong monotonically increasing relationship.
        p_value : float
            Two-sided p-value for the test of no monotonic association.
    """
    probs = np.asarray(estimated_regime_probabilities).ravel()
    evid = np.asarray(evidence_strength).ravel()

    if probs.shape != evid.shape:
        raise ValueError(
            f"Shape mismatch after flattening: "
            f"probs shape {probs.shape} vs evidence shape {evid.shape}"
        )

    # Drop NaNs pairwise (if you encode non-speaking entities as NaN, they'll be ignored)
    mask = ~np.isnan(probs) & ~np.isnan(evid)
    if mask.sum() < 2:
        raise ValueError(
            "Need at least two valid (probability, evidence) pairs "
            "to compute Spearman correlation."
        )

    rho, p_value = spearmanr(probs[mask], evid[mask])
    return rho, p_value

def windowed_spearman_no_evidence_prob(
    probs,
    evidence_strengths,
    labels_int,
    episode_end_times,
    window_size=20,
    no_evidence_class=1,
):
    """
    Purpose: Computes Spearman correlation between:
      - mean probabilities *restricted to No-evidence positions* in each window
      - mean evidence strengths (all positions) in each window

    Arguments: 
        probs : array-like, shape (T, J_max)
            Model probabilities per timestep per student.
        evidence_strengths : array-like, shape (T, J_max)
            Evidence strengths (1–8).
        labels_int : array-like, shape (T, J_max)
            Integer class labels (1..C). No-evidence class == 1 by default.
        episode_end_times : list of ints
            End times for each episode (exclusive).
        window_size : int
            Number of consecutive timesteps per window.
        no_evidence_class : int
            Class index corresponding to "No evidence".

    Returns: 
        rho, p_value : float
            Spearman correlation and p-value.
    """

    probs = np.asarray(probs)
    evid = np.asarray(evidence_strengths)
    labels = np.asarray(labels_int)

    prob_means = []
    evid_means = []

    for end_t in episode_end_times:
        last_start = end_t - window_size
        if last_start < 0:
            continue

        for start in range(0, last_start + 1, window_size):
            end = start + window_size
            if end > end_t:
                break

            window_probs  = probs[start:end]      # (20, J)
            window_evid   = evid[start:end]       # (20, J)
            window_labels = labels[start:end]     # (20, J)

            # ---- Evidence: mean over all positions ----
            evid_means.append(np.mean(window_evid))

            # ---- Probabilities: only positions where label == No evidence ----
            mask = (window_labels == no_evidence_class)
            masked_probs = window_probs[mask]

            if masked_probs.size == 0:
                # No no-evidence in this window → skip
                continue

            prob_means.append(np.mean(masked_probs))

    # Require at least 2 windows to compute correlation
    if len(prob_means) < 2:
        raise ValueError(
            f"Not enough windows with no-evidence entries: only {len(prob_means)}."
        )

    rho, p_value = spearmanr(prob_means, evid_means)
    return rho, p_value
