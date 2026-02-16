import datetime
import warnings
from typing import List, Tuple
import os
import json
import copy
from typing import Optional, Tuple
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import torch
import tensorflow_probability.substrates.jax.bijectors as tfb
from jax.scipy.special import logsumexp as logsumexp_JAX
from scipy.special import logsumexp
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from utilities.types import NumpyArray1D, NumpyArray2D, JaxNumpyArray1D, JaxNumpyArray2D, JaxNumpyArray3D, JaxNumpyArray4D
from model import Model 

"""
Utility functions used throughout the repository. 
"""

###
# Functions to create specific parameters (shared, non-shared, fixed) for specific modeling states 
###

def generate_silent_observation(observations: JaxNumpyArray3D
)-> list[float, float]:
    """
    Purpose: Return a list of the fixed means and fixed covariances per entity for the gaussian distributions that I want to clamp when an entity 
    is in a certain state 

    Attributes: 
        observations: np.array of shape (T,J,D) where the (t,j)-th entry is in R^D.

    Returns: 
        List of fixed means and fixed covariances per entity for gaussian distrubtions that I want to clamp when an entity is in a certain state 
    """
    mu_fixed = jnp.array(observations[0][1])  # set to an example silence vector 
    J = observations.shape[1]

    fixed_means_per_entity = jnp.tile(mu_fixed[None, :], (J, 1)) # (J, D) fixed means for each entity 

    fixed_sigma2 = 1e-3  # small variance
    D = fixed_means_per_entity.shape[-1]
    fixed_cov = fixed_sigma2 * jnp.eye(D)  # (D, D) isotropic covariance matrix 

    return fixed_means_per_entity, fixed_cov

def clamp_parameters( 
    observations: JaxNumpyArray3D,
    As: JaxNumpyArray4D,
    bs: JaxNumpyArray3D,
    Qs: JaxNumpyArray4D,
    K_SPECIAL=0
) -> list[float, float]:
    """
    Purpose: Return a list of the fixed means and fixed covariances per entity for the gaussian distributions that I want to clamp when an entity 
    is in a certain state 

    Attributes: 
        observations: np.array of shape (T,J,D) where the (t,j)-th entry is in R^D.
        As: has shape (J, K, D, D) 
        bs: has shape (J, K, D) for the current entity
        Qs: has shape (J, K, D,D) for the covariance matrix 
            Each Qs[j] is a covariance matrix
        K_SPECIAL: the specific z latent state that you want to clamp the params in for each entity. In this case, it is k=0. 

    Returns: 
        CSP: the continuous state parameters As, bs, Qs with a hard coded As = 0, bs = specific observation mean, and Qs = small isotropic covariance matrix
    """
    fixed_means_per_entity, fixed_cov = generate_silent_observation(observations)
    J, K, D, _ = As.shape

    # Remove dependence on x_{t-1}
    As = jnp.asarray(As).at[:, K_SPECIAL, :, :].set(jnp.zeros((J, D, D)))
    # Hard-coded mean
    bs = jnp.asarray(bs).at[:, K_SPECIAL, :].set(fixed_means_per_entity)

    # Hard-coded tiny isotropic cov
    Qs = jnp.asarray(Qs).at[:, K_SPECIAL, :, :].set(fixed_cov)

    return As, bs, Qs

def _tile_shared_LKDe_to_JLKDe(Psis_shared, J):
    """
    Purpose: Return entity shared parameters (one set of entity params for all entities) for the entity state parameters Psis that are in the form (J,L,K,D_e).
    So each j in all J will be the same paramter values. 

    Attributes: 
        Psis_shared: Shared (L,K,D_e) -> (J,L,K,D_e)
        J: number of entities 
    Returns: 
        Psis_shared params that are the same across all entities in the form (L,K,D_e) -> (J,L,K,D_e)
    """
    return jnp.tile(Psis_shared[None, ...], (J, 1, 1, 1))

def _tile_shared_LKK_to_JLKK(Ps_shared, J):
    """
    Purpose: Return entity shared parameters (one set of entity params for all entities) for the entity state parameters Ps that are in the form (J,L,K,K).
    So each j in all J will be the same paramter values. 

    Attributes: 
        Ps_shared: Shared (L,K,K) -> (J,L,K,K)
        J: number of entities 
    Returns: 
        Ps_shared params that are the same across all entities in the form (L,K,K) -> (J,L,K,K)
    """
    return jnp.tile(Ps_shared[None, ...], (J, 1, 1, 1))


###
# Functions for computing the model metrics and final plots
###

def evidence_strength_from_onehot(evid_onehot: np.ndarray) -> np.ndarray:
    """
    Purpose: 
        Compute the evidence strength from the one-hot label form. 
    
    Arguments: 
        evid_onehot: (T, J, C) one-hot or soft one-hot
    
    Returns: 
        (T, J) integer class index via argmax over C
    """
    evid = np.asarray(evid_onehot)
    if evid.ndim != 3:
        raise ValueError(f"Expected evid_onehot shape (T,J,C); got {evid.shape}")
    return np.argmax(evid, axis=-1)

def speaker_index_per_t(
    observations: np.ndarray,
) -> np.ndarray:
    """
    Purpose: 
        Returns speaker_idx: (T,) where speaker_idx[t] is the index j of the entity that is NOT silent.
        If no unique speaker is found at time t, returns -1 for that t. Identifies the observation that is not silence. 

    Arguments: 
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D

    Returns: Returns an array (T,) of the speaker indexes at each time point. 
    """
    obs = np.asarray(observations)
    T, J, D = obs.shape
    silent_embedding = observations[0][1]
    silent = np.asarray(silent_embedding).reshape(1, 1, D)

    diff = np.linalg.norm(obs - silent, axis=-1)  # (T,J)
    return diff.argmax(axis=1).astype(int)        # (T,)

def pearson_corr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Purpose: Computes the pearson correlation coefficient between two arrays (datasets). 

    Arguments: 
        x: array 1 of data 
        y: array 2 of data 
    
    Returns correlations between datasets
    """
    x = np.asarray(x).astype(float)
    y = np.asarray(y).astype(float)

    if x.size < 2:
        return np.nan

    x0 = x - x.mean()
    y0 = y - y.mean()
    vx = np.mean(x0 * x0)
    vy = np.mean(y0 * y0)
    if vx < eps or vy < eps:
        return np.nan

    cov = np.mean(x0 * y0)
    return cov / (np.sqrt(vx) * np.sqrt(vy))


def spearman_corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    """
    Returns Spearman correlation or np.nan if undefined.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size < 2:
        return np.nan
    if np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan

    rho, _ = spearmanr(x, y)
    return rho

def silent_mask_from_observations(observations: np.ndarray) -> np.ndarray:
    """
    Purpose: 
        Define the 'silent' embedding as observations[0,1,:]),
    and mark (t,j) as silent if obs[t,j,:] matches that embedding.
    Arguments: 
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D  
    Returns:
        silent_mask: (T,J) boolean
    """
    obs = np.asarray(observations)
    silent_embedding = obs[0, 1, :]              
    silent_mask = np.all(np.isclose(obs, silent_embedding[None, None, :]), axis=-1)
    return silent_mask

def make_tasteful_pink_cmap(name="tasteful_pink", N=256):
    """
    Purpose: Defines my pink cmap with gradients for the evidence strengths. 

    Arguments: 
        name: defined name of the map 
        N: number of samples/shades in the entire gradient 
    
    Returns Defines my pink cmap with gradients for the evidence strengths. 
    """
    # blush -> rose -> raspberry
    stops = ["#FAF3F6", "#F6CEDB", "#E78FB3", "#B1125D"]
    return LinearSegmentedColormap.from_list(name, stops, N=N)

def _to_listed_pink_gradient(colors: list, num_levels: int):
    """
    Purpose: If you pass a list of colors, treat them as gradient stops.
    Otherwise, use default tasteful pink. Provides a gradient version of hte colormap. 

    Arguments: 
        colors: list of colors in the map 
        num_levels: number of gradient stops
    
    Returns a ListedColormap with num_levels discrete steps.
    """
    if colors is None or len(colors) == 0:
        cmap = make_tasteful_pink_cmap()
    else:
        cmap = LinearSegmentedColormap.from_list("custom_pink", colors, N=256)

    # Discretize into num_levels
    return ListedColormap([cmap(i / max(1, num_levels - 1)) for i in range(num_levels)])

def distinct_colors(K: int):
    """
    Purpose: If you pass a list of colors, treat them as gradient stops.
    Otherwise, use default tasteful pink. Provides a gradient version of hte colormap. 

    Arguments: 
        K: number of states
    
    Returns a list of distinct colors for my k states 
    """
    cmaps = [get_cmap("tab20"), get_cmap("tab20b"), get_cmap("tab20c")]
    colors = []
    for cmap in cmaps:
        colors.extend([cmap(i) for i in range(cmap.N)])
    if K <= len(colors):
        return colors[:K]
    # fallback: evenly-spaced hues (still reasonably distinct, but not as good as categorical)
    hsv = get_cmap("hsv")
    return [hsv(i / K) for i in range(K)]

def append_gaussian_params_by_iter_csv(*, save_dir: str, iteration: int, As, bs, Qs, j0: int = 0):
    """
    Purpose: Appends rows to: f"{save_dir}__gaussian_params_by_iter.csv"
        One row per k with full A,b,Q stored as JSON strings.
    Arguments: 
        save_dir: str for the path to save the file
        iteration: training epoch
        As : has shape (J, K, D, D) 
        bs : has shape (J, K, D) for the current entity 
        Qs : has shape (J, K, D,D) for the covariance matrix 
            Each Qs[j] is a covariance matrix
        j0: int = 0 for one of the entities for when all entities share params 

    Returns: csv table of gaussian parameters per training iteration
    """
    As0 = np.asarray(As[j0])  # (K,D,D)
    bs0 = np.asarray(bs[j0])  # (K,D)
    Qs0 = np.asarray(Qs[j0])  # (K,D,D)

    K = As0.shape[0]
    rows = []
    for k in range(K):
        rows.append({
            "iter": int(iteration),
            "k": int(k),
            "A": json.dumps(As0[k].tolist()),
            "b": json.dumps(bs0[k].tolist()),
            "Q": json.dumps(Qs0[k].tolist()),
        })

    df = pd.DataFrame(rows)
    out_csv = f"{save_dir}_gaussian_params_by_iter_{iteration}.csv"
    df.to_csv(out_csv, mode="a", header=not Path(out_csv).exists(), index=False)

def save_posteriors_as_strings(
    *,
    save_dir: str,
    iteration: int | None,
    posterior_probabilities: np.ndarray,
    system_posterior_probabilities: np.ndarray,
):
    """
    Purpose:
        Save the full posterior probability tensors for entity-level and system-level
        latent states as JSON-encoded strings in a CSV file. Each call appends a single
        row corresponding to one training iteration (or final state).

    Arguments:
        save_dir:
            Base path used to construct the output CSV filename. The file written is
            f"{save_dir}__posteriors_as_strings.csv".
        iteration:
            Training iteration index. Use None to indicate the final posteriors after
            training has completed.
        posterior_probabilities:
            Entity-level posterior probabilities, typically of shape (T, J, K),
            where T is the number of time steps, J is the number of entities, and
            K is the number of entity latent states.
        system_posterior_probabilities:
            System-level posterior probabilities, typically of shape (T, L) or
            (T, 1, L), where L is the number of system latent states.

    Returns:
        None. Appends a row to the CSV file on disk.
    """

    row = pd.DataFrame([{
        "iter": None if iteration is None else int(iteration),
        "entity_posteriors_json": json.dumps(np.asarray(posterior_probabilities).tolist()),
        "system_posteriors_json": json.dumps(np.asarray(system_posterior_probabilities).tolist()),
    }])

    out_csv = f"{save_dir}__posteriors_as_strings.csv"
    row.to_csv(out_csv, mode="a", header=not Path(out_csv).exists(), index=False)


def save_maxprob_tables(
    *,
    save_dir: str,
    iteration: int | None,
    posterior_probabilities: np.ndarray,          # (T,J,K)
    system_posterior_probabilities: np.ndarray,   # (T,L) or (T,1,L)
    one_hot_evidence: np.ndarray,                 # (T,J,C)
):
    """
    Purpose:
        Save tables summarizing the maximum posterior probability and corresponding
        argmax state at each time point. Entity-level results are saved per (t, j),
        while system-level results are saved per t. Additionally, compute the evidence
        class index from the one-hot evidence vectors per (t, j) and save it as its
        own table.

    Arguments:
        save_dir:
            Base path used to construct the output CSV filenames. The files written are:
              - f"{save_dir}__entity_maxprob_by_tj.csv"
              - f"{save_dir}__system_maxprob_by_t.csv"
              - f"{save_dir}__evidence_class_by_tj.csv"
        iteration:
            Training iteration index. Use None to indicate the final posteriors after
            training has completed.
        posterior_probabilities:
            Entity-level posterior probabilities of shape (T, J, K).
        system_posterior_probabilities:
            System-level posterior probabilities of shape (T, L) or (T, 1, L).
        one_hot_evidence:
            One-hot evidence labels of shape (T, J, C), where C is the number of evidence
            classes. The index of the '1' is computed via argmax.

    Returns:
        None. Appends rows to the corresponding CSV files on disk.
    """

    # ----- entity-level posterior max prob per (t,j) -----
    P = np.asarray(posterior_probabilities)
    T, J, K = P.shape

    argmax_k = np.argmax(P, axis=2)   # (T,J)
    max_prob = np.max(P, axis=2)      # (T,J)

    df_ent = pd.DataFrame({
        "iter": None if iteration is None else int(iteration),
        "t": np.repeat(np.arange(T), J),
        "j": np.tile(np.arange(J), T),
        "argmax_state": argmax_k.reshape(-1),
        "max_prob": max_prob.reshape(-1),
    })

    out_csv = f"{save_dir}__entity_maxprob.csv"
    df_ent.to_csv(out_csv, mode="a", header=not Path(out_csv).exists(), index=False)

    # ----- system-level posterior max prob per t -----
    S = np.asarray(system_posterior_probabilities)
    if S.ndim == 3 and S.shape[1] == 1:
        S = S[:, 0, :]  # (T,L)
    Ts, L = S.shape

    df_sys = pd.DataFrame({
        "iter": None if iteration is None else int(iteration),
        "t": np.arange(Ts),
        "argmax_state": np.argmax(S, axis=1),
        "max_prob": np.max(S, axis=1),
    })

    out_csv = f"{save_dir}__system_maxprob.csv"
    df_sys.to_csv(out_csv, mode="a", header=not Path(out_csv).exists(), index=False)

    # ----- evidence class index per (t,j) -----
    E = np.asarray(one_hot_evidence)  # (T,J,C)
    if E.shape[0] != T or E.shape[1] != J:
        raise ValueError(f"one_hot_evidence shape {E.shape} does not match (T,J)=({T},{J})")

    evidence_class = np.argmax(E, axis=2)   # (T,J)
    evidence_sum = np.sum(E, axis=2)        # (T,J) should be 1 for proper one-hot

    df_evid = pd.DataFrame({
        "iter": None if iteration is None else int(iteration),
        "t": np.repeat(np.arange(T), J),
        "j": np.tile(np.arange(J), T),
        "evidence_class": evidence_class.reshape(-1),
        "evidence_is_onehot": (evidence_sum.reshape(-1) == 1),
    })

    out_csv = f"{save_dir}__evidence_class_by_tj.csv"
    df_evid.to_csv(out_csv, mode="a", header=not Path(out_csv).exists(), index=False)

def save_state_frequency_counts(
    *,
    save_dir: str,
    iteration: int | None,
    posterior_probabilities: np.ndarray,
    system_posterior_probabilities: np.ndarray,
):
    """
    Purpose:
        Save frequency counts of latent states using hard assignments (argmax of
        posterior probabilities). Entity-level counts are aggregated across all
        time steps and entities, while system-level counts are aggregated across
        time steps only.

    Arguments:
        save_dir:
            Base path used to construct the output CSV filename. The file written is
            f"{save_dir}__state_counts.csv".
        iteration:
            Training iteration index. Use None to indicate the final posteriors after
            training has completed.
        posterior_probabilities:
            Entity-level posterior probabilities of shape (T, J, K).
        system_posterior_probabilities:
            System-level posterior probabilities of shape (T, L) or (T, 1, L).

    Returns:
        None. Appends rows to the CSV file on disk.
    """

    # ----- entity-level counts -----
    P = np.asarray(posterior_probabilities)
    T, J, K = P.shape

    z_hat = np.argmax(P, axis=2).reshape(-1)
    ent_counts = np.bincount(z_hat, minlength=K)

    df_ent = pd.DataFrame([{
        "iter": None if iteration is None else int(iteration),
        "level": "entity",
        **{f"count_k{k}": int(ent_counts[k]) for k in range(K)},
    }])

    out_csv = f"{save_dir}__latent_frequency_counts.csv"
    df_ent.to_csv(out_csv, mode="a", header=not Path(out_csv).exists(), index=False)

    # ----- system-level counts -----
    S = np.asarray(system_posterior_probabilities)
    if S.ndim == 3 and S.shape[1] == 1:
        S = S[:, 0, :]

    Ts, L = S.shape
    s_hat = np.argmax(S, axis=1)
    sys_counts = np.bincount(s_hat, minlength=L)

    df_sys = pd.DataFrame([{
        "iter": None if iteration is None else int(iteration),
        "level": "system",
        **{f"count_l{l}": int(sys_counts[l]) for l in range(L)},
    }])

    df_sys.to_csv(out_csv, mode="a", header=False, index=False)



def sample_emissions_for_transition_types(
    *,
    observations: np.ndarray,      # (T, J, D)
    one_hot_evidence: np.ndarray,   # (T, J, C)
    log_emissions: np.ndarray,    
    iteration: int,  # (T, J, K)
    save_dir: str,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Purpose:
        Randomly select one example of several transition types and record the
        emission probability vectors (converted from log-emissions) at t-1 and t
        for that entity.

        Transition types sampled (one each, if available):
          1) silent -> silent
          2) silent -> talk (NO evidence in talk)
          3) talk (NO evidence) -> silent
          4) silent -> talk (WITH evidence in talk)
          5) talk (WITH evidence) -> silent

    Arguments:
        observations:
            Array of shape (T, J, D). Silence is defined as
            observations[t, j] == observations[0, 1].
        one_hot_evidence:
            One-hot evidence labels of shape (T, J, C).
            Class 0 is assumed to mean "no evidence".
        log_emissions:
            Log emission values of shape (T, J, K).
        iteration:
            Training_iteration.
        save_dir: 
            Directory to save the csv file.
        seed:
            RNG seed used to randomly select one example per transition type.

    Returns:
        A pandas DataFrame with one row per sampled transition type. Each row contains:
            - transition_type
            - t_prev, t_curr, j
            - is_silent_prev, is_silent_curr
            - evid_class_prev, evid_class_curr
            - emiss_prev_json, emiss_curr_json   (length-K probability vectors)
            - emiss_prev_argmax, emiss_curr_argmax
            - emiss_prev_maxprob, emiss_curr_maxprob

        Transition types with no eligible examples are omitted.
    """
    obs = np.asarray(observations)
    evid = np.asarray(one_hot_evidence)
    log_em = np.asarray(log_emissions)

    T, J, D = obs.shape
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # SILENCE LOGIC (exact match to observations[0][1])
    # ------------------------------------------------------------
    silent_vec = obs[0, 1]                        # (D,)
    is_silent = np.all(obs == silent_vec, axis=2)  # (T, J)
    is_talk = ~is_silent

    # ------------------------------------------------------------
    # Evidence logic (class 0 = no evidence)
    # ------------------------------------------------------------
    evid_class = evidence_strength_from_onehot(evid)   # (T, J)
    has_evidence = evid_class != 0
    no_evidence = evid_class == 0

    # ------------------------------------------------------------
    # Convert log-emissions -> normalized probabilities
    # ------------------------------------------------------------
    log_em_max = np.max(log_em, axis=2, keepdims=True)
    em_prob = np.exp(log_em - log_em_max)
    em_prob /= np.sum(em_prob, axis=2, keepdims=True) + 1e-12

    # ------------------------------------------------------------
    # Transition masks over (t-1, j)
    # ------------------------------------------------------------
    silent_to_silent = is_silent[:-1] & is_silent[1:]
    silent_to_talk_noevid = is_silent[:-1] & is_talk[1:] & no_evidence[1:]
    talk_noevid_to_silent = is_talk[:-1] & no_evidence[:-1] & is_silent[1:]
    silent_to_talk_evid = is_silent[:-1] & is_talk[1:] & has_evidence[1:]
    talk_evid_to_silent = is_talk[:-1] & has_evidence[:-1] & is_silent[1:]

    transition_specs = [
        ("silent_to_silent", silent_to_silent),
        ("silent_to_talk_no_evidence", silent_to_talk_noevid),
        ("talk_no_evidence_to_silent", talk_noevid_to_silent),
        ("silent_to_talk_with_evidence", silent_to_talk_evid),
        ("talk_with_evidence_to_silent", talk_evid_to_silent),
    ]

    rows = []

    for name, mask in transition_specs:
        eligible = np.argwhere(mask)  # rows are (t_prev, j)
        if eligible.shape[0] == 0:
            continue

        t_prev, j = eligible[rng.integers(0, eligible.shape[0])]
        t_prev = int(t_prev)
        j = int(j)
        t_curr = t_prev + 1

        prev_probs = em_prob[t_prev, j]
        curr_probs = em_prob[t_curr, j]

        rows.append({
            "transition_type": name,
            "t_prev": t_prev,
            "t_curr": t_curr,
            "j": j,
            "is_silent_prev": bool(is_silent[t_prev, j]),
            "is_silent_curr": bool(is_silent[t_curr, j]),
            "evid_class_prev": int(evid_class[t_prev, j]),
            "evid_class_curr": int(evid_class[t_curr, j]),
            "emiss_prev_json": json.dumps(prev_probs.tolist()),
            "emiss_curr_json": json.dumps(curr_probs.tolist()),
            "emiss_prev_argmax": int(np.argmax(prev_probs)),
            "emiss_curr_argmax": int(np.argmax(curr_probs)),
            "emiss_prev_maxprob": float(np.max(prev_probs)),
            "emiss_curr_maxprob": float(np.max(curr_probs)),
        })

        if not rows:
            return

        df = pd.DataFrame(rows)

        out_csv = f"{save_dir}__sampled_emissions_iter_{iteration}.csv"
        df.to_csv(
            out_csv,
            mode="a",
            header=not Path(out_csv).exists(),
            index=False,
        )

    return pd.DataFrame(rows)


def sample_transitions_for_transition_types_TJLKk(
    *,
    observations: np.ndarray,       # (T, J, D)
    one_hot_evidence: np.ndarray,    # (T, J, C)
    log_transitions: np.ndarray,     # (T, J, L, K, K)
    save_dir: str,
    iteration: int, 
    seed: int = 0,
) -> None:
    """
    Purpose:
        Randomly sample one example of several observation/evidence transition types
        (based on silent/talk and evidence labels) and save the corresponding
        transition probability tensors to a CSV file for diagnostics.

        Here transitions are provided per system-state, so the saved transition object
        per sampled example is an L×K×K tensor:
            P[t_prev, j, l, k_prev, k_next] = p(z_{t_curr}=k_next | z_{t_prev}=k_prev, s_{t_prev}=l, ...)

        Transition types sampled (one each, if available):
          1) silent -> silent
          2) silent -> talk (NO evidence in talk)
          3) talk (NO evidence) -> silent
          4) silent -> talk (WITH evidence in talk)
          5) talk (WITH evidence) -> silent

    Arguments:
        observations:
            Array of shape (T-1, J, D). Silence is defined as exact match to observations[0, 1].
        one_hot_evidence:
            One-hot evidence labels of shape (T, J, C). Class 0 is assumed "no evidence".
        log_transitions:
            Log transition values of shape (T-1, J, L, K, K), interpreted as:
                log_transitions[t_prev, j, l, k_prev, k_next]
                = log p(z_{t_curr}=k_next | z_{t_prev}=k_prev, s_{t_prev}=l, ...).
            These are converted to probabilities by a row-wise softmax over k_next for each (l, k_prev).
        save_dir:
            Base path used to construct the output CSV filename:
                f"{save_dir}__sampled_transitions_{iteration}.csv"
        iteration: training iteration
        seed:
            RNG seed used to randomly select one example per transition type.

    Returns:
        None. Appends rows to the CSV file on disk. If no eligible examples exist
        for any type, the function writes nothing.
    """
    obs = np.asarray(observations)
    evid = np.asarray(one_hot_evidence)[:-1]
    logP = np.asarray(log_transitions)

    T, J, D = obs.shape
    if logP.ndim != 5:
        raise ValueError(f"log_transitions must have 5 dims (T-1,J,L,K,K), got shape {logP.shape}")
    if logP.shape[0] != T or logP.shape[1] != J:
        raise ValueError(f"log_transitions shape {logP.shape} inconsistent with observations (T,J)=({T},{J})")
    L, K1, K2 = logP.shape[2], logP.shape[3], logP.shape[4]
    if K1 != K2:
        raise ValueError(f"log_transitions last two dims must be (K,K), got {(K1, K2)}")
    K = K1

    rng = np.random.default_rng(seed)

    # Silence logic: exact match to observations[0,1]
    silent_vec = obs[0, 1]
    is_silent = np.all(obs == silent_vec, axis=2)  # (T,J)
    is_talk = ~is_silent

    # Evidence logic: class 0 = no evidence
    evid_class = evidence_strength_from_onehot(evid)        # (T,J)
    has_evidence = evid_class != 0
    no_evidence = evid_class == 0

    # Transition masks over (t_prev, j)
    silent_to_silent = is_silent[:-1] & is_silent[1:]
    silent_to_talk_noevid = is_silent[:-1] & is_talk[1:] & no_evidence[1:]
    talk_noevid_to_silent = is_talk[:-1] & no_evidence[:-1] & is_silent[1:]
    silent_to_talk_evid = is_silent[:-1] & is_talk[1:] & has_evidence[1:]
    talk_evid_to_silent = is_talk[:-1] & has_evidence[:-1] & is_silent[1:]

    transition_specs = [
        ("silent_to_silent", silent_to_silent),
        ("silent_to_talk_no_evidence", silent_to_talk_noevid),
        ("talk_no_evidence_to_silent", talk_noevid_to_silent),
        ("silent_to_talk_with_evidence", silent_to_talk_evid),
        ("talk_with_evidence_to_silent", talk_evid_to_silent),
    ]

    rows = []
    for name, mask in transition_specs:
        eligible = np.argwhere(mask)  # rows are (t_prev, j)
        if eligible.shape[0] == 0:
            continue

        t_prev, j = eligible[rng.integers(0, eligible.shape[0])]
        t_prev = int(t_prev)
        j = int(j)
        t_curr = t_prev + 1

        # log transition tensor for this (t_prev, j): (L, K, K)
        logP_t = logP[t_prev, j]  # (L,K,K)

        # row-wise softmax over k_next for each (l, k_prev)
        row_max = np.max(logP_t, axis=2, keepdims=True)      # (L,K,1)
        P_t = np.exp(logP_t - row_max)
        P_t /= np.sum(P_t, axis=2, keepdims=True) + 1e-12    # (L,K,K)

        # Per-l global argmax and maxprob (helpful quick summary)
        per_l_maxprob = []
        per_l_argmax_prev = []
        per_l_argmax_next = []
        for l in range(L):
            flat_idx = int(np.argmax(P_t[l]))
            k_prev_star = flat_idx // K
            k_next_star = flat_idx % K
            per_l_maxprob.append(float(P_t[l, k_prev_star, k_next_star]))
            per_l_argmax_prev.append(int(k_prev_star))
            per_l_argmax_next.append(int(k_next_star))

        rows.append({
            "transition_type": name,
            "t_prev": t_prev,
            "t_curr": t_curr,
            "j": j,
            "L": int(L),
            "K": int(K),
            "is_silent_prev": bool(is_silent[t_prev, j]),
            "is_silent_curr": bool(is_silent[t_curr, j]),
            "evid_class_prev": int(evid_class[t_prev, j]),
            "evid_class_curr": int(evid_class[t_curr, j]),
            "trans_probs_json": json.dumps(P_t.tolist()),  # LxKxK object
            "per_l_global_maxprob_json": json.dumps(per_l_maxprob),
            "per_l_global_argmax_prev_json": json.dumps(per_l_argmax_prev),
            "per_l_global_argmax_next_json": json.dumps(per_l_argmax_next),
        })

    if not rows:
        return

    df = pd.DataFrame(rows)
    out_csv = f"{save_dir}__sampled_transitions_iter_{iteration}.csv"
    df.to_csv(out_csv, mode="a", header=not Path(out_csv).exists(), index=False)


def save_embedding_similarity_metrics(
    *,
    X: np.ndarray,                 # (T, J, D) normalized
    Y: np.ndarray,                 # (T, J, C) one-hot, class0 = no evidence
    save_dir: str,
    n_pairs: int = 2000,
    seed: int = 0,
) -> None:
    """
    Purpose:
        Compute embedding similarity diagnostics focused on separation between:
          - talk vs silence
          - talk(no evidence) vs silence
          - talk(with evidence) vs silence
        and cohesion/separation between evidence/no-evidence talk embeddings via random
        pairwise cosine similarities.

        Appends one row to a CSV table at:
            f"{save_dir}__embedding_similarity_metrics.csv"

    Arguments:
        X:
            Embedding tensor of shape (T, J, D). Assumed L2-normalized.
            Silence is defined as exact match to X[0, 1, :].
        Y:
            One-hot evidence tensor of shape (T, J, C). Assumes class 0 = "No evidence".
        save_dir:
            Base path used to construct output CSV filename.
        n_pairs:
            Number of random pairs used for each pairwise cosine estimate.
        seed:
            RNG seed for reproducible random pairing.

    Returns:
        None. Appends a single row to the CSV file on disk.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    Y = np.asarray(Y)

    T, J, D = X.shape
    C = Y.shape[2]

    # ---- masks: silent / talk ----
    silent_vec = X[0, 1, :]  # your convention
    is_silent = np.all(X == silent_vec.reshape(1, 1, D), axis=2)  # (T,J)
    is_talk = ~is_silent

    # ---- evidence masks among talk ----
    evid_class = np.argmax(Y, axis=2)  # (T,J)
    talk_no_evid = is_talk & (evid_class == 0)
    talk_evid = is_talk & (evid_class != 0)

    # ---- cosine to silence (dot product; normalized) ----
    def mean_cos_to_silence(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        Xs = X[mask]  # (N,D)
        return float(np.mean(Xs @ silent_vec))

    mean_talk_sil = mean_cos_to_silence(is_talk)
    mean_sil_sil = mean_cos_to_silence(is_silent)
    mean_noevid_sil = mean_cos_to_silence(talk_no_evid)
    mean_evid_sil = mean_cos_to_silence(talk_evid)

    # ---- random pairwise cosine helpers ----
    def mean_pairwise_cos(A: np.ndarray, B: np.ndarray, n: int) -> float:
        """
        A: (NA,D), B: (NB,D). Returns mean over n random pairs of dot products.
        If A is B (same object), samples two indices (i != j) when possible.
        """
        NA = A.shape[0]
        NB = B.shape[0]
        if NA == 0 or NB == 0:
            return float("nan")
        if NA == 1 and NB == 1 and A is B:
            return float("nan")

        i = rng.integers(0, NA, size=n)
        j = rng.integers(0, NB, size=n)

        if A is B and NA >= 2:
            # enforce i != j for same-set pairing (best effort)
            same = (i == j)
            # resample the j's that collide
            j[same] = (j[same] + rng.integers(1, NA, size=np.sum(same))) % NA

        return float(np.mean(np.sum(A[i] * B[j], axis=1)))

    X_noevid = X[talk_no_evid]  # (N0,D)
    X_evid = X[talk_evid]       # (N1,D)

    mean_evid_vs_noevid = mean_pairwise_cos(X_evid, X_noevid, n_pairs)
    mean_noevid_vs_noevid = mean_pairwise_cos(X_noevid, X_noevid, n_pairs)
    mean_evid_vs_evid = mean_pairwise_cos(X_evid, X_evid, n_pairs)

    # counts for context
    N_talk = int(np.sum(is_talk))
    N_noevid = int(np.sum(talk_no_evid))
    N_evid = int(np.sum(talk_evid))

    row = pd.DataFrame([{
        "T": int(T),
        "J": int(J),
        "D": int(D),
        "N_talk": N_talk,
        "N_talk_no_evidence": N_noevid,
        "N_talk_with_evidence": N_evid,
        "mean_cos_sil_to_silence":  mean_sil_sil,
        "mean_cos_talk_to_silence": mean_talk_sil,
        "mean_cos_talk_no_evidence_to_silence": mean_noevid_sil,
        "mean_cos_talk_with_evidence_to_silence": mean_evid_sil,
        "n_pairs": int(n_pairs),
        "mean_pair_cos_evidence_vs_no_evidence": mean_evid_vs_noevid,
        "mean_pair_cos_no_evidence_vs_no_evidence": mean_noevid_vs_noevid,
        "mean_pair_cos_evidence_vs_evidence": mean_evid_vs_evid,
        "seed": int(seed),
    }])

    out_csv = f"{save_dir}__embedding_similarity_metrics.csv"
    row.to_csv(out_csv, mode="a", header=not Path(out_csv).exists(), index=False)


def save_transition_type_counts_per_entity(
    *,
    observations: np.ndarray,      # (T, J, D)   (can be X)
    one_hot_evidence: np.ndarray,  # (T, J, C)   (can be Y)
    save_dir: str,
) -> None:
    """
    Purpose:
        Compute per-entity counts of specific transition types based on silent/talk
        status and evidence labels, and save them to a CSV file.

        Transition types counted for each entity j (over t=1..T-1):
          1) silent -> silent
          2) silent -> talk (NO evidence in talk)
          3) silent -> talk (WITH evidence in talk)
          4) talk (NO evidence) -> silent
          5) talk (WITH evidence) -> silent

    Arguments:
        observations:
            Array of shape (T, J, D). Silence is defined as exact match to observations[0, 1].
        one_hot_evidence:
            One-hot evidence labels of shape (T, J, C). Class 0 is assumed "No evidence".
        save_dir:
            Base path used to construct the output CSV filename:
                f"{save_dir}__transition_type_counts_by_entity.csv"
        file_id:
            Identifier to store in the table (e.g., CSV filename).

    Returns:
        None. Appends J rows (one per entity) to the CSV file on disk.
    """
    obs = np.asarray(observations)
    evid = np.asarray(one_hot_evidence)

    T, J, D = obs.shape

    # Silence logic: exact match to observations[0,1]
    silent_vec = obs[0, 1]
    is_silent = np.all(obs == silent_vec.reshape(1, 1, D), axis=2)  # (T,J)
    is_talk = ~is_silent

    # Evidence class logic (class 0 = no evidence)
    evid_class = np.argmax(evid, axis=2)  # (T,J)
    has_evidence = evid_class != 0
    no_evidence = evid_class == 0

    # Masks over transitions (t-1 -> t) for each j: shape (T-1, J)
    silent_to_silent = is_silent[:-1] & is_silent[1:]
    silent_to_talk_noevid = is_silent[:-1] & is_talk[1:] & no_evidence[1:]
    silent_to_talk_evid = is_silent[:-1] & is_talk[1:] & has_evidence[1:]
    talk_noevid_to_silent = is_talk[:-1] & no_evidence[:-1] & is_silent[1:]
    talk_evid_to_silent = is_talk[:-1] & has_evidence[:-1] & is_silent[1:]

    # Count per entity (sum over t)
    rows = []
    for j in range(J):
        rows.append({
            "j": int(j),
            "count_silent_to_silent": int(np.sum(silent_to_silent[:, j])),
            "count_silent_to_talk_no_evidence": int(np.sum(silent_to_talk_noevid[:, j])),
            "count_silent_to_talk_with_evidence": int(np.sum(silent_to_talk_evid[:, j])),
            "count_talk_no_evidence_to_silent": int(np.sum(talk_noevid_to_silent[:, j])),
            "count_talk_with_evidence_to_silent": int(np.sum(talk_evid_to_silent[:, j])),
        })

    df = pd.DataFrame(rows)

    out_csv = f"{save_dir}__transition_type_counts_by_entity_real_data.csv"
    df.to_csv(out_csv, mode="a", header=not Path(out_csv).exists(), index=False)



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


def extract_pooled_ar_pairs_by_evidence_with_silent_predictor(
    observations: np.ndarray,                 # (T,J,D)
    evid_onehot: np.ndarray,                  # (T,J,C)
    example_end_times: np.ndarray,            # (N+1,)
    mask_observations: Optional[np.ndarray] = None,  # (T,J) bool
    *,
    want_evidence: bool,                      # False -> class0, True -> any non-zero class
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Purpose: Returns pooled predictors/outcomes/weights across all entities j for AR training,
        using ONLY pairs where:
            - predictor x_{t-1} is SILENT (exact match to observations[0][1])
            - outcome   x_t     is NON-SILENT (not equal to observations[0][1])
            - label at time t matches the requested bucket:
                * want_evidence=False: evid_onehot[t,j,0] == 1  (no evidence)
                * want_evidence=True:  evid_onehot[t,j,0] == 0  (evidence in any non-zero class; assumes true one-hot)
            - excludes the first timestep of each example (via sample_weights helper)
            - respects mask_observations if provided

    Arguments:
        observations: np.array of shape (T,J,D) where the (t,j)-th entry is in R^D.
        evid_onehot: One hot vector labels that are the annotated class evidence of mechanistic reasoning.
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.
        want_evidence: bool that for False -> class0, True -> any non-zero class.

    Returns:
        predictors: np.ndarray of shape (N,D)
            The pooled predictor vectors x_{t-1}, all of which correspond to SILENT observations.
        outcomes: np.ndarray of shape (N,D)
            The pooled outcome vectors x_t, all of which correspond to NON-SILENT observations.
        weights: np.ndarray of shape (N,)
            Sample weights aligned with the returned predictor/outcome pairs.
    """
    obs = np.asarray(observations)
    evid = np.asarray(evid_onehot)

    T, J, D = obs.shape
    assert evid.shape[0] == T and evid.shape[1] == J, "evid_onehot must match (T,J,*) of observations"

    # Silent template (exact)
    silent_vec = np.asarray(observations[0][1])

    sample_weights = make_sample_weights_which_mask_the_initial_timestep_for_each_event(
        obs, example_end_times, mask_observations
    )  # (T,J)

    # Label condition applied at outcome time t
    if want_evidence:
        label_ok = (evid[:, :, 0] == 0)   # evidence = any non-zero class (assuming one-hot)
    else:
        label_ok = (evid[:, :, 0] == 1)   # no-evidence class

    ok = label_ok & (sample_weights > 0)  # valid outcome times

    t_idx, j_idx = np.where(ok)
    keep = (t_idx >= 1)                   # need predictors at t-1
    t_idx = t_idx[keep]
    j_idx = j_idx[keep]

    predictors = obs[t_idx - 1, j_idx, :]
    outcomes   = obs[t_idx,     j_idx, :]

    # predictor must be silent; outcome must be non-silent (exact comparison)
    pred_is_silent = np.all(predictors == silent_vec[None, :], axis=1)
    out_is_silent  = np.all(outcomes   == silent_vec[None, :], axis=1)

    keep2 = pred_is_silent & (~out_is_silent)

    predictors = predictors[keep2]
    outcomes   = outcomes[keep2]
    weights    = sample_weights[t_idx, j_idx].astype(float)[keep2]

    return predictors, outcomes, weights

def mean_gaussian_loglik_ar(
    predictors: np.ndarray,  # (N, D)
    outcomes: np.ndarray,    # (N, D)
    A: np.ndarray,           # (D, D)
    b: np.ndarray,           # (D,)
    Q: np.ndarray,           # (D, D)
) -> float:
    """
    Purpose: Return the mean log-likelihood of the data for a set of gaussian density parameters. 
    Arguments: 
        predictors: x_{t-1} datapoints 
        outcomes: x_t datapoints 
        As : has shape (J, K, D, D) 
        bs : has shape (J, K, D) for the current entity 
        Qs : has shape (J, K, D,D) for the covariance matrix 
            Each Qs[j] is a covariance matrix
    Return: mean_n log N(y_n | A x_n + b, Q)."""
    predictors = np.asarray(predictors)
    outcomes = np.asarray(outcomes)
    As = np.asarray(A)
    bs = np.asarray(b)
    Qs = np.asarray(Q)

    N, D = outcomes.shape

    yhat = (As @ predictors.T).T + bs
    resid = outcomes - yhat

    Q_sym = 0.5 * (Qs + Qs.T)
    L = np.linalg.cholesky(Q_sym)

    z = np.linalg.solve(L, resid.T).T
    maha = np.sum(z**2, axis=1)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))

    loglik = -0.5 * (maha + logdet + D * np.log(2.0 * np.pi))
    return float(np.mean(loglik))

def weighted_ridge_fit_multioutput(
    X: np.ndarray,          # (N, D+1)
    Y: np.ndarray,          # (N, D)
    w: np.ndarray,          # (N,)
    alpha: float,           # ridge strength
    *,
    penalize_intercept: bool = True,
    jitter: float = 1e-8,
) -> np.ndarray:
    """
    Purpose: Return the parameters associated with the Ridge regression. 
    Arguments: 
        X: data at the previous time-point x_t
        Y: data at the next time-point x_{t+1}
        w: diagonal of the response weights matrix 
    Return: B of shape (D+1, D) such that Y ≈ X @ B, with weighted ridge.
    Last column of X is assumed to be the intercept column (all ones).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    w = np.asarray(w, dtype=float)

    # Guard: if all weights are ~0, you'll get nonsense
    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError("Sum of weights is non-positive; cannot fit.")

    # Compute XtWX and XtWY without forming full diagonal W
    # XtWX = X^T W X = (X * w[:,None])^T X
    Xw = X * w[:, None]                       # (N, D+1)
    XtWX = Xw.T @ X                           # (D+1, D+1)
    XtWY = Xw.T @ Y                           # (D+1, D)

    p = X.shape[1]                            # D+1
    P = np.eye(p)
    if not penalize_intercept:
        P[-1, -1] = 0.0                       # don't penalize intercept

    # Regularize / stabilize
    A = XtWX + alpha * P + jitter * np.eye(p) # (D+1, D+1)

    # Solve for B: (XtWX + alpha P) B = XtWY
    B = np.linalg.solve(A, XtWY)              # (D+1, D)
    return B


def extract_pooled_ar_pairs_to_silence(
    *,
    observations: np.ndarray,                    # (T,J,D)
    example_end_times: np.ndarray,               # (N+1,) with [-1, t1, ..., tN]
    mask_observations: np.ndarray | None,        # (T,J) bool
    silence_vec: np.ndarray,                     # (D,)
    from_non_silent_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Purpose: 
    Arguments: 
        observations: np.array of shape (T,J,D) where the (t,j)-th entry is in R^D.
        example_end_times: optional, has shape (N+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across N i.i.d "examples".
            If there are N examples, then along with the observations, we store
            end_times=[-1, t_1, …, t_N], where t_n is the timestep at which the n-th example ended.
        mask_observations: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            observations[t,j] should be utilized in inference and False otherwise.
        silence_vec: set as observations[0][1] throughout 
        from_non_silent_only: Boolean flag True if we only care about non-silent to silent transitions rather than silent to silent
    Returns: pooled (predictors, outcomes) for all (t-1 -> t) pairs where outcome at t is SILENCE.
        Optionally restrict predictors at t-1 to be NON-silent (true "to-silence" transitions).
        Respects example boundaries so we don't create cross-example pairs.
    """
    obs = np.asarray(observations)
    T, J, D = obs.shape
    s = np.asarray(silence_vec).reshape(1, 1, D)

    # exact match silence mask
    is_silent = np.all(obs == s, axis=-1)  # (T,J)

    # valid pair indices are t=1..T-1, but exclude t that are starts of new examples
    # example_end_times = [-1, end_1, ..., end_N]; starts are end_{n-1}+1
    starts = set(int(example_end_times[i - 1] + 1) for i in range(1, len(example_end_times)))
    valid_t = np.ones(T, dtype=bool)
    for st in starts:
        if 0 <= st < T:
            valid_t[st] = False  # cannot use pair (st-1 -> st)

    # outcomes at time t (t>=1)
    t_idx = np.arange(1, T)
    ok_t = valid_t[t_idx]  # (T-1,)

    outcome_is_silence = is_silent[1:, :]         # (T-1,J)
    if from_non_silent_only:
        predictor_is_non_silence = ~is_silent[:-1, :]  # (T-1,J)
    else:
        predictor_is_non_silence = np.ones((T - 1, J), dtype=bool)

    keep = outcome_is_silence & predictor_is_non_silence  # (T-1,J)
    keep &= ok_t[:, None]

    if mask_observations is not None:
        m = np.asarray(mask_observations).astype(bool)
        keep &= m[1:, :] & m[:-1, :]

    if not np.any(keep):
        return np.zeros((0, D)), np.zeros((0, D))

    predictors = obs[:-1, :, :][keep]  # (N_pairs, D)
    outcomes   = obs[1:,  :, :][keep]  # (N_pairs, D)
    return predictors, outcomes



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

