import pandas as pd
import os
import numpy as np
from typing import List, Union
from pathlib import Path
from ssm.util import find_permutation
from scipy.stats import spearmanr
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.lines as mlines
from typing import List, Sequence, Optional

from utilities.types import(
    NumpyArray1D,
    NumpyArray2D,
    NumpyArray3D,
    JaxNumpyArray1D, 
    JaxNumpyArray2D,
)
from utilities.util import evidence_strength_from_onehot, pearson_corr, speaker_index_per_t, _to_listed_pink_gradient, ensure_dir, make_tasteful_pink_cmap, distinct_colors, silent_mask_from_observations, spearman_corr_safe
from params import AllParameters, dims_from_params
from compute_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summaries_NUMPY,
    HMM_Posterior_Summary,
)

"""
Computes official metrics and plots visuals for the entire HSRDM: latent state probabilistic dynamics. 
"""

def plot_elbo(elbo_values, plots_dir, example_end_times, J, filename="elbo_over_iterations.pdf"):
    """
    Purpose: Plot ELBO over iterations and save to plots_dir.

    Arguments: 
        elbo_values : list or np.ndarray
            Sequence of ELBO values (inlcudes ELBO computation at every step in training VES-step, VEZ-step, M-step for each param).
        plots_dir : pathlib.Path or str
            Directory where the plot will be saved.
        filename : str
            Name of the output image file.
    """

    elbo_values = np.asarray(elbo_values)
    data_normalized_values = elbo_values / (example_end_times[-1]*J)

    plt.figure(figsize=(8, 5))
    plt.plot(elbo_values, linewidth=2)
    plt.title("ELBO Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.grid(True, linestyle="--", alpha=0.5)

    save_path = Path(plots_dir) / filename
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[plot_elbo] Saved ELBO plot to {save_path}")


def get_entity_correlation_table(
    probs: np.ndarray,         # (T,J,K)
    evid_onehot: np.ndarray,   # (T,J,C)
    ks=(0, 1,2,3),
    spearman: bool = True,
    out_csv: str | None = None,
) -> pd.DataFrame:
    """
    Purpose: For each entity j, compute corr( probs[t+1, j, k], evidence_strength[t, j] ) for k in ks.
    This metric provides a table of correlation values of whether an entity is more likely to be in a certain silent state at time t when
    there was evidence of mechanistic reasoning in the previous entity observation. 

    Arguments: 
        Probs: Posterior probabilities output from the training (T, J, K)
        evid_onehot: One hot vector labels that are the annotated class evidence of mechanistic reasoning 
        ks: a tuple of values of k-states (0 or 1 in this case). Both 0 and 1 represent silent states. 
        out_csv: Name of the csv file that the table of results will save to 

    Returns: a DataFrame with one row per entity giving the correlation between the poserior probability at t+1 of being in state k and the evidence class label in the previous observation
    """
    probs = np.asarray(probs)
    T, J, K = probs.shape
    evid_strength = evidence_strength_from_onehot(evid_onehot)  # (T,J)

    rows = []
    for j in range(J):
        y = evid_strength[:-1, j]  # (T-1,)
        row = {"entity": j}
        for k in ks:
            x = probs[1:, j, k]    # (T-1,)
            if spearman:
                row[f"spearman_corr_next_p_k{k}_vs_prev_evidence_self"] = spearman_corr_safe(x, y)
            row[f"pearson_corr_next_p_k{k}_vs_prev_evidence_self"] = pearson_corr(x,y)
        rows.append(row)

    df = pd.DataFrame(rows)

    if out_csv is not None:
        df.to_csv(f"{out_csv}entity_correlation_table.csv", index=False)

    return df

def get_entity_correlation_metric(
    probs: np.ndarray,         # (T,J,K)
    evid_onehot: np.ndarray,   # (T,J,C)
    ks=(0, 1,2,3),
) -> List:
    """
    Purpose: For each entity j, compute corr( probs[t+1, j, k], evidence_strength[t, j] ) for k in ks.
    This metric is just the list of correlation values of whether an entity is more likely to be in a certain silent state at time t when
    there was evidence of mechanistic reasoning in the previous entity observation. 

    Arguments: 
        Probs: Posterior probabilities output from the training (T, J, K)
        evid_onehot: One hot vector labels that are the annotated class evidence of mechanistic reasoning 
        ks: a tuple of values of k-states (0 or 1 in this case). Both 0 and 1 represent silent states. 

    Returns: a list with one row per entity giving the correlation between the poserior probability at t+1 of being in state k and the evidence class label in the previous observation
    """
    probs = np.asarray(probs)
    T, J, K = probs.shape
    evid_strength = evidence_strength_from_onehot(evid_onehot)  # (T,J)

    rows = []
    for j in range(J):
        y = evid_strength[:-1, j]  # (T-1,)
        row = {"entity": j}
        for k in ks:
            x = probs[1:, j, k]    # (T-1,)
            row[f"spearman_corr_next_p_k{k}_vs_prev_evidence_self"] = spearman_corr_safe(x, y)
            row[f"pearson_corr_next_p_k{k}_vs_prev_evidence_self"] = pearson_corr(x,y)
        rows.append(row)

    return rows

def get_system_silent_correlation_table(
    probs: np.ndarray,        # (T,J,K)
    evid_onehot: np.ndarray,  # (T,J,C)
    observations: np.ndarray, # (T,J,D)
    out_csv: str | None = None,
) -> pd.DataFrame:
    """
    Purpose: Metric 1 (Silent -> Silent), entity-wise, ALL K states.
        For each entity j, and for each k in [0..K-1], compute Spearman corr between:
        x_t = evidence strength of speaker at time t
        y_t = probs[t+1, j, k]
        using only times t where:
        - j != speaker(t)
        - j is silent at t
        - j is silent at t+1

    Arguements: 
        Probs: Posterior probabilities output from the training (T, J, K)
        evid_onehot: One hot vector labels that are the annotated class evidence of mechanistic reasoning 
        obervations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        out_csv: Name of the csv file that the table of results will save to 

    Returns: one row per entity, with columns:
      spearman_corr_k{k}, for all k
      n_samples (same across all k for that entity)
    """
    probs = np.asarray(probs)
    T, J, K = probs.shape

    evid_strength = evidence_strength_from_onehot(evid_onehot)  # (T,J)
    speaker_idx = speaker_index_per_t(observations)             # (T,)
    silent = silent_mask_from_observations(observations)  # (T,J)

    rows = []
    for j in range(J):
        xs = []
        ys_by_k = [[] for _ in range(K)]

        for t in range(T - 1):
            if speaker_idx[t] == j:
                continue
            if not (silent[t, j] and silent[t + 1, j]):
                continue

            sp = speaker_idx[t]
            x_t = float(evid_strength[t, sp])
            xs.append(x_t)

            # collect y for every k at t+1
            for k in range(K):
                ys_by_k[k].append(float(probs[t + 1, j, k]))

        x = np.asarray(xs, dtype=float)
        row = {"entity": j, "metric": "silent_to_silent", "n_samples": int(x.size)}

        for k in range(K):
            y = np.asarray(ys_by_k[k], dtype=float)
            rho = spearman_corr_safe(x, y)
            row[f"spearman_corr_k{k}"] = rho
            pe = pearson_corr(x, y)
            row[f"pearson_corr_k{k}"] = pe
   
        rows.append(row)

    df = pd.DataFrame(rows)

    if out_csv:
        df.to_csv(f"{out_csv}system_silent_correlation_table.csv", index=False)

    return df


def get_k1_posterior_for_speaker_transition_to_silence_by_mech_evidence(
    probs: np.ndarray,        # (T, J, K)
    evid_onehot: np.ndarray,  # (T, J, C)
    observations: np.ndarray, # (T, J, D)
    k_target: int = 1,
    require_not_silent_at_t: bool = True,
    std_dev: bool = False,
    out_csv: str | None = None,
) -> pd.DataFrame:
    """
    Purpose: Look at each time t, take the speaker sp = speaker(t).
    Keep only events where the speaker is silent at t+1 (and optionally not silent at t),
    then split by whether the speaker has mechanistic evidence at t.

    Arguements: 
    Probs: Posterior probabilities output from the training (T, J, K)
    evid_onehot: One hot vector labels that are the annotated class evidence of mechanistic reasoning 
    obervations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
    k_target: this state is k=1 in our experiments 
    require_not_silent_at_t: must be set to true to only care about the speaker at time t 
    out_csv: Name of the csv file that the table of results will save to 

    Returns: mean posterior P(z_{t+1, sp} = k_target) for each scenario + counts.
    """

    probs = np.asarray(probs)
    evid_onehot = np.asarray(evid_onehot)
    observations = np.asarray(observations)

    T, J, K = probs.shape
    assert evid_onehot.shape[0] == T and evid_onehot.shape[1] == J, "evid_onehot must be (T,J,C)"
    assert observations.shape[0] == T and observations.shape[1] == J, "observations must be (T,J,D)"
    assert 0 <= k_target < K

    speaker_idx = speaker_index_per_t(observations)          # (T,)
    silent = silent_mask_from_observations(observations)     # (T,J) bool
    evid_present = np.any(evid_onehot[..., 1:] > 0, axis=-1) # (T,J) bool

    mech_vals, no_mech_vals = [], []

    for t in range(T - 1):
        sp = int(speaker_idx[t])

        # speaker must be silent at t+1
        if not silent[t + 1, sp]:
            continue

        # optionally require an actual transition: not silent at t -> silent at t+1
        if require_not_silent_at_t and silent[t, sp]:
            continue

        val = float(probs[t + 1, sp, k_target])

        if bool(evid_present[t, sp]):
            mech_vals.append(val)
        else:
            no_mech_vals.append(val)

    def summarize(vals, label):
        arr = np.asarray(vals, dtype=float)
        return {
            "scenario": label,
            "k_target": int(k_target),
            "n_events": int(arr.size),
            "mean_posterior_k": float(np.mean(arr)) if arr.size else np.nan,
            "std_posterior_k": float(np.std(arr)) if std_dev and arr.size else np.nan,
            "min_count": (arr < 0.01).sum(),
            "max_count": (arr > 0.99).sum(),
        }

    df = pd.DataFrame([
        summarize(mech_vals, "mech_evidence"),
        summarize(no_mech_vals, "no_mech_evidence"),
    ])

    if out_csv:
        df.to_csv(f"{out_csv}speaker_transition_to_silence_k{k_target}_by_mech_evidence.csv", index=False)

    return df


def get_transition_posteriors_conditioned_on_speaker_evidence(
    probs: np.ndarray,        # (T, J, K)
    evid_onehot: np.ndarray,  # (T, J, C)
    observations: np.ndarray, # (T, J, D)
    k_silent_to_talk: int = 3,
    k_silent_to_silent: int = 1,
    std_dev: bool = False,
    out_csv: str | None = None,
) -> pd.DataFrame:
    """
    Purpose: For ALL time points t and entities j, compute conditional averages:

    Condition is determined by the SPEAKER at time t:
      - evidence_present(t) = True iff evid_onehot[t, speaker(t), 0] is NOT the active class
        (equivalently: any class in positions 1..C-1 is active)

    Then compute:
      1) mean probs[t+1, j, k_silent_to_talk] over (t,j) where silent(t,j)=True and silent(t+1,j)=False
      2) mean probs[t+1, j, k_silent_to_silent] over (t,j) where silent(t,j)=True and silent(t+1,j)=True

    Arguements: 
        Probs: Posterior probabilities output from the training (T, J, K)
        evid_onehot: One hot vector labels that are the annotated class evidence of mechanistic reasoning 
        obervations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        k_silent_to_talk: this state is k=3 in our experiments 
        k_silent_to_silent: this state is k=1 in our experiments 
        out_csv: Name of the csv file that the table of results will save to 


    Returns:  a small 2x2 table (with counts).
    """

    probs = np.asarray(probs)
    evid_onehot = np.asarray(evid_onehot)
    observations = np.asarray(observations)

    T, J, K = probs.shape
    assert evid_onehot.shape[0] == T and evid_onehot.shape[1] == J, "evid_onehot must be (T,J,C)"
    assert observations.shape[0] == T and observations.shape[1] == J, "observations must be (T,J,D)"
    assert 0 <= k_silent_to_talk < K
    assert 0 <= k_silent_to_silent < K

    # Speaker index per time
    speaker_idx = speaker_index_per_t(observations)  # (T,)

    # Silent mask per (t,j)
    silent = silent_mask_from_observations(observations)  # (T,J) bool

    # Evidence present for each (t,j): True iff NOT class-0 (i.e., any of classes 1.. is active)
    # Robust to one-hot being "soft": treat > 0 as active evidence.
    evid_present = np.any(evid_onehot[..., 1:] > 0, axis=-1)  # (T,J) bool

    buckets = {
        ("evidence", "silent_to_talk"): [],
        ("evidence", "silent_to_silent"): [],
        ("no_evidence", "silent_to_talk"): [],
        ("no_evidence", "silent_to_silent"): [],
    }

    for t in range(T - 1):
        sp = int(speaker_idx[t])
        cond = "evidence" if bool(evid_present[t, sp]) else "no_evidence"

        # transitions for all entities j
        s_t = silent[t]      # (J,)
        s_tp1 = silent[t + 1]  # (J,)

        # silent -> talk  : silent at t, NOT silent at t+1
        m_st = s_t & (~s_tp1)
        # silent -> silent: silent at t, silent at t+1
        m_ss = s_t & (s_tp1)

        if np.any(m_st):
            buckets[(cond, "silent_to_talk")].extend(probs[t + 1, m_st, k_silent_to_talk].astype(float).tolist())
        if np.any(m_ss):
            buckets[(cond, "silent_to_silent")].extend(probs[t + 1, m_ss, k_silent_to_silent].astype(float).tolist())

    rows = []
    for cond in ["evidence", "no_evidence"]:
        for trans in ["silent_to_talk", "silent_to_silent"]:
            vals = np.asarray(buckets[(cond, trans)], dtype=float)
            rows.append({
                "condition": cond,  # based on speaker(t)
                "transition": trans,
                "k_used": (k_silent_to_talk if trans == "silent_to_talk" else k_silent_to_silent),
                "n_pairs": int(vals.size),
                "mean_posterior": float(np.mean(vals)) if vals.size else np.nan,
                "std_posterior": float(np.std(vals)) if std_dev and vals.size else np.nan,
                "min_count": (vals < 0.01).sum(),
                "max_count": (vals > 0.99).sum(),
            })

    df = pd.DataFrame(rows)

    if out_csv:
        df.to_csv(f"{out_csv}transition_posteriors_conditioned_on_speaker_evidence.csv", index=False)

    return df





def get_system_correlation_metric(
    sys_probs: np.ndarray,         # (T, L)
    evid_onehot: np.ndarray,       # (T, J, C)
    observations: np.ndarray,      # (T, J, D) to identify speaker per t
    ls: Optional[Sequence[int]] = None,
) -> List[dict]:
    """
    Purpose:
        Compute corr( sys_probs[t+1, l], evidence_strength[t, speaker(t)] ) for l in ls.

    Arguments: 
        sys_probs: posterior probabilities of system states
        evid_onehot: One hot vector labels that are the annotated class evidence of mechanistic reasoning 
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        ls: possible latent states 
    Returns:
        A list with a single dict row containing correlations for each l.
    """
    P = np.asarray(sys_probs)
    if P.ndim != 2:
        raise ValueError(f"Expected sys_probs shape (T,L); got {P.shape}")
    T, L = P.shape
    if T < 2:
        raise ValueError("Need T >= 2 for lag-1 correlation.")

    evid_strength = evidence_strength_from_onehot(evid_onehot)  # (T, J)
    speaker_idx = speaker_index_per_t(observations)             # (T,)

    evid_strength = np.asarray(evid_strength)
    speaker_idx = np.asarray(speaker_idx)

    if evid_strength.shape[0] != T or speaker_idx.shape[0] != T:
        raise ValueError("T mismatch among sys_probs, evid_onehot, observations.")


    e_speaker = evid_strength[np.arange(T), speaker_idx]        # (T,)

    y = e_speaker[:-1]   

    if ls is None:
        ls = range(L)


    valid = (speaker_idx[:-1] >= 0)
    yv = y[valid]

    row = {}
    for l in ls:
        x = P[1:, l]    
        xv = x[valid]
        row[f"spearman_corr_next_p_l{l}_vs_prev_speaker_evidence"] = spearman_corr_safe(xv, yv)
        row[f"pearson_corr_next_p_l{l}_vs_prev_speaker_evidence"] = pearson_corr(xv, yv)

    return [row]







def plot_speaking_and_evidence_boxes(
    evid_onehot: np.ndarray,   # (T,J,C)
    observations: np.ndarray,  # (T,J,D)
    colors: list,
    plot_dir: str = "/path/to/output",
    figsize_per_entity: float = 0.6,
    inches_per_timestep: float = 0.012, 
    max_fig_width: float = 40.0,    

):
    """
    Purpose: Provide a plot of the "ground truth" behavior for all entities at different points in time. This would demonstrate whether they are talking 
    or silent at each time point and then also if they are talking, if there is evidence of mechanistic reasoning. 

    Arguments: 
        evid_onehot: One hot vector labels that are the annotated class evidence of mechanistic reasoning
        observations: np.array of shape (T,J,D) where the (t,j)-th entry isin R^D
        colors: list of colors that will be transformed into a colormap 
        figsize_per_entity: figsize for each sub-plot 
        plot_dir: str of the path to save the plot 
        inches_per_timestep: float = 0.012, Longer inches per timestep to really see the distinctions in colors 
        max_fig_width: float = 40.0 Max width because the number of timesteps is long 

    Returns: 
    A plot of the ground truth behavior for the entities (not the posterior probabilities). 
    """
    ensure_dir(plot_dir)

    obs = np.asarray(observations)
    T, J, D = obs.shape
    C = np.asarray(evid_onehot).shape[-1]

    evid_strength = evidence_strength_from_onehot(evid_onehot)  # (T,J), 0..C-1
    speaker_idx = speaker_index_per_t(observations)             # (T,)

    values = np.zeros((J, T), dtype=int)
    values[speaker_idx, np.arange(T)] = evid_strength[np.arange(T), speaker_idx] + 1

    # ---- COLORMAP: silence = pure white ----
    evidence_cmap = _to_listed_pink_gradient(colors=colors, num_levels=C)
    silence_white = (1.0, 1.0, 1.0, 1.0)

    cmap = ListedColormap([silence_white] + list(evidence_cmap.colors))

    # ---- FIGURE SIZE: scale with time ----
    fig_w = min(max_fig_width, max(12.0, T * inches_per_timestep))
    fig_h = max(2.5, figsize_per_entity * J)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        values,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=C,
    )

    ax.set_xlabel("time t")
    ax.set_ylabel("entity j")
    ax.set_yticks(np.arange(J))
    ax.set_yticklabels([str(j) for j in range(J)])

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_ticks(np.arange(C + 1))
    cbar.set_ticklabels(["silence"] + [f"evidence {c}" for c in range(C)])

    plt.tight_layout()

    out_path = os.path.join(plot_dir, "entity_evidence.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_posteriors_per_entity(
    probs: np.ndarray,     # (T,J,K)
    colors: list = None,   # ignored for non-highlight lines now; kept for API compatibility
    max_cols: int = 3,
    figsize_per_subplot=(5.2, 2.4),
    show_legend: bool = True,
    plot_dir: str = "/path/to/output",
    highlight_ks_by_entity=None,
    deep_pink=(0.86, 0.11, 0.52, 1.0),
    *,
    t_start: int | None = None,
    t_end: int | None = None,
    keep_original_time: bool = True,
    filename: str = "entity_posteriors.pdf", 
):
    """
    Purpose: Plot posterior probabilities probs[t, j, k] over time, optionally restricted
    to a time window [t_start, t_end).

    Arguments:
        t_start: first time index to include (inclusive). None -> 0.
        t_end: last time index to include (exclusive). None -> T.
        keep_original_time: if True, x-axis shows original t indices; otherwise reindex to 0..len-1.
        filename: output PDF filename.
    Returns: 
        Plot of the posterior probabilities for each entity for a designated time window. 

    """

    probs = np.asarray(probs)
    T, J, K = probs.shape

    # ---- resolve interval ----
    if t_start is None:
        t_start = 0
    if t_end is None:
        t_end = T

    # allow negative indexing like python slices
    if t_start < 0:
        t_start = T + t_start
    if t_end < 0:
        t_end = T + t_end

    # clip to valid range
    t_start = int(np.clip(t_start, 0, T))
    t_end = int(np.clip(t_end, 0, T))

    if t_end <= t_start:
        raise ValueError(f"Empty/invalid interval: t_start={t_start}, t_end={t_end}, T={T}")

    probs_win = probs[t_start:t_end, :, :]  # (T_win, J, K)
    T_win = probs_win.shape[0]

    if keep_original_time:
        x = np.arange(t_start, t_end)
    else:
        x = np.arange(T_win)

    # Distinct colors for all ks (used unless highlighted)
    line_colors = distinct_colors(K)

    # Normalize highlight_ks_by_entity into dict: j -> set(ks)
    highlight_map = {j: set() for j in range(J)}
    if highlight_ks_by_entity is not None:
        if isinstance(highlight_ks_by_entity, dict):
            for j, ks in highlight_ks_by_entity.items():
                highlight_map[int(j)] = set(int(k) for k in ks)
        else:
            for j in range(min(J, len(highlight_ks_by_entity))):
                ks = highlight_ks_by_entity[j]
                highlight_map[j] = set(int(k) for k in (ks if isinstance(ks, (list, tuple, set)) else [ks]))

    ncols = min(max_cols, J)
    nrows = int(np.ceil(J / ncols))

    fig_w = figsize_per_subplot[0] * ncols
    fig_h = figsize_per_subplot[1] * nrows
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(fig_w, fig_h),
        sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    # ---- plot ----
    for j in range(J):
        r = j // ncols
        c = j % ncols
        ax = axes[r, c]
        highlight_set = highlight_map[j]

        for k in range(K):
            is_hi = (k in highlight_set)
            color = deep_pink if is_hi else line_colors[k]
            lw = 2.8 if is_hi else 1.4
            z = 3 if is_hi else 1
            alpha = 1.0 if is_hi else 0.95

            ax.plot(
                x,
                probs_win[:, j, k],
                color=color,
                linewidth=lw,
                alpha=alpha,
                label=f"k={k}",
                zorder=z,
            )

        ax.set_title(f"entity {j}")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(x[0], x[-1])

    # Hide unused
    for jj in range(J, nrows * ncols):
        r = jj // ncols
        c = jj % ncols
        axes[r, c].axis("off")

    # Legend: show all k=0..K-1, tint highlighted ks deep pink if highlighted anywhere
    if show_legend:
        highlighted_anywhere = set().union(*highlight_map.values()) if highlight_ks_by_entity is not None else set()
        handles = []
        for k in range(K):
            col = deep_pink if (k in highlighted_anywhere) else line_colors[k]
            lw = 3.0 if (k in highlighted_anywhere) else 2.0
            handles.append(mlines.Line2D([], [], color=col, linewidth=lw, label=f"k={k}"))
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    out_path = os.path.join(plot_dir, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig, axes
