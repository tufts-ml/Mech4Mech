from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from pathlib import Path
import time
import pandas as pd
from utilities.util import ( _to_listed_pink_gradient, evidence_strength_from_onehot, speaker_index_per_t )
from params import load_params
from model import Model, save_model_type
from recurrence import mechanisticfeedback_recurrence_transformation
from initialize import ( initialize_HSRDM, )
from params import ( Dims, )
from compute_transitions import (
    compute_log_entity_transition_probability_matrices_JAX,
    compute_log_system_transition_probability_matrices_JAX,
)
from compute_emissions import( compute_log_continuous_state_emissions_after_initial_timestep_JAX, compute_log_initial_continuous_state_emissions_JAX, )
from metrics import get_entity_correlation_table, get_transition_posteriors_conditioned_on_speaker_evidence, plot_speaking_and_evidence_boxes, plot_posteriors_per_entity, get_k1_posterior_for_speaker_transition_to_silence_by_mech_evidence
from utilities.types import JaxNumpyArray3D

repo_root = Path(__file__).resolve().parents[1]


# Generation code for Figure 3 in the paper

def get_evid_and_obs(dataset: str):
    """
    Purpose: Get the evidence strengths for a specified dataset

    Arguments:
        dataset (str): Dataset to get evidence strengths for. Should be either "seen" or "new".

    Returns:
        np.ndarray: Evidence strengths for the specified dataset, shape (T, J, C)
        np.ndarray: Observations for the specified dataset, shape (T, J, D)
        list: List of example end times for the specified dataset
        str: Path to the data file for the specified dataset
    """
    repo_root = Path(__file__).resolve().parents[1]
    
    data_filename = repo_root / "data" / "unsupervised_inference" / f"test_{dataset}_problem" / f"test_dataset_{dataset}_problem.npz"
    
    data = np.load(data_filename, allow_pickle=True)
    all_X = data["X"].tolist()     
    all_Y = data["Y"].tolist()
    DATA = np.concatenate(all_X, axis=0)
    example_end_times = data["example_end_times"].tolist()
    evidence_strengths = np.concatenate(all_Y , axis=0)
    
    return evidence_strengths, DATA, example_end_times, data_filename

def get_annotations(evid_onehot: np.ndarray, observations: np.ndarray, timesteps: int, entity: int):
    """
    Purpose:
        Return a contiguous block of length `timesteps` for the given entity
        such that the average evidence strength over that block is maximized.
    
    Arguments:
        evid_onehot (np.ndarray): One-hot encoded evidence strengths, shape (T, J, C)
        observations (np.ndarray): Observations, shape (T, J, D)
        timesteps (int): Number of contiguous timesteps to return
        entity (int): Entity index to return evidence strengths for (0 or 1)
    
    Returns:
        np.ndarray: Evidence strengths for the specified entity and timesteps, shape (timesteps,)
        int: Start index of the contiguous block
        int: End index of the contiguous block
    """
    
    T, J, D = observations.shape
    C = np.asarray(evid_onehot).shape[-1]

    evid_strength = evidence_strength_from_onehot(evid_onehot)  # (T,J), 0..C-1
    speaker_idx = speaker_index_per_t(observations)             # (T,)
    T = len(evid_strength)
    
    evid_strength2 = np.max(evid_strength, axis=1)

    values = np.zeros((J, T), dtype=int)
    values[speaker_idx, np.arange(T)] = evid_strength[np.arange(T), speaker_idx]
    
    evid_strength = values[entity, :]

    
    if timesteps > T:
        raise ValueError("timesteps window larger than total available timesteps")
    
    # Sliding window sum via convolution (efficient and clean)
    window_sums = np.convolve(evid_strength2, 
                              np.ones(timesteps, dtype=float), 
                              mode='valid')
    
    # Index where average (equivalently sum) is maximal
    start_idx = np.argmax(window_sums)
    end_idx = start_idx + timesteps
    
    return evid_strength[start_idx:end_idx], start_idx, end_idx

def get_posterior_plot(posterior_probabilities: np.ndarray, entity: int, start_idx: int, end_idx: int):
    """
    Purpose:
        Given the posterior probabilities from the model, extract the probabilities for the specified entity and timesteps.

    Args:
        posterior_probabilities (np.ndarray): Posterior probabilities from the model, shape (T, J, K)
        entity (int): Entity index to extract probabilities for (0 or 1)
        start_idx (int): Start index of the contiguous block of timesteps to extract
        end_idx (int): End index of the contiguous block of timesteps to extract

    Returns:
        np.ndarray: Probabilities of k=0 / k=2 and k=1 / k=3 for the specified entity and timesteps, each of shape (timesteps,)
    """
    
    probs = posterior_probabilities[start_idx:end_idx, entity, :]  # (timesteps, K)
    
    p0 = probs[:, 0] + probs[:, 2]  # No evidence of mechanistic reasoning
    p1 = probs[:, 1] + probs[:, 3]  # Evidence of mechanistic reasoning
    
    return p0, p1
    

posterior_probabilities: JaxNumpyArray3D = None
evid_onehot: np.ndarray = None   # (T,J,C)
observations: np.ndarray = None  # (T,J,D)

evid_onehot, observations, example_end_times, data_filename = get_evid_and_obs("new")

param_dir = repo_root / "results" / "unsupervised_inference" / "seed_17_system_size_2_n_iterations_15_adjustment_None_noisy_evidence" / "artifacts"
param_loc = f"{param_dir}/_params.pkl"

params = load_params(param_loc)

model = Model(
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_system_transition_probability_matrices_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    internal_entity_recurrence_JAX=None,
    internal_system_recurrence_JAX= None,
)

perfect_evidence = False

outside_system_recurrence = mechanisticfeedback_recurrence_transformation(observations, "system", data_filename, perfect_evidence)
outside_entity_recurrence = mechanisticfeedback_recurrence_transformation(observations, "entity", data_filename, perfect_evidence)

J = 2
K = 4
L = 2

D = 128 
D_e = 8
D_s = 8
DIMS = Dims(J, K, L, D, D_e, D_s)

results_init = initialize_HSRDM(
    DIMS,
    observations,
    example_end_times, 
    model,
    1,
    1,
    166,
    None,
    save_dir=None,
    outside_system_recurrence = outside_system_recurrence,
    outside_entity_recurrence = outside_entity_recurrence,
    params_frozen=params
)

posterior_probabilities = results_init.EZ_summaries.expected_regimes  # (T, J, K)

C = 8
include_classifier = True

bar_colors = ["#D95F02", "#2FAE7C"]
mpl.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "legend.title_fontsize": 16,
})


entity_data = {}
for entity in [0, 1]:
    evid_strength, start_idx, end_idx = get_annotations(
        evid_onehot, observations, timesteps=20, entity=entity
    )
    p0, p1 = get_posterior_plot(
        posterior_probabilities,
        entity=entity,
        start_idx=start_idx,
        end_idx=end_idx
    )
    x = np.arange(len(p0)) + start_idx

    classifier_output = evidence_strength_from_onehot(outside_entity_recurrence)
    speaker_idx = speaker_index_per_t(observations)[:-1]
    T_c = len(classifier_output)
    values_c = np.zeros((J, T_c), dtype=int)
    values_c[speaker_idx, np.arange(T_c)] = classifier_output[np.arange(T_c), speaker_idx]
    classifier_slice = values_c[entity, start_idx:end_idx]

    entity_data[entity] = dict(
        evid_strength=evid_strength,
        start_idx=start_idx,
        end_idx=end_idx,
        p0=p0,
        p1=p1,
        x=x,
        classifier_slice=classifier_slice,
    )


s0, e0 = entity_data[0]["start_idx"], entity_data[0]["end_idx"]
s1, e1 = entity_data[1]["start_idx"], entity_data[1]["end_idx"]
global_start = min(s0, s1)
global_end   = max(e0, e1)
T_full = global_end - global_start

def place_slice(arr, s, e, T_full, global_start):
    out = np.zeros(T_full, dtype=float)
    out[s - global_start: e - global_start] = arr
    return out


n_rows = 3 if include_classifier else 2
fig = plt.figure(figsize=(8, 7))

gs = fig.add_gridspec(
    n_rows, 2,
    width_ratios=[1, 0.05],
    height_ratios=[1] * n_rows,
    hspace=0.15,
    wspace=0.05
)

def add_gridlines(ax: plt.Axes):
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.5, color='gray')

def bar_height(val):
    val = np.asarray(val, dtype=float)
    return np.where(val == 0, 0.0, 0.5 + (val - 1) * (0.5 / 6))
    
def draw_bar_plot(ax, x_positions, slice1: np.ndarray, slice2: np.ndarray):
    values = np.maximum(slice1, slice2).astype(int)
    
    # color based on which student has higher value at each position
    color = np.where(slice1 > slice2, bar_colors[0], bar_colors[1])
    
    heights = bar_height(values)
    ax.bar(x_positions, heights, width=0.8, color=color, alpha=0.85, align="center")

# Row 0: Posterior line plot - one line for each student (prob. of moving to k=1/3 when evidence is present)
ax_line = fig.add_subplot(gs[0, 0])
ax_class = fig.add_subplot(gs[1, 0], sharex=ax_line)
heat_row = 2 if include_classifier else 1
ax_heat = fig.add_subplot(gs[heat_row, 0], sharex=ax_line)

ax_line.xaxis.set_major_locator(MultipleLocator(2))
ax_line.xaxis.set_minor_locator(MultipleLocator(1))

for entity in [0, 1]:
    d = entity_data[entity]
    ax_line.plot(d["x"], d["p1"], 'o-', color=bar_colors[entity],
                 label=f"Student {entity}")

plt.setp(ax_line.get_xticklabels(), visible=False)
ax_line.tick_params(bottom=False)
ax_line.set_ylabel("Posterior Prob.")
ax_line.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
ax_line.margins(y=0.08)
fig.add_subplot(gs[0, 1]).axis("off")

add_gridlines(ax_line)

# Row 1: Classifier output bar plot. One bar plot showing classifier output for each student.
if include_classifier:

    c0 = place_slice(entity_data[0]["classifier_slice"], s0, e0, T_full, global_start)
    c1 = place_slice(entity_data[1]["classifier_slice"], s1, e1, T_full, global_start)

    x_bars = np.arange(global_start, global_end)
    draw_bar_plot(ax_class, x_bars, c0, c1)

    ax_class.set_xlim(global_start - 0.5, global_end - 0.5)
    ax_class.set_ylim(0, 1.0)
    plt.setp(ax_class.get_xticklabels(), visible=False)
    ax_class.tick_params(bottom=False)
    ax_class.set_yticks([0, 0.5, 0.75, 1.0])
    ax_class.set_yticklabels(["0", "1", "4", "7"])
    ax_class.set_ylabel("Classifier\nFeedback")
    add_gridlines(ax_class)   # inside the if include_classifier block
    fig.add_subplot(gs[1, 1]).axis("off")
    

# Row 2: Human feedback output bar plot. One bar plot showing human feedback for each student.
e0_arr = place_slice(entity_data[0]["evid_strength"], s0, e0, T_full, global_start)
e1_arr = place_slice(entity_data[1]["evid_strength"], s1, e1, T_full, global_start)
combined_evid = np.maximum(e0_arr, e1_arr).astype(int)

x_bars = np.arange(global_start, global_end)
draw_bar_plot(ax_heat, x_bars, e0_arr, e1_arr)

ax_heat.set_xlim(global_start - 0.5, global_end - 0.5)
ax_heat.set_ylim(0, 1.0)
ax_heat.set_yticks([0, 0.5, 0.75, 1.0])
ax_heat.set_yticklabels(["0", "1", "4", "7"])
ax_heat.set_ylabel("Human\nFeedback")
ax_heat.set_xlabel("Timestep")
add_gridlines(ax_heat)
fig.add_subplot(gs[heat_row, 1]).axis("off")
for ax in [ax_line, ax_class, ax_heat]:
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
plt.savefig(repo_root / "results" / "unsupervised_inference_test" / "posterior_and_evidence_plot_new.pdf", bbox_inches="tight")

