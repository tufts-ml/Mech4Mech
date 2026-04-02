import numpy as np
import os
from pathlib import Path
import time

from utilities.util import ( prepare_run_directories, ensure_dir )

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


"""
Main script to test the HSRDM. 
"""

###
# DATA SPLITTING & PRE-PROCESSING
###

dataset_no = 1
model_adjustments = ["perf_ev", "class_ev", "class_ev_no_sys", "no_ev", "kmeans_init_full_ev", "kmeans_init_noisy_ev"]

###
# DATA LOADING
###

repo_root = Path(__file__).resolve().parents[1]

if dataset_no == 1:
    data_dir = repo_root / "data" / "unsupervised_inference" / "test_seen_problem"
    data_filename = data_dir / f"test_dataset_seen_problem.npz"
else:
    data_dir = repo_root / "data" / "unsupervised_inference" / "test_new_problem"
    data_filename = data_dir / f"test_dataset_new_problem.npz"

data = np.load(data_filename, allow_pickle=True)
all_X = data["X"].tolist()     
all_Y = data["Y"].tolist()
DATA = np.concatenate(all_X, axis=0)
example_end_times = data["example_end_times"].tolist()
evidence_strengths = np.concatenate(all_Y , axis=0)

seed = 17

params_loc_dict = {
    "perf_ev": f"seed_{seed}_system_size_2_n_iterations_15_adjustment_None_full_evidence",
    "class_ev": f"seed_{seed}_system_size_2_n_iterations_15_adjustment_None_noisy_evidence",
    "class_ev_no_sys": f"seed_{seed}_system_size_1_n_iterations_15_adjustment_one_system_regime_noisy_evidence",
    "no_ev": f"seed_{seed}_system_size_2_n_iterations_15_adjustment_remove_recurrence_no_evidence",
    "kmeans_init_full_ev": f"seed_{seed}_system_size_2_n_iterations_15_adjustment_kmeans_full_evidence",
    "kmeans_init_noisy_ev": f"seed_{seed}_system_size_2_n_iterations_15_adjustment_kmeans_noisy_evidence",
}



###
# SPECIFY MODEL
###


# Structure
J = 2 #Max number of students
K = 4 #The zero and one states are the silent observation states; The other 2 are for no evidence of mechanistic reasoning, evidence of mechanistic reasoning in the specific dialogue 
L = 2 #Number of system states

num_em_iterations_for_bottom_half_init = 1
num_em_iterations_for_top_half_init = 1


for model_adjustment in model_adjustments:
        
    model = Model(
        compute_log_initial_continuous_state_emissions_JAX,
        compute_log_continuous_state_emissions_after_initial_timestep_JAX,
        compute_log_system_transition_probability_matrices_JAX,
        compute_log_entity_transition_probability_matrices_JAX,
        internal_entity_recurrence_JAX=None,
        internal_system_recurrence_JAX= None,
    )
    
    
    params_dir = repo_root / "results" / "unsupervised_inference" / f"{params_loc_dict[model_adjustment]}" / "artifacts" / "_params.pkl"
    params = load_params(params_dir)
    
    ###
    # MODEL ADJUSTMENTS
    ###
    # Remove system and/or Internal recurrence 

    perfect_evidence = model_adjustment in ["perf_ev", "kmeans_init_full_ev"]

    outside_system_recurrence = mechanisticfeedback_recurrence_transformation(DATA, "system", data_filename, perfect_evidence)
    outside_entity_recurrence = mechanisticfeedback_recurrence_transformation(DATA, "entity", data_filename, perfect_evidence)

    if model_adjustment == "class_ev_no_sys":
        L = 1
    if model_adjustment == "no_ev":
        model.internal_entity_recurrence_JAX = (
            lambda x_vec: np.zeros(DIMS.D_e)  
        )
        outside_system_recurrence = None
        outside_entity_recurrence = None

    # Create directories

    # Repo root
    repo_root = Path(__file__).resolve().parents[1]

    # Put everything inside results/unsupervised_inference
    base_dir = repo_root / "results" / "unsupervised_inference_test"

    run_dir = base_dir / f"dataset_{'seen' if dataset_no == 1 else 'new'}_problem" / f"adjustment_{model_adjustment}"
    artifacts_dir = run_dir / "artifacts"
    ensure_dir(artifacts_dir)

    #### Setup Dims
    D = 128 
    D_e = 8
    D_s = 8
    DIMS = Dims(J, K, L, D, D_e, D_s)

    # Masking
    mask_observations = None  

    results_init = initialize_HSRDM(
        DIMS,
        DATA,
        example_end_times, 
        model,
        num_em_iterations_for_bottom_half_init,
        num_em_iterations_for_top_half_init,
        166,
        mask_observations,
        save_dir=None,
        outside_system_recurrence = outside_system_recurrence,
        outside_entity_recurrence = outside_entity_recurrence,
        params_frozen=params
    )
    params_init = results_init.params
    VES_summary, VEZ_summaries = results_init.ES_summary, results_init.EZ_summaries

    ####
    # MODEL VALIDATION 
    ####

    #Compute the correlations we care about 
    posterior_probabilities = VEZ_summaries.expected_regimes
    system_posterior_probabilities = VES_summary.expected_regimes

    artifacts_dir = f"{run_dir}/artifacts/"

    compute_entity_correlations = get_entity_correlation_table(posterior_probabilities, evidence_strengths, spearman = False, out_csv = artifacts_dir)

    #Compute the mean posterior probabilities we care about 
    get_transition_posteriors_conditioned_on_speaker_evidence(posterior_probabilities, evidence_strengths, DATA, std_dev=True, out_csv = artifacts_dir )
    get_k1_posterior_for_speaker_transition_to_silence_by_mech_evidence(posterior_probabilities, evidence_strengths, DATA, std_dev=True, out_csv = artifacts_dir )