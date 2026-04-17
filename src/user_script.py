import numpy as np
import os
from pathlib import Path
import time

from utilities.util import ( prepare_run_directories, ensure_dir, save_maxprob_tables, plot_top_evidence_windows, plot_highest_evidence_window)

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
User script to run the adapted-HSRDM on new data. 
"""

"""
Users required to change
"""
J = 4 #PLEASE CHANGE TO THE TOTAL NUMBER OF STUDENTS IN THE TRANSCRIPT.
return_max_probs = True #SET TO TRUE RETURNS A CSV FILE OF MAX PROBABILITIES AND ASSOCIATED STATES OVER TIME. FALSE DOES NOT RETURN.
return_segmentation_plots = True #SET TO TRUE RETURNS SEGMENTATION PLOTS OF PROBABILITIES OVER TIME FOR THE HIGHEST REGIONS OF MECHANISTIC REASONING FOR A FIXED TIMESTEP DURATION. FALSE DOES NOT RETURN.
num_timesteps = 130 #SET TO THE NUMBER OF TIMESTEPS FOR TAKING THE MOVING AVERAGE 



"""
Users to leave unchanged
"""

###
# DATA LOADING
###

repo_root = Path(__file__).resolve().parents[1]
data_dir = repo_root / "data" / "new_user" 
data_filename = data_dir / f"user_data.npz"

data = np.load(data_filename, allow_pickle=True)
all_X = data["observations"].tolist()     
DATA = data["observations"]
example_end_times = [-1, len(all_X)] 
num_regions = 1
num_timesteps = len(all_X) #SET TO THE NUMBER OF TIMESTEPS FOR TAKING THE MOVING AVERAGE 

###
# SPECIFY MODEL
###


# Structure
K = 4 #The zero and one states are the silent observation states; The other 2 are for no evidence of mechanistic reasoning, evidence of mechanistic reasoning in the specific dialogue 
L = 2 #Number of system states

num_em_iterations_for_bottom_half_init = 1
num_em_iterations_for_top_half_init = 1

model = Model(
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_system_transition_probability_matrices_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    internal_entity_recurrence_JAX=None,
    internal_system_recurrence_JAX= None,
    )
    
    
params_dir = repo_root / "results" / "unsupervised_inference" / f"seed_17_system_size_2_n_iterations_15_classifier" / "artifacts" / "_params.pkl"
params = load_params(params_dir)
    

outside_system_recurrence = mechanisticfeedback_recurrence_transformation(DATA, "system", data_filename, False)
outside_entity_recurrence = mechanisticfeedback_recurrence_transformation(DATA, "entity", data_filename, False)


# Create directories

# Repo root
repo_root = Path(__file__).resolve().parents[1]

# Put everything inside results/unsupervised_inference
base_dir = repo_root / "results" / "new_user"
ensure_dir(base_dir)

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
    # MODEL OUTPUTS
    ####

#Get the posterior probabilities
posterior_probabilities = VEZ_summaries.expected_regimes
system_posterior_probabilities = VES_summary.expected_regimes



if return_max_probs == True: 
#Obtain a CSV of the max posterior probability and the associated state for each entity at each time point
    save_maxprob_tables(save_dir=base_dir, posterior_probabilities=posterior_probabilities)
else: 
    None

if return_segmentation_plots == True: 
    plot_highest_evidence_window(system_posterior_probabilities, num_timesteps, base_dir)
    
