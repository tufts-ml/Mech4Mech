import numpy as np
import jax.numpy as jnp
from pathlib import Path
import numpy.random as npr
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import time 
import os
from sklearn.cluster import KMeans

from utilities.util import (
    prepare_run_directories, ensure_dir, save_state_frequency_counts, save_maxprob_tables, save_posteriors_as_strings, save_embedding_similarity_metrics, save_transition_type_counts_per_entity
)


from model import Model, save_model_type
from recurrence import mechanisticfeedback_recurrence_transformation
from initialize import (
    initialize_HSRDM,
)
from params import (
    Dims,
    get_dim_of_internal_entity_recurrence,
    get_dim_of_internal_system_recurrence,
    save_params,
)
from compute_transitions import (
    compute_log_entity_transition_probability_matrices_JAX,
    compute_log_system_transition_probability_matrices_JAX,
)
from compute_emissions import(compute_log_continuous_state_emissions_after_initial_timestep_JAX, compute_log_initial_continuous_state_emissions_JAX,
)
from maximization_step import M_step_toggles_from_strings
from compute_posterior import save_hmm_posterior_summary
from cavi_training import SystemTransitionPrior_JAX, run_CAVI_with_JAX
from metrics import plot_elbo, get_entity_correlation_table, get_entity_correlation_metric, get_transition_posteriors_conditioned_on_speaker_evidence, get_system_correlation_metric, plot_speaking_and_evidence_boxes, plot_posteriors_per_entity


"""
Main script to train the HSRDM. 
"""

###
# DATA SPLITTING & PRE-PROCESSING
###
repo_root = Path(__file__).resolve().parents[1]
data_dir = repo_root / "data" / "unsupervised_inference" / "training"
data = np.load(data_dir / "training_dataset.npz", allow_pickle=True)
all_X = data["X"].tolist()     
all_Y = data["Y"].tolist()
DATA = np.concatenate(all_X, axis=0)
example_end_times = data["example_end_times"].tolist()
evidence_strengths = np.concatenate(all_Y , axis=0)


###
# SPECIFY MODEL
###

# Structure
n_train_sequences = 8 #Number of training segments 
J = 4 #Max number of students
K = 4 #The zero and one states are the silent observation states; The other 2 are for no evidence of mechanistic reasoning, evidence of mechanistic reasoning in the specific dialogue 
L = 2 #Tunable


model = Model(
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_system_transition_probability_matrices_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    internal_entity_recurrence_JAX=None,
    internal_system_recurrence_JAX= None,
)
model_adjustment = "None"

# Initialization
seed_for_initialization = 126
num_em_iterations_for_bottom_half_init = 1
num_em_iterations_for_top_half_init = 1


# Inference
n_cavi_iterations = 15
M_step_toggle_for_STP = "gradient_descent"  
M_step_toggle_for_ETP = "gradient_descent"
M_step_toggle_for_continuous_state_parameters = "closed_form_gaussian"
M_step_toggle_for_IP = "closed_form_gaussian"
num_M_step_iters = 50
alpha_system_prior, kappa_system_prior = 1, 0
show_system_states = False 

# Create directories
run_description = f"seed_{seed_for_initialization}_system_size_{L}_n_iterations_{n_cavi_iterations}_adjustment_{model_adjustment}_new_metrics3"
prepare_run_directories(run_description)

repo_root = Path(__file__).resolve().parents[1]
base_dir = repo_root / "results" / "unsupervised_inference" / f"{run_description}"
plots_dir = f"{base_dir}/plots"
artifacts_dir = f"{base_dir}/artifacts/"

ensure_dir(plots_dir)
ensure_dir(artifacts_dir)

# Make prior
system_transition_prior = SystemTransitionPrior_JAX(alpha_system_prior, kappa_system_prior)

#### Setup Dims
D = 128 
D_e = 8
D_s = 8
DIMS = Dims(J, K, L, D, D_e, D_s)

###
# MODEL ADJUSTMENTS
###
# Remove system and/or Internal recurrence 
if model_adjustment == "one_system_regime":
    DIMS.L = 1
elif model_adjustment == "remove_recurrence":
    model.internal_entity_recurrence_JAX = (
        lambda x_vec: np.zeros(DIMS.D_e) 
    )
elif model_adjustment == "no_recurrence_and_system":
    DIMS.L = 1
    model.internal_entity_recurrence_JAX = (
        lambda x_vec: np.zeros(DIMS.D_e)  
    )

# External recurrence 
outside_system_recurrence = mechanisticfeedback_recurrence_transformation(DATA, "system",True)
# mechanisticfeedback_recurrence_transformation(DATA, "system")
outside_entity_recurrence = mechanisticfeedback_recurrence_transformation(DATA, "entity", True)

#  mechanisticfeedback_recurrence_transformation(DATA, "entity")

# Masking
mask_observations = None  

###
# INITIALIZATION
###
start_time = time.time() 
print("Running smart initialization.")


results_init = initialize_HSRDM(
    DIMS,
    DATA,
    example_end_times, 
    model,
    num_em_iterations_for_bottom_half_init,
    num_em_iterations_for_top_half_init,
    seed_for_initialization,
    mask_observations,
    save_dir=artifacts_dir,
    outside_system_recurrence = outside_system_recurrence,
    outside_entity_recurrence= outside_entity_recurrence,
)
params_init = results_init.params
VES_init, VEZ_init = results_init.ES_summary, results_init.EZ_summaries


####
# INFERENCE
####

VES_summary, VEZ_summaries, params_learned, elbo_decomposed = run_CAVI_with_JAX(
    params_init,
    VES_init, VEZ_init,
    system_transition_prior,
    model,
    jnp.asarray(DATA),
    example_end_times,
    n_cavi_iterations,
    M_step_toggles_from_strings(
        M_step_toggle_for_STP,
        M_step_toggle_for_ETP,
        M_step_toggle_for_continuous_state_parameters,
        M_step_toggle_for_IP,
    ),
    num_M_step_iters,
    outside_system_recurrence,
    outside_entity_recurrence,
    mask_observations,
    evid_onehot = evidence_strengths, 
    save_dir=artifacts_dir,
)


end_time = time.time() 
elapsed_time = end_time - start_time
print(f"Code execution time: {elapsed_time} seconds")

### Save model, learned params, latent state distribution
save_model_type(artifacts_dir, basename_prefix=run_description)
save_params(params_learned, artifacts_dir)
save_hmm_posterior_summary(VES_summary, "qS", artifacts_dir)
save_hmm_posterior_summary(VEZ_summaries, "qZ", artifacts_dir)


####
# MODEL VALIDATION 
####

#Get stats about the data
save_embedding_similarity_metrics(X = DATA, Y = evidence_strengths, save_dir = artifacts_dir)
save_transition_type_counts_per_entity(observations = DATA, one_hot_evidence = evidence_strengths, save_dir = artifacts_dir) 

#Plot the ELBO over time 
elbo_history = [d["elbo"] for d in elbo_decomposed]
plot_elbo(elbo_history, plots_dir, example_end_times, J)

#Compute the correlations we care about 
posterior_probabilities = VEZ_summaries.expected_regimes
system_posterior_probabilities = VES_summary.expected_regimes

compute_entity_correlations = get_entity_correlation_table(posterior_probabilities, evidence_strengths, out_csv = artifacts_dir )

#Compute the mean posterior probabilities we care about 
get_transition_posteriors_conditioned_on_speaker_evidence(posterior_probabilities, evidence_strengths, DATA, out_csv = artifacts_dir )


#Get the post-training diagnostics
save_state_frequency_counts(save_dir = artifacts_dir, iteration = None, posterior_probabilities = posterior_probabilities , system_posterior_probabilities = system_posterior_probabilities)
save_maxprob_tables(save_dir = artifacts_dir, iteration = None, posterior_probabilities = posterior_probabilities , system_posterior_probabilities = system_posterior_probabilities, one_hot_evidence = evidence_strengths)
save_posteriors_as_strings(save_dir = artifacts_dir, iteration = None, posterior_probabilities = posterior_probabilities , system_posterior_probabilities = system_posterior_probabilities)


#Plot the trajectories of both the evidence strengths and the posterior probabilities 
plot_speaking_and_evidence_boxes(evidence_strengths, DATA, colors = None, plot_dir=plots_dir) 
plot_posteriors_per_entity(posterior_probabilities, colors = None, plot_dir=plots_dir)
# Example end times [-1, 484, 721, 1297, 1625, 2233, 2624, 3417, 3691]
plot_posteriors_per_entity(posterior_probabilities, plot_dir=plots_dir, t_start =1550, t_end = 1600,keep_original_time = False, filename = "entity_posteriors_short.pdf")


