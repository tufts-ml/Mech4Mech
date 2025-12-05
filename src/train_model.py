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
    prepare_run_directories, ensure_dir
)

from data_generation import get_training_data, get_test_data
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
from metrics import compute_monotonicity_correlation, windowed_spearman_no_evidence_prob



"""
Main script to train the HSRDM. 
"""

###
# DATA SPLITTING & PRE-PROCESSING
###
gen = get_training_data()
DATA = np.concatenate(gen[0], axis=0)
example_end_times = gen[3]


###
# SPECIFY MODEL
###

# Structure
n_train_sequences = 8 #Number of training segments 
J = 4 #Max number of students
K = 2 #Set to 2 for no evidence of mechanistic reasoning, evidence of mechanistic reasoning
L = 3 #Tunable


model = Model(
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_system_transition_probability_matrices_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    internal_entity_recurrence_JAX= None,
    internal_system_recurrence_JAX= None,
)
model_adjustment = "None"

# Initialization
seed_for_initialization = 126
num_em_iterations_for_bottom_half_init = 1
num_em_iterations_for_top_half_init = 1


# Inference
n_cavi_iterations = 10
M_step_toggle_for_STP = "gradient_descent"  
M_step_toggle_for_ETP = "gradient_descent"
M_step_toggle_for_continuous_state_parameters = "closed_form_gaussian"
M_step_toggle_for_IP = "closed_form_gaussian"
num_M_step_iters = 50
alpha_system_prior, kappa_system_prior = 1.0, 10.0 
show_system_states = False 

# Create directories
run_description = f"seed_{seed_for_initialization}_system_size_{L}_n_iterations_{n_cavi_iterations}_adjustment_{model_adjustment}"
prepare_run_directories(run_description)

repo_root = Path(__file__).resolve().parents[1]
base_dir = repo_root / "results" / "unsupervised_inference" / f"{run_description}"
plots_dir = f"{base_dir}/plots"
artifacts_dir = f"{base_dir}/artifacts"

ensure_dir(plots_dir)
ensure_dir(artifacts_dir)

# Make prior
system_transition_prior = SystemTransitionPrior_JAX(alpha_system_prior, kappa_system_prior)

#### Setup Dims
D = 128
D_e = 1
D_s = 4
DIMS = Dims(J, K, L, D, D_e, D_s)

###
# MODEL ADJUSTMENTS
###
# Remove system and/or Internal recurrence 
if model_adjustment == "one_system_regime":
    DIMS.L = 1
elif model_adjustment == "remove_recurrence":
    model.transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX = (
        lambda x_vec: np.zeros(DIMS.D_e)  
    )
elif model_adjustment == "no_recurrence_and_system":
    DIMS.L = 1
    model.transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX = (
        lambda x_vec: np.zeros(DIMS.D_e)  
    )

# External recurrence 
outside_system_recurrence = mechanisticfeedback_recurrence_transformation(DATA, "system")
outside_entity_recurrence = mechanisticfeedback_recurrence_transformation(DATA, "entity")

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
    save_dir=plots_dir,
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

elbo_history = [d["elbo"] for d in elbo_decomposed]

def plot_elbo(elbo_values, plots_dir, filename="elbo_over_iterations.pdf"):
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

plot_elbo(elbo_history, plots_dir)

expected = VEZ_summaries.expected_regimes   # shape (T, J, 2)
zero_val = expected[..., 0] #Just takes index 0 
one_val = expected[..., 1] #Just takes index 1
evidence_strengths = np.argmax(np.concatenate(gen[1], axis=0), axis=2) + 1

print(expected)
print(zero_val)
print(one_val)
print(evidence_strengths)
# We compute for both placeholders (0,1) as we're not sure which one corresponds most to evidence of mechanistic reasoning
# We'll take the higher one of the two scores 

#Computing monotonicity 
rho, p_value = compute_monotonicity_correlation(zero_val, evidence_strengths)
print("zero rho: " + str(rho))
print("zero p-value: " + str(p_value))

rho, p_value = compute_monotonicity_correlation(one_val, evidence_strengths)
print("one rho: " + str(rho))
print("one p-value: " + str(p_value))

# We compute for both placeholders (0,1) as we're not sure which one corresponds most to evidence of mechanistic reasoning
# We'll take the higher one of the two scores 

#Computing contextual monotonicity: correlation between probabilities of the "no evidence" labels in a set of 20 
# and the mean evidence across all sets of 20 in an episode; discarding the <20 observations at the end. 

rho, p_value = windowed_spearman_no_evidence_prob(
    zero_val,
    evidence_strengths,
    evidence_strengths,
    example_end_times[1:],
    window_size=20,
    no_evidence_class=1,
)
print("zero cluster rho: " + str(rho))
print("zero cluster p-value: " + str(p_value))

rho, p_value = windowed_spearman_no_evidence_prob(
    one_val,
    evidence_strengths,
    evidence_strengths,
    example_end_times[1:],
    window_size=20,
    no_evidence_class=1,
)
print("one cluster rho: " + str(rho))
print("one cluster p-value: " + str(p_value))

def plot_vals_scatter(values, plots_dir, filename="vals_scatter.pdf"):
    """
    Purpose: Scatter plot the vals (T x J_max), where each J_max is plotted
        in a different color over time steps.

    Arguments: 
        values : np.ndarray
            Array of shape (T, J_max) containing max values per student per time.
        plots_dir : Path or str
            Directory where to save the plot.
        filename : str
            Output file name.
    """

    values = np.asarray(values)
    T, J_max = values.shape

    plt.figure(figsize=(10, 6))

    for j in range(J_max):
        plt.scatter(
            np.arange(T),
            values[:, j],
            s=20,
            alpha=0.8,
            label=f"Student {j}",
        )

    plt.xlabel("Time step (t)")
    plt.ylabel("Probability")
    plt.title("Mechanistic Reasoning Probability per Student Over Time")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()

    save_path = Path(plots_dir) / filename
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

plot_vals_scatter(zero_val, plots_dir, filename="zero_vals_scatter.pdf")
plot_vals_scatter(one_val, plots_dir, filename="one_vals_scatter.pdf")


            







