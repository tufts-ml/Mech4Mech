
import os
from enum import Enum
from utilities.model import Model
from recurrence import cluster_trigger_system_recurrence_transformation, direction_entity_recurrence_transformation,identity_recurrence_system, identity_recurrence_entity
from gaussian.transition_and_emission_models import (
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_system_transition_probability_matrices_JAX,
)


marching_model_JAX = Model(
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_system_transition_probability_matrices_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    identity_recurrence_entity,
    cluster_trigger_system_recurrence_transformation
)

def save_model_type(model_dir: str, basename_prefix: str = ""):
    filepath = os.path.join(model_dir, f"{basename_prefix}_model_type_string.txt")
    with open(filepath, "w") as file:
        file.write("marching_band")