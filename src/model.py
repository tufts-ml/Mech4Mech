from enum import Enum
import os
from dataclasses import dataclass
from typing import Callable, Optional

"""
Defines the model structure, which takes in transition and emission probability matrices as well as the recurrence functions f(x) and g(x). 
"""

@dataclass
class Model:
    """
    Gives the ingredients necessary to define a hierarchical switching recurrent dynamical model,
    as defined in the TMLR publication.

    """
    compute_log_initial_continuous_state_emissions_JAX: Callable
    compute_log_continuous_state_emissions_after_initial_timestep_JAX: Callable
    compute_log_system_transition_probability_matrices_JAX: Callable
    compute_log_entity_transition_probability_matrices_JAX: Callable
    transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX: Callable
    transform_of_flattened_continuous_state_vectors_before_premultiplying_by_system_recurrence_matrix_JAX: Optional[
        Callable
    ] = None


def save_model_type(model_dir: str, basename_prefix: str = ""):
    filepath = os.path.join(model_dir, f"{basename_prefix}_model_type_string.txt")
    with open(filepath, "w") as file:
        file.write("marching_band")
