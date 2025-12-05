from enum import Enum
import os
from dataclasses import dataclass
from typing import Callable, Optional

"""
Defines the model structure - i.e. the factorized joint probability distribution,
 which requires assuming transition and emission probability matrices as well as the optional internal recurrence functions f(x) and g(x). 
 Internal recurrence functions are those that work within the JAX tracer landscape -i.e. transformations to the observations 
 that can be performed on JAX tracer objects. External recurrence functions are defined outside of the model class. 
"""

@dataclass
class Model:
    """
    Purpose: Gives the ingredients necessary to define the joint probability distribution 
        of the hierarchical switching recurrent dynamical model.

    Attributes: 
        compute_log_initial_continuous_state_emissions_JAX: Computes the initial log emission distributions -> (compute_emissions.py)
        compute_log_continuous_state_emissions_after_initial_timestep_JAX: Computes the log emissions for all T after t=1 -> (compute_emissions.py)
        compute_log_system_transition_probability_matrices_JAX: Computes the log system transitions -> (compute_transitions.py)
        compute_log_entity_transition_probability_matrices_JAX: Computes the log entity transitions -> (compute_transitions.py)
        internal_entity_recurrence_JAX: Computes the recurrence coefficients from transforming the observations internally -> (recurrence.py) and (compute_transitions.py)
        internal_system_recurrence_JAX: Computes the recurrence coefficients from transforming the observations internally -> (recurrence.py) and (compute_transitions.py)
    """
    compute_log_initial_continuous_state_emissions_JAX: Callable
    compute_log_continuous_state_emissions_after_initial_timestep_JAX: Callable
    compute_log_system_transition_probability_matrices_JAX: Callable
    compute_log_entity_transition_probability_matrices_JAX: Callable
    internal_entity_recurrence_JAX: Optional[
        Callable
    ] = None
    internal_system_recurrence_JAX: Optional[
        Callable
    ] = None


def save_model_type(model_dir: str, basename_prefix: str = ""):
    filepath = os.path.join(model_dir, f"{basename_prefix}_model_type_string.txt")
    with open(filepath, "w") as file:
        file.write("student_reasoning")
