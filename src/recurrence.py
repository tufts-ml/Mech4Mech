import jax.numpy as jnp
import numpy as np 
import jax
import torch
import types
from pathlib import Path
from typing import Union
from feedback_mechanism.utils import flatten_params, inv_softplus, use_posterior, add_variational_layers

from utilities.types import JaxNumpyArray1D, NumpyArray3D, NumpyArray2D

"""
Defines the recurrence or feedback functions f(x) and g(x) for entity and system states from the observations, respectively. 
"""

def identity_recurrence_entity(
    observations: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    """

    Purpose: Inside recurrence function (JAX world). Uses the observations as the feedback component to interact with parameters (Psis) to compute the probability distribution for 
    that particular entity latent state at the next time step. 

    Arguments:
        x_prevs_reshaped: Has shape (JD,) where we scroll through j's first, and then d's.

    Returns: the identity of the values - i.e. just returns observations. 
    """

    return observations


def identity_recurrence_system(
    observations: JaxNumpyArray1D,
) -> JaxNumpyArray1D:
    """
    Purpose: Inside recurrence function (JAX world). Uses the observations as the feedback component to interact with parameters (Upsilon) to compute the probability distribution for 
    that particular system latent state at the next time step. 

    Arguments:
        observations: Has shape (JD,) where we scroll through j's first, and then d's.

    Returns: the identity of the values - i.e. just returns observations. 
    """
    return observations

def mechanisticfeedback_recurrence_transformation(
    observations: NumpyArray3D,
    latent_variable: str 
) -> Union[NumpyArray2D, NumpyArray3D]:

    """
    Purpose: Outside recurrence function from the JAX world. Uses the scalar class values predicted from a PyTorch model as the feedback 
        component to interact with the system and entity recurrence parameters. 

        All training scripts for the PyTorch model can be found in -> (feedback_mechanism). The trained model is saved in ->
        (feedback_mechanism -> best_model.pt) 

    Arguments:
        observations: has shape (T, J, D). 
        latent_variable: string that specifies the "system"or the "entity" recurrence.

    Returns: the scalar class value of each embedding in TxJxD (observations), predicted by a previously trained NN torch model (in feedback mechanism). 
        Input is a 3D numpy array of TxJxD. If the latent_variable = "system": the output is a (T-1)xJ 2D numpy array. If the latent_variable 
        = "entity": the output is a (T-1)xJx1 3D numpy array. 
    """

    device = torch.device("cpu")

    model = torch.nn.Sequential(
    torch.nn.Linear(in_features=128, out_features=128),
    torch.nn.ReLU(inplace=False),
    torch.nn.Linear(in_features=128, out_features=8),
    ).to(device)


    model.raw_sigma = torch.nn.Parameter(inv_softplus(torch.tensor(1e-4, device=device))) 
    add_variational_layers(model, model.raw_sigma) # Make all regular layers variational - have parameters that can be obtained from the approximate posterior.

    model.use_posterior = types.MethodType(use_posterior, model) 

    repo_root = Path(__file__).resolve().parent
    model_dir = repo_root / "feedback_mechanism" / "best_model.pt"   

    state_dict = torch.load(model_dir, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()   

    observations = observations[:-1, ...] 

    T, J, D = observations.shape
    x_torch = torch.from_numpy(observations).float()
    x_flat = x_torch.reshape(T * J, D)
    with torch.no_grad():
        y_pred = model(x_flat)
    pred_class = y_pred.argmax(dim=-1)

    y_TJ = pred_class.reshape(T, J)

    if latent_variable == "entity": 
        y_TJ_vector = y_TJ.unsqueeze(-1)
        y_TJ_np = y_TJ_vector.cpu().numpy()

    elif latent_variable == "system": 
        y_TJ_np = y_TJ.cpu().numpy()
        
    return y_TJ_np