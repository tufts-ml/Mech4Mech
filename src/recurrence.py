import jax.numpy as jnp
import numpy as np 
import jax
import torch
import types
from pathlib import Path
from typing import Union
from feedback_mechanism.utils import flatten_params, inv_softplus, use_posterior, add_variational_layers
from utilities.types import JaxNumpyArray1D, NumpyArray3D, NumpyArray2D
import torch.nn.functional as F

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
    latent_variable: str, 
    data_filename: str,
    perf_evidence: bool
) -> Union[NumpyArray2D, NumpyArray3D]:

    """
    Purpose: Outside recurrence function from the JAX world. Uses the scalar class values predicted from a PyTorch model as the feedback 
        component to interact with the system and entity recurrence parameters. 

        All training scripts for the PyTorch model can be found in -> (feedback_mechanism). The trained model is saved in ->
        (feedback_mechanism -> best_model.pt) 

    Arguments:
        observations: has shape (T, J, D). 
        latent_variable: string that specifies the "system"or the "entity" recurrence.
        data_filename: the filename for the data, which is needed to load the evidence strengths if perf_evidence = True.
        perf_evidence: if True the evidence is from the perfect annotations. If False, the evidence is from the classifier 

    Returns: the scalar class value of each embedding in TxJxD (observations), predicted by a previously trained NN torch model (in feedback mechanism). 
        Input is a 3D numpy array of TxJxD. If the latent_variable = "system": the output is a (T-1)xJ 2D numpy array. If the latent_variable 
        = "entity": the output is a (T-1)xJx1 3D numpy array. 
    """

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "unsupervised_inference" 
    z = np.load(data_dir / "silence_embedding.npz")
    silence = z["silence"]

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
        logits = model(x_flat)
    probs = torch.softmax(logits, dim=1)
    max_probs, preds = torch.max(probs, dim=1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
    pred_prob = probs.argmax(dim=-1)

    y_TJ_probs = probs.reshape(T, J, 8)

    preds_onehot_flat = F.one_hot(preds, num_classes=8).float()
    y_TJ = preds_onehot_flat.view(T, J, 8)

    if perf_evidence == False: 

        if latent_variable == "entity": 
            y_TJ_np = y_TJ.cpu().numpy()

        elif latent_variable == "system": 
            v = silence
            X8 = torch.as_tensor(y_TJ)
            XD = torch.as_tensor(observations)
            v  = torch.as_tensor(v)

            mask = (XD != v.view(1, 1, -1)).any(dim=-1)
            j_idx = mask.long().argmax(dim=1)
            X_T8 = X8[torch.arange(X8.shape[0]), j_idx]
            y_TJ_np = X_T8.cpu().numpy()

    if perf_evidence == True: 
        data = np.load(data_filename, allow_pickle=True)   
        all_Y = data["Y"].tolist()
        evidence_strengths = np.concatenate(all_Y , axis=0)

        if latent_variable == "entity": 
            X8 = torch.as_tensor(evidence_strengths[:-1])
            y_TJ_np = X8.cpu().numpy()

        elif latent_variable == "system": 
            v = silence
            X8 = torch.as_tensor(evidence_strengths[:-1])
            XD = torch.as_tensor(observations)
            v  = torch.as_tensor(v)

            mask = (XD != v.view(1, 1, -1)).any(dim=-1)
            j_idx = mask.long().argmax(dim=1)
            X_T8 = X8[torch.arange(X8.shape[0]), j_idx]
            y_TJ_np = X_T8.cpu().numpy()
 
    return y_TJ_np