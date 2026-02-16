import os
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy
import matplotlib.pyplot as plt 
from pathlib import Path
import torch
import torchvision
import torchmetrics
#Import the below when training the HSRDM. Comment out and just import layers when training the feedback mechanism 
import feedback_mechanism.layers as layers
#import layers

def inv_softplus(x):
    """
    Purpose: Implements softplus function on the input. 

    Arguments:
        x: scalar input

    Returns:
        Softplus transformation of the scalar input x. 
    """
    return x + torch.log(-torch.expm1(-x))
        
def add_variational_layers(module, raw_sigma):
    """
    Purpose: Incorporates posterior variance parameters for all MODEL parameters 

    Arguments:
        module: torch object
        raw_sigma: variance parameter of the assumed approximate posterior

    Returns:
        Adds assumed approximate posterior variances to the weights/parameters from the MODEL layers 
    """
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, layers.VariationalLinear(child, raw_sigma))
        elif isinstance(child, torch.nn.Conv2d):
            setattr(module, name, layers.VariationalConv2d(child, raw_sigma))
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(module, name, layers.VariationalBatchNorm2d(child, raw_sigma))
        elif isinstance(child, torchvision.models.convnext.LayerNorm2d):
            setattr(module, name, layers.VariationalLayerNorm2d(child, raw_sigma))
        elif isinstance(child, torch.nn.LayerNorm):
            setattr(module, name, layers.VariationalLayerNorm(child, raw_sigma))
        elif isinstance(child, torchvision.models.convnext.CNBlock):
            setattr(module, name, layers.VariationalCNBlock(child, raw_sigma))
            add_variational_layers(child, raw_sigma)
        elif isinstance(child, torch.nn.MultiheadAttention):
            setattr(module, name, layers.VariationalMultiheadAttention(child, raw_sigma))
        elif isinstance(child, torch.nn.Embedding):
            setattr(module, name, layers.VariationalEmbedding(child, raw_sigma))
        else:
            add_variational_layers(child, raw_sigma)
            
def use_posterior(self, flag):
    """
    Purpose: Turns on using the posterior for inference/predictive probabilities

    Arguments:
        flag: boolean to set to True if 
        raw_sigma: variance of the assumed approximate posterior

    Returns:
        Adds assumed approximate posterior variances to the weights/parameters from the model layers 
    """
    for child in self.modules():
        if isinstance(child, (
            layers.VariationalLinear, 
            layers.VariationalConv2d, 
            layers.VariationalBatchNorm2d,
            layers.VariationalLayerNorm2d,
            layers.VariationalLayerNorm,
            layers.VariationalCNBlock,
            layers.VariationalMultiheadAttention,
            layers.VariationalEmbedding,
        )):
            child.use_posterior = flag
            
def flatten_params(model, excluded_params=["raw_lengthscale", "raw_noise", "raw_outputscale", "raw_sigma", "raw_tau"]):
    """
    Purpose: Flattens all model parameters together 

    Arguments:
        model: PyTorch MODEL  
        excluded_params: parameters of the assumed approximate variational distribution 

    Returns:
        Flattens the parameters into a list
    """
    return torch.cat([param.view(-1) for name, param in model.named_parameters() if param.requires_grad and name not in excluded_params])

def plot_losses(num_epochs, loss_data, loss_type, seed, lr): 

    """
    Purpose: Plot the data likelihood, prior/entropy term/KL, and the full Negative DE-ELBO loss curves 

    Arguments:
        num_epochs: number of training iterations 
        loss_data: 1-D numpy array of loss values
        loss_type: str for "Negative ELBO", "kl", "nll"
        seed: random initialization seed 
        lr: learning rate  

    Returns:
        Saves loss plots 
    """

    if loss_type == "loss":
        loss_type = "Negative ELBO"
    else: 
        loss_type = loss_type
    fig = plt.figure(figsize=(3.8, 2))
    plt.tight_layout()  
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.18, top=0.95)  
    plt.rcParams.update({
    "font.family": "serif",
    "text.usetex": False,   
    "font.size": 9,        
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.2,})  
    plt.plot([i for i in range(num_epochs)], loss_data, c = "black")
    plt.xlabel("Number of Iterations")
    plt.ylabel(loss_type)
    plt.yscale("log")
    repo_root = Path(__file__).resolve().parents[2]
    plots_dir = repo_root / "results" / "feedback_mechanism" / f"log-epochs-{num_epochs}" /f"{loss_type}_{seed}_{lr}.pdf"
    plt.savefig(plots_dir)

def save_evidence_dict(evidence_dict):
    """
    Purpose: Save a dictionary of the final Negative DE-ELBO values with the corresponding seeds and lr values for the model run into a CSV file. 

    Arguments:
        evidence dict: dictionary of Negative DE-ELBO values with model run arguments (seeds, lr)

    Returns:
        Saves dictionary to CSV file. 
    """

    records = []
    for key, value in evidence_dict.items():
        # split the "seed and lr" string
        seed_str, lr_str = key.split(" and ")
        records.append({
            "seed": int(seed_str),
            "lr": float(lr_str),
            "final_loss": float(value)
        })

    df = pd.DataFrame(records)
    repo_root = Path(__file__).resolve().parents[2]
    plots_dir = repo_root / "results" / "feedback_mechanism" / "final_losses.csv"
    df.to_csv(plots_dir, index=False)
    return df