import os
import copy
import numpy as np
from sklearn.model_selection import train_test_split
import scipy
import matplotlib.pyplot as plt 
from pathlib import Path
# PyTorch
import torch
import torchvision
import torchmetrics
# Importing our custom module(s)
import layers

def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))
        
def add_variational_layers(module, raw_sigma):
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
    return torch.cat([param.view(-1) for name, param in model.named_parameters() if param.requires_grad and name not in excluded_params])

def plot_losses(num_epochs, loss_data, loss_type, lr): 

    if loss_type == "loss":
        loss_type = "ELBO"
    else: 
        loss_type = loss_type
    plt.figure()
    plt.plot([i for i in range(num_epochs)], loss_data)
    plt.title(f"lr: {lr} and loss_type: {loss_type}")
    plt.xlabel("Number of Iterations")
    plt.ylabel(loss_type)

    repo_root = Path(__file__).resolve().parents[2]
    plots_dir = repo_root / "results" / "feedback_mechanism" / f"{loss_type}_{lr}.pdf"
    plt.savefig(plots_dir)

