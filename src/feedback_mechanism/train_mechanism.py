import types
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
# PyTorch
import torch
from likelihoods import CategoricalLikelihood
from losses import TemperedELBOLoss
from priors import IsotropicGaussianPrior
from utils import flatten_params, inv_softplus, use_posterior, add_variational_layers, plot_losses, save_evidence_dict


def main():
    """
    Purpose: Implements the Negative DE-ELBO training of the NN model for the feedback mechanism. 

    Returns:
        Best model in the best_model.pt file, saves the DE-ELBO dictionary into a CSV file (results) -> (feedback_mechanism) -> (final_losses.csv), 
        and plots the Negative DE-ELBO losses in (results) -> (feedback_mechanism) -> (log-epochs-__) 
    """
    # --- define data path ---
    repo_root = Path(__file__).resolve().parents[2]  
    data_dir = repo_root / "data" / "feedback_mechanism"      
    path = data_dir / "training_data.npz"
    bundle = np.load(path, allow_pickle=True)

    # --- call cpu/gpu device ---
    device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
    # --- extract data ---
    embeddings = bundle["embeddings"]          # (N, 128)
    labels = bundle["labels"]                  # (N, K)
    texts = bundle["texts"].tolist()           # List of all text strings
    label_names = bundle["label_names"].tolist() #List of label names 
    K = 8 #Number of label classes
    D = bundle["emb_dim"] #Embedding dimension
    
    N = len(texts)
    class_weights = torch.tensor([1/1452 , 1/5 , 1/151 , 1/53, 1/19, 1/29, 1/10, 1/2], dtype=torch.float32)

    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    # --- tensors on desired device ---
    X = torch.tensor(embeddings, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    class_weights = class_weights.to(device)

    # --- create 2-layer NN ---
    model = torch.nn.Sequential(
        torch.nn.Linear(in_features=D, out_features=D),
        torch.nn.ReLU(inplace=False),
        torch.nn.Linear(in_features=D, out_features=K),
    ).to(device)

    num_classifier_params = len(flatten_params(model))

    model.raw_sigma = torch.nn.Parameter(inv_softplus(torch.tensor(1e-4, device=device))) # Add a single variance parameter raw_sigma of the approximate posterior q(theta |x,y,M) = N(current theta, raw_sigma^2I) that we will learn via gradient descent.
    add_variational_layers(model, model.raw_sigma) # Make all regular layers variational - have parameters that can be obtained from the approximate posterior.

    model.use_posterior = types.MethodType(use_posterior, model) # Lets us use the approximate posterior when in torch evaluation mode in order to compute and select the lowest ELBO. 
    #Rather than just using the current weights, we use sample weights from q via the re-parameterization trick. We need this for computing the ELBO; we do not need this for future test.

    likelihood = CategoricalLikelihood(num_classes=K).to(device) #Assume a true categorical likelihood for our data p(y|x, theta, M); includes softmax function via torch. 
    prior = IsotropicGaussianPrior(num_params=num_classifier_params).to(device) #Assume a true isotropic gaussian prior p(theta|m). Contains a weight decay that is learned via a closed form solution. 

    init_model_state = deepcopy(model.state_dict())
    init_lik_state = deepcopy(likelihood.state_dict()) 
    init_prior_state = deepcopy(prior.state_dict()) 

    criterion = TemperedELBOLoss(model, likelihood, prior, kappa=(num_classifier_params)/N) #Computes the ELBO: E_q log p(y|x, theta, M) - KL(q(theta|x,y,m) || p(theta|m)).

    lr_list = [3.3, 2.9, 2.5, 2.1, 1.7, 1.3, 0.9, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]  #List of learning rates. Older rates: 0.001, 0.0001, 0.00001, 0.000001
    num_epochs = 1000 #Number of epochs. 
    seed_list = [7, 123, 213, 512, 637] # A seed list of starting points for initial params 
    best_loss = float('inf') 
    evidence_dict = {} #Logs the final ELBO values that are associated with each learning rate. 
    kL_dict = {} #Logs the final KL values that are associated with each learning rate.
    nll_dict = {} #Logs the final NLL values that are associated with each learning rate.
    save_path = repo_root / "src" / "feedback_mechanism" / "best_model.pt"


    for seed in seed_list: 
        for lr in lr_list: 

            torch.manual_seed(seed)
            model.load_state_dict(init_model_state) #Ensures the same initial parameters for a new run. 
            likelihood.load_state_dict(init_lik_state) #Ensures the same initial parameters for a new run.
            prior.load_state_dict(init_prior_state) #Ensures the same initial parameters for a new run. 

            optimizer = torch.optim.SGD([{"params": model.parameters()}, {"params": likelihood.parameters()}, {"params": prior.parameters()}], lr=lr, weight_decay=0.0, momentum=0.9, nesterov=True)
            loss_list = []
            kl_list = []
            nll_list = []
            for epoch in range(num_epochs):

                optimizer.zero_grad()
                params = flatten_params(model)

                logits = model(X)
                loss = criterion(logits, y, class_weights, params, len(X)) #Computes the ELBO 

                loss_list.append(loss["loss"].item())
                kl_list.append(loss["kl"].item())
                nll_list.append(loss["nll"].item())

                if epoch % 50 == 0:
                    print(f"Epoch {epoch} | LR {lr} | ELBO loss: {loss['loss'].item():.6}")
                loss["loss"].backward()
                optimizer.step()

            if loss["loss"].item() < best_loss: 
                best_loss = loss["loss"].item() 
                torch.save(model.state_dict(), save_path) #Only saves the model parameters with the seed and learning rate that has the lowest negative ELBO. 

            plot_losses(num_epochs, loss_list, "loss", seed, lr)
            plot_losses(num_epochs, kl_list, "kl", seed, lr)
            plot_losses(num_epochs, nll_list, "nll", seed, lr)

            evidence_dict[f"{seed} and {lr}"] = loss["loss"].item()
            kL_dict[f"{seed} and {lr}"] = loss["kl"].item()
            nll_dict[f"{seed} and {lr}"] = loss["nll"].item()
        
    #save_evidence_dict(evidence_dict)
    print("Best ELBO:", best_loss)
    print("Evidence by LR:", evidence_dict)
    print("KL by LR:", kL_dict)
    print("NLL by LR:", nll_dict)


if __name__ == "__main__":
    main()





