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
from utils import flatten_params, inv_softplus, use_posterior, add_variational_layers


def main():
    # --- define data path ---
    repo_root = Path(__file__).resolve().parents[2]  
    data_dir = repo_root / "data" / "embeddings"      
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

    num_of_params = len(flatten_params(model)) # + 308,000,000 Add an additional 308 million params to represent the number of params in gemmaembed LLM.

    model.raw_sigma = torch.nn.Parameter(inv_softplus(torch.tensor(1e-4, device=device))) # Add a single variance parameter raw_sigma of the approximate posterior q(theta |x,y,M) = N(current theta, raw_sigma^2I) that we will learn via gradient descent.
    add_variational_layers(model, model.raw_sigma) # Make all regular layers variational - have parameters that can be obtained from the approximate posterior.

    model.use_posterior = types.MethodType(use_posterior, model) # Lets us use the approximate posterior when in torch evaluation mode in order to compute and select the lowest ELBO. 
    #Rather than just using the current weights, we use sample weights from q via the re-parameterization trick. We need this for computing the ELBO; we do not need this for future test.

    likelihood = CategoricalLikelihood(num_classes=K).to(device) #Assume a true categorical likelihood for our data p(y|x, theta, M); includes softmax function via torch. 
    prior = IsotropicGaussianPrior(num_params=num_of_params).to(device) #Assume a true isotropic gaussian prior p(theta|m). Contains a weight decay that is learned via a closed form solution. 

    init_model_state = deepcopy(model.state_dict())
    init_lik_state = deepcopy(likelihood.state_dict()) 
    init_prior_state = deepcopy(prior.state_dict()) 

    criterion = TemperedELBOLoss(model, likelihood, prior, kappa=num_of_params/N) #Computes the ELBO: E_q log p(y|x, theta, M) - KL(q(theta|x,y,m) || p(theta|m)).

    lr_list = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001] #List of learning rates. 
    best_loss = float('inf') 
    evidence_dict = {} #Logs the evidence values that are associated with each learning rate. 
    weight_decay_dict = {}
    save_path = repo_root / "src" / "feedback_mechanism" / "best_model.pt"

    for lr in lr_list: 

        torch.manual_seed(123)
        model.load_state_dict(init_model_state) #Ensures the same initial parameters for a new run. 
        likelihood.load_state_dict(init_lik_state) #Ensures the same initial parameters for a new run.
        prior.load_state_dict(init_prior_state) #Ensures the same initial parameters for a new run. 

        optimizer = torch.optim.SGD([{"params": model.parameters()}, {"params": likelihood.parameters()}, {"params": prior.parameters()}], lr=lr, weight_decay=0.0, momentum=0.9, nesterov=True)
        for epoch in range(1000):

            optimizer.zero_grad()
            params = flatten_params(model)

            logits = model(X)
            loss = criterion(logits, y, class_weights, params, len(X)) #Computes the ELBO 
            if epoch % 50 == 0
                print(f"Epoch {epoch} | LR {lr} | ELBO loss: {loss['loss'].item():.6}")
            loss["loss"].backward()
            optimizer.step()

        if loss["loss"].item() < best_loss: 
            best_loss = loss["loss"].item() 
            torch.save(model.state_dict(), save_path) #Only saves the model parameters with the learning rate that has the lowest ELBO. 
        evidence_dict[lr] = loss["loss"].item()
        weight_decay_dict[lr] = prior.prior_variance

    print("Best ELBO:", best_loss)
    print("Evidence by LR:", evidence_dict)
    print("Prior variance by LR:", weight_decay_dict)


if __name__ == "__main__":
    main()





