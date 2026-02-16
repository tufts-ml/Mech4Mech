
import jax.numpy as jnp
import numpy as np 
import jax
import torch
import types
from pathlib import Path
from typing import Union
from utils import flatten_params, inv_softplus, use_posterior, add_variational_layers



def main():

    """
    Purpose: Test the feedback mechanism on the training data just to obtain a quick sanity check that it performs well on the data it was trained on. 

    Returns: the model accuracy - i.e. the number of correct labels over the total number of model generated labels 
    """
    # --- call cpu/gpu device ---
    device = torch.device("cpu")

        # --- define data path --- Using the same data as our training data to test the model for a sanity check 
    repo_root = Path(__file__).resolve().parents[2]  
    data_dir = repo_root / "data" / "feedback_mechanism"      
    path = data_dir / "training_data.npz"
    bundle = np.load(path, allow_pickle=True)

    # --- extract data ---
    embeddings = bundle["embeddings"]          # (N, 128)
    labels = bundle["labels"]                  # (N, K)
    texts = bundle["texts"].tolist()           # List of all text strings
    label_names = bundle["label_names"].tolist() #List of label names 
    K = 8 #Number of label classes
    D = bundle["emb_dim"] #Embedding dimension
    
    N = len(texts)

    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    # --- tensors on desired device ---
    X = torch.tensor(embeddings, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)

    model = torch.nn.Sequential(
    torch.nn.Linear(in_features=128, out_features=128),
    torch.nn.ReLU(inplace=False),
    torch.nn.Linear(in_features=128, out_features=8),
    ).to(device)


    model.raw_sigma = torch.nn.Parameter(inv_softplus(torch.tensor(1e-4, device=device))) 
    add_variational_layers(model, model.raw_sigma) # Make all regular layers variational - have parameters that can be obtained from the approximate posterior.

    model.use_posterior = types.MethodType(use_posterior, model) 

    repo_root = Path(__file__).resolve().parent
    model_dir = repo_root  / "best_model.pt"   

    state_dict = torch.load(model_dir, map_location="cpu")
    model.load_state_dict(state_dict)

    model.eval()   

    with torch.no_grad():
        logits = model(X)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=1)
    pred_prob = probs.argmax(dim=-1)

    y_true = y.argmax(dim=1) # y_true are individual class labels like preds rather than a one-hot vector like y 
    accuracy = (preds == y_true).float().mean()
    print(f"Accuracy: {accuracy.item():.4f}")

    classes = torch.unique(y_true)   # all classes present in labels

    for c in classes:
        idx = (y_true == c)
        correct_mask = idx & (preds == y_true)
        incorrect_mask = idx & (preds != y_true)

        # ---- correct predictions ----
        # prob assigned to the true class c
        prob_correct_class = probs[correct_mask, c]
        mean_prob_correct = (
            prob_correct_class.mean().item()
            if correct_mask.any() else float("nan")
        )

        # ---- incorrect predictions ----
        # prob assigned to the predicted (wrong) class
        prob_predicted_wrong = probs[incorrect_mask, preds[incorrect_mask]]
        mean_prob_incorrect = (
            prob_predicted_wrong.mean().item()
            if incorrect_mask.any() else float("nan")
        )

        print(
            f"Class {int(c)} | "
            f"p(true | correct)={mean_prob_correct:.4f} | "
            f"p(pred | incorrect)={mean_prob_incorrect:.4f} | "
            f"n_correct={correct_mask.sum().item()} | "
            f"n_incorrect={incorrect_mask.sum().item()}"
        )



if __name__ == "__main__":
    main()