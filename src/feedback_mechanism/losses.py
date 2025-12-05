# PyTorch
import torch
import torchmetrics

class TemperedELBOLoss(torch.nn.Module):
    """
    Purpose: Define the TemperedELBO, aka the DE-ELBO (data emphasized-ELBO). 

    Attributes: 
        kappa: weight to upweight the data term in the normal ELBO (or downweight the prior term)
        likelihood: likelihood term 
        model: Pytorch NN model 
        prior: prior distribution  
    """
    def __init__(self, model, likelihood, prior, kappa=1.0):
        super().__init__()
        self.kappa = kappa
        self.likelihood = likelihood
        self.model = model
        self.prior = prior

    def forward(self, logits, labels, class_weights, params, N):
        nll = self.likelihood(logits, labels, class_weights)
        sigma = torch.nn.functional.softplus(self.model.raw_sigma)
        kl = self.prior.kl(params, sigma)
        loss = nll + (1 / self.kappa) * (1 / N) * kl
        return {"kl": kl, "loss": loss, "nll": nll}
        