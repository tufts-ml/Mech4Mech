import math
# PyTorch
import torch

class IsotropicGaussianPrior(torch.nn.Module):

    """
    Purpose: The Isotropic Gaussian Prior defines a single variance across outputs with zero everywhere else in the covariance 
    matrix. This is the prior over model parameters, and this assumes probabilistic independence in the parameters. 

    Attributes: 
        prior_params: params of the prior 
        num_params: total number of parameters 
        prior_variance: variance parameter of the prior 
    """
    def __init__(self, num_params=None, prior_params=None, prior_variance=1.0):
        super().__init__()
        self.prior_params = prior_params
        self.num_params = len(self.prior_params) if num_params is None else num_params
        self.prior_variance = torch.tensor(prior_variance, dtype=torch.float32)

    def kl(self, params, sigma):
        """
        Purpose: Computes the KL divergence between the prior and the assumed posterior 

        Attributes: 
            params: params of the posterior 
            sigma: variance of the prior  
        Return: KL of the probability distributions 
        """
        assert len(params) == self.num_params
        params_diff_norm = (params**2).sum() if self.prior_params is None else ((params - self.prior_params)**2).sum()
        if sigma.shape == ():
            prior_variance = (params_diff_norm + sigma**2 * self.num_params) / self.num_params
            trace = (sigma**2 / prior_variance) * self.num_params
            quad_term = (1 / prior_variance) * params_diff_norm
            log_det = self.num_params * torch.log(prior_variance) - self.num_params * torch.log(sigma**2)
        elif sigma.shape == (self.num_params,):
            prior_variance = (params_diff_norm + (sigma**2).sum()) / self.num_params
            trace = (sigma**2).sum() / prior_variance
            quad_term = (1 / prior_variance) * params_diff_norm
            log_det = self.num_params * torch.log(prior_variance) - torch.log(sigma**2).sum()
        kl = 0.5 * (trace + quad_term - self.num_params + log_det)
        return kl
    
    def log_prob(self, params):
        """
        Purpose: Computes the log probabilities of the prior 

        Attributes: 
            params: params of the model

        Return: Log probs 
        """
        assert len(params) == self.num_params
        params_diff_norm = (params**2).sum() if self.prior_params is None else ((params - self.prior_params)**2).sum()
        log_norm_const = self.num_params * math.log(2.0 * math.pi)
        log_det = self.num_params * torch.log(self.prior_variance)
        quad_term = (1 / self.prior_variance) * params_diff_norm
        log_prob = -0.5 * (log_norm_const + log_det + quad_term)
        return log_prob
          
