# PyTorch
import torch

class CategoricalLikelihood(torch.nn.Module):
    """
    Purpose: Define the categorical likelihood. 

    Attributes: 
        num_classes: number of classes/categories for labels 
    """
    def __init__(self, num_classes):
        super().__init__()
        
        self.num_classes = num_classes
            
    def forward(self, logits, labels, class_weights, reduction="mean"):
        return torch.nn.functional.cross_entropy(logits, labels, class_weights, reduction=reduction)
