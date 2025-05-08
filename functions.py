import torch
import numpy as np  

from utils import checkpoint, warning

def standard_cauchy(x):
    """
    This function takes an array of d-dimensional points as input
        x = [x_1, x_2, ..., x_d]

    For each point x_i, it computes the value of the cauchy distribution as:
        p(x_i) = ( 1/(1+x_{i,1}**2) )*( 1/(1+x_{i,2}**2) )*...*( 1/(1+x_{i,d}**2) )

    The output is a tensor containing the values of the distribution for each point.
    """
    # serve che sia tensore di torch per implementazione del gradiente
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if x.dim() == 1:
        x = x.unsqueeze(1)
    return torch.prod(1 / (np.pi * (1 + x**2)), axis=1)