import torch
import numpy as np  
from torch.distributions.multivariate_normal import MultivariateNormal
from utils import checkpoint, warning
import utils

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

def bimodal(x):
    '''
    This function takes an array of d-dimensional points as input
        x = [x_1, x_2, ..., x_d]

    For each point x_i, it computes the value of the bimodal distribution as:
        p(x_i) = w_1 * p(x_i) + w_2 * p(x_i)
      where p(x_i) is a standard 50-dimensional normal distribution.

    The output is a tensor containing the values of the distribution for each point.
    '''
    device = utils.choose_device()

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if x.dim() == 1:
        x = x.unsqueeze(1)

    x = x.to(device)
    d = x.shape[1]
    mu1 = torch.zeros(d, device=device)
    mu2 = torch.full((d,), 8.0 / (d ** 0.5), device=device) # they are separated by 8 sigma in the d-dimensional space 

    pdf1 = MultivariateNormal(mu1, torch.eye(d, device=device)).log_prob(x).exp()
    pdf2 = MultivariateNormal(mu2, torch.eye(d, device=device)).log_prob(x).exp()
    pdf = 0.8*pdf1 + 0.2*pdf2

    return pdf