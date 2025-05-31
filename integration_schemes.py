import numpy as np
import torch

import utils
from utils import checkpoint, warning

def grad_log_likelihood(x, fn, **kwargs):
    """
    This function takes an array of d-dimensional points as input
        x = [x_1, x_2, ..., x_d]     with x_i  in R^d
    and a function fn that takes this array as argument.

    Then, computes the gradient (vector of the partial derivatives) of the loglikelihood
    of this function and evaluates it on each of the points in x
    """

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, requires_grad=True, dtype=torch.float32)

    x = x.detach().requires_grad_() # Returns a new Tensor, detached from the current graph
    #print(f'##### FN \n{fn(x)}')

    #L = -torch.log(fn(x))
    L = -fn(x, **kwargs)
    #print(f'##### NLOGL \n{L}')
    L.sum().backward() # with this command it calculates the gradients of L with respect to x and stores them in x.grad

    nablaL = x.grad

    x.grad = None # clears the gradient

    return nablaL

def position_update_map(x: np.ndarray, u: np.ndarray, w: float, epsilon: float) -> tuple:
    new_x = x + epsilon * u
    new_u = u
    new_w = w
    return new_x, new_u, new_w

def momentum_update_map(x: np.ndarray, u: np.ndarray, w: float, epsilon: float, d: int, fn, **kwargs) -> tuple:
    nablaL = grad_log_likelihood(x, fn, **kwargs) # computing the gradient of the loglikelihood of the pdf fn
    #print(f'##### GRAD \n{nablaL}')
    delta = epsilon * torch.linalg.norm(nablaL) / d
    e = - nablaL / torch.linalg.norm(nablaL)
    # updating position
    new_x = x

    # updating momentum
    prod = torch.dot(e,u)
    s = torch.sinh(delta)
    c = torch.cosh(delta)
    new_u = (u + e * (s + prod * (c - 1)))/(c + prod * s)
    #new_u /= np.linalg.norm(new_u)

    # updating weight
    new_w = w * (c + prod * s)
    return new_x, new_u, new_w

def stochastic_update_map(x, u, w, epsilon, L, d, fn, **kwargs):
    nu = torch.sqrt((torch.e**(2*epsilon/L))-1)/d

    # updating position
    new_x = x

    # updating momentum
    device = utils.choose_device()
    z = torch.normal(0, 1, size=(d,)).to(device)
    new_u = u + nu * z
    new_u /= torch.linalg.norm(new_u)

    # updating weight
    new_w = w

    return new_x, new_u, new_w


def leapfrog(x: np.ndarray, u: np.ndarray, w: float, epsilon: float, d: int, fn, **kwargs) -> tuple:

    # Update momentum by half step
    x, u, w = momentum_update_map(x, u, w, epsilon/2, d, fn, **kwargs)

    # Update position by one step
    x, u, w = position_update_map(x, u, w, epsilon)

    # Update momentum by half step
    x, u, w = momentum_update_map(x, u, w, epsilon/2, d, fn, **kwargs)

    return x, u, w

def minimal_norm(x: np.ndarray, u: np.ndarray, w: float, epsilon: float, d: int, fn, **kwargs) -> tuple:

    l = 0.19318

    x, u, w = momentum_update_map(x, u, w, l*epsilon, d, fn, **kwargs)
    x, u, w = position_update_map(x, u, w, epsilon/2)
    x, u, w = momentum_update_map(x, u, w, epsilon*(1-2*l), d, fn, **kwargs)
    x, u, w = position_update_map(x, u, w, epsilon/2)
    x, u, w = momentum_update_map(x, u, w, epsilon*l, d, fn, **kwargs)

    return x, u, w