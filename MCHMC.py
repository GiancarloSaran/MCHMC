import torch
import numpy as np
import time
from datetime import datetime
import pytz
from tqdm import tqdm
import sys
import utils
from utils import checkpoint, warning
import integration_schemes as integ
import functions as funct


def MCHMC_bounces(d, N, L, epsilon, fn, int_scheme=integ.leapfrog, metrics=False, debug=False, pbar=False):
    """
    This function implements the MCHMC algorithm for the q=0 case with random momentum
    bounces every K steps.
    Args:
        d: dimension of the problem
        N: number of steps
        L: distance between momentum bounces
        epsilon: step size
        fn: function to sample from
        int_scheme: either leapfrog or minimal_norm
        metrics: if True returns the ESS and b_squared metrics 
    Output:
        X: positions during evolution
        E: energies during evolution
        ESS: effective sample sizes during evolution
        b_squared: values useful for one figure of the paper

    """
    device = utils.choose_device()

    K = int(L // epsilon) #  steps between bounces

    # idea da discutere
    if K==0:
        K=1

    # Defining tensors where to store results of evolution
    X = torch.zeros((N+1, d), device=device)
    E = torch.zeros(N+1, device=device)
    
    if metrics:
        B_squared = torch.zeros(N+1)

    # STEP 0: Intial conditions
    if fn == funct.bimodal:    
        mu = np.zeros(d)
        mu[0] = np.random.choice(np.array([0,8]))
        x = np.random.normal(loc=mu, scale=1, size=(d, ))
    else:
        x = np.random.uniform(low=-10, high=10, size=(d,)) # Sample initial position x_o in R^d from prior
        
    x = torch.tensor(x, dtype=torch.float32, device=device)
    x.requires_grad_()

    u = torch.randn(d, device=device) # Sample initial direction of momentum u_0 from isotropic distribution in R^d
    u /= torch.linalg.norm(u)

    w = 1 # Set the intial weight
    w = torch.tensor(w, requires_grad=False, dtype=torch.float32, device=device)

    X[0] = x.detach()
    E[0] = utils.energy(x, w, d, fn)

    if fn == funct.standard_cauchy:
        cauchy=True
    
    if metrics:
        ESS, b_squared = utils.effective_sample_size(X, d, cauchy=cauchy)
        B_squared[0] = b_squared

    # EVOLUTION: Algorithm implementation
    if pbar:
        bar = tqdm(range(1,N+1))
    else:
        bar = range(1,N+1)
    
    for n in bar:

        # if K steps have been done, apply a bounce
        if n % K == 0:
          u = torch.randn(d, device=device)
          u /= torch.linalg.norm(u)

        # Updating coordinate and momentum
        x, u, w = int_scheme(x, u, w, epsilon, d, fn)

        # Storing results
        X[n] = x.detach()
        E[n] = utils.energy(x, w, d, fn)
        
        if metrics:
            ESS, b_squared = utils.effective_sample_size(X, d, cauchy=cauchy, debug=debug) #compute it after determining the new points in phase space
            B_squared[n] = b_squared
            
    if metrics:
        return X.cpu(), E.cpu(), ESS.cpu(), B_squared.cpu()
    else:
        return X.cpu(), E.cpu()