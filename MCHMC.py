#hellooooo
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


def MCHMC_bounces(d, N, L, epsilon, fn, int_scheme=integ.leapfrog, debug=False):
    """
    This function implements the MCHMC algorithm for the q=0 case with random momentum
    bounces every K steps.
    Args:
        d: dimension of the problem
        N: number of steps
        L: distance between momentum bounces
        epsilon: step size
        fn: function to sample from
    Output:
        X: positions during evolution
        E: energies during evolution
        ESS: effective sample sizes during evolution
        b_squared: values useful for one figure of the paper

    """
    device = utils.choose_device()

    #safety check
    #if epsilon > L:
    #    sys.exit(f"Epsilon passed: {epsilon}  BUT should be smaller than the length between bounces L: {L}")
    K = int(L // epsilon) #  steps between bounces

    # Defining tensors where to store results of evolution
    X = torch.zeros((N+1, d), device=device)
    E = torch.zeros(N+1)
    B_squared = torch.zeros(N+1)

    # STEP 0: Intial conditions
    x = np.random.uniform(low=-2, high=2, size=(d,)) # Sample initial position x_o in R^d from prior
    x = torch.tensor(x, dtype=torch.float32, device=device)
    x.requires_grad_()

    u = torch.randn(d, device=device) # Sample initial direction of momentum u_0 from isotropic distribution in R^d
    u /= torch.linalg.norm(u)

    w = 1 # Set the intial weight
    w = torch.tensor(w, requires_grad=False, dtype=torch.float32, device=device)

    checkpoint(f"Step {0} (initialization):", debug)
    #checkpoint(f"\tx = {x}\n\tu = {u}\n\tw = {w}", debug)

    X[0] = x.detach()
    E[0] = utils.energy(x, w, d, fn)
    ESS, b_squared = utils.effective_sample_size(X, d, cauchy=True, debug=debug)
    B_squared[0] = b_squared

    #if d < 100:
    #    warning(f"For the validity of ESS results d should be very large but I am using d={d}")
    #warning("Check that X has not been flatten")

    # EVOLUTION: Algorithm implementation
    for n in tqdm(range(1,N+1)):

        #checkpoint(f"Step {n}:", debug)

        # if K steps have been done, apply a bounce
        if n % K == 0:
          u = torch.randn(d, device=device)
          u /= torch.linalg.norm(u)

        # Updating coordinate and momentum
        x, u, w = int_scheme(x, u, w, epsilon, d, fn)
        #checkpoint(f"\tx = {x}\n\tu = {u}\n\tw = {w}", debug)

        '''
        if not np.isclose(np.linalg.norm(u), 1.0, atol=1e-4):
            sys.exit(f"Vector u should be normalized, while its norm is {np.linalg.norm(u)}.")
        '''

        # Storing results
        X[n] = x.detach()
        E[n] = utils.energy(x, w, d, fn)
        ESS, b_squared = utils.effective_sample_size(X, d, cauchy=True, debug=debug) #compute it after determine the new points in phase space
        B_squared[n] = b_squared

    return X.cpu(), E.cpu(), ESS.cpu(), B_squared.cpu()