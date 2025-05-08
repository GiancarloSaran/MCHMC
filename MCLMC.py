import torch
import numpy as np

import utils
from utils import checkpoint, warning
import integration_schemes as integ


def MCLMC(d, N, L, epsilon, fn, int_scheme=integ.leapfrog, debug=False):

    device = utils.choose_device()

    X = torch.zeros((N+1, d), device=device)
    E = torch.zeros(N+1)

    # STEP 0: Intial conditions
    x = np.random.uniform(low=-2, high=2, size=(d,)) # Sample initial position x_o in R^d from prior
    x = torch.tensor(x, dtype=torch.float32, device=device)
    x.requires_grad_()

    u = torch.randn(d, device=device) # Sample initial direction of momentum u_0 from isotropic distribution in R^d
    u /= torch.linalg.norm(u)

    w = 1 # Set the intial weight
    w = torch.tensor(w, requires_grad=False, dtype=torch.float32, device=device)

    checkpoint(f"Step {0} (initialization):", debug)
    checkpoint(f"\tx = {x}\n\tu = {u}\n\tw = {w}", debug)


    X[0] = x.detach()
    E[0] = utils.energy(x, w, d, fn)
    ESS = utils.effective_sample_size(X, d, cauchy=True, debug=debug)


    # EVOLUTION: Algorithm implementation
    for n in range(1,N+1):

        checkpoint(f"Step {n+1}:", debug)

        # Updating coordinate and momentum
        x, u, w = int_scheme(x, u, w, epsilon, d, fn)
        x, u, w = integ.stochastic_update_map(x, u, w, epsilon, L, d, fn)

        '''
        if d<10:
            checkpoint(f"\tx = {x}\n\tu = {u}\n\tw = {w}", debug)
        if not np.isclose(np.linalg.norm(u), 1.0, atol=1e-4):
            sys.exit(f"Vector u should be normalized, while its norm is {np.linalg.norm(u)}.")
        '''

        # Storing results
        X[n] = x.detach()
        E[n] = utils.energy(x, w,d, fn)
        ESS = utils.effective_sample_size(X, d, cauchy=True, debug=debug)

    return X.cpu(), E.cpu(), ESS.cpu()