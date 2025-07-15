import torch
import numpy as np
from tqdm import tqdm
import utils
from utils import checkpoint, warning
import integration_schemes as integ
import functions as funct
import metrics

def MCLMC(d, N, L, epsilon, fn, int_scheme=integ.leapfrog, x0=None, debug=False, pbar=False, **kwargs):
    """
    This function implements the MCLMC algorithm for the q=0 case.
    Args:
        d: dimension of the problem
        N: number of steps
        L: distance between momentum refreshment
        epsilon: step size
        fn: function to sample from
        int_scheme: either leapfrog or minimal_norm
    Output:
        X: positions during evolution
        E: energies during evolution
        ESS: effective sample sizes during evolution
    """

    device = utils.choose_device()

    X = torch.zeros((N+1, d), device=device)
    E = torch.zeros(N+1, device=device)

    # STEP 0: Intial conditions
    if x0 is None:
        if fn == funct.bimodal:
            mu = np.zeros(d)
            mu[0] = np.random.choice([0,8], size=1, p=[0.8, 0.2])
            x = torch.normal(mean=torch.tensor(mu, dtype=torch.float32), std=1.0)
        else:
            x = np.random.standard_normal(d) ###### DEBUGGING PRIOR ONLY, CHANGE LATER
    else:
        x = x0.copy()
        
    x = torch.tensor(x, dtype=torch.float32, device=device)
    x.requires_grad_()

    u = torch.randn(d, device=device) # Sample initial direction of momentum u_0 from isotropic distribution in R^d
    u /= torch.linalg.norm(u)

    w = 1 # Set the intial weight
    w = torch.tensor(w, requires_grad=False, dtype=torch.float32, device=device)

    X[0] = x.detach()
    E[0] = utils.energy(x=x, w=w, d=d, fn=fn, **kwargs)

    # EVOLUTION: Algorithm implementation
    if pbar:
        bar = tqdm(range(1,N+1))
    else:
        bar = range(1,N+1)
        
    for n in bar:

        # Updating coordinate and momentum
        x, u, w = int_scheme(x, u, w, epsilon, d, fn, **kwargs)
        x, u, w = integ.stochastic_update_map(x, u, w, epsilon, L, d, fn, **kwargs)

        # Storing results
        X[n] = x.detach()
        E[n] = utils.energy(x, w,d, fn, **kwargs)
        
    else:
        return X.cpu(), E.cpu()


def MCLMC_chain(n_chains, d, fn, L, eps, int_scheme, N=5000, **kwargs):
    
    X = []
    ESS_mean = []
    ESS_min = []
    ESS_truth = []
    
    for i in range(n_chains):
        Xi, *_ = MCLMC(d=d, N=N, L=L, epsilon=eps, fn=fn, int_scheme=int_scheme, pbar=True, **kwargs)
        X.append(Xi[500:,:]) # burn-in
        ess_mean = metrics.ESS(Xi)
        ess_min = metrics.ESS(Xi, take_minimum=True)
        ESS_mean.append(ess_mean)
        ESS_min.append(ess_min)
        
        if fn == funct.bimodal or fn == funct.ill_cond_gaussian:
            ess_truth = metrics.ESS_truth(Xi, fn=fn, **kwargs)
            ESS_truth.append(ess_truth)

    ESS_mean = np.array(ESS_mean)
    ESS_min = np.array(ESS_min)
    ESS_truth = np.array(ESS_truth)

    ess_mean = np.mean(ESS_mean[ESS_mean!=None])
    ess_min = np.mean(ESS_min[ESS_min!=None])
    ess_truth = np.mean(ESS_truth[ESS_truth!=None])

    X = torch.cat(X, dim=0)

    return X, ess_mean, ess_min, ess_truth       