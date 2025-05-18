import torch
import numpy as np  
from utils import checkpoint, warning
import utils
import MCHMC
import MCLMC

def tune_eps(d, N, L, fn, algorithm, iterations=10, debug=True):

    eps_values = torch.zeros(iterations)
    sigma_effs = torch.zeros(iterations)
    
    #Change to an "until convergence" criterion, could track ((eps_i+1 - eps_i)/eps_i)^2 and 
    #get it lower than a threshold
    
    checkpoint(f"\nRunning {iterations} iterations of {algorithm} with {N} steps, updating epsilon")
    
    for i in range(iterations):

        epsilon = 0.5 # initial value

        if algorithm == MCHMC.MCHMC_bounces:
            X, E = MCHMC.MCHMC_bounces(d, N, L, epsilon, fn)
        else:
            X, E = MCLMC.MCLMC(d, N, L, epsilon, fn)
            
        varE = E.var()
        epsilon *= (0.0005 * d / varE)**(1/4)
        
        eps_values[i]=epsilon

        sigma_eff = torch.sqrt((X**2).var(axis=0).mean())
        sigma_effs[i] = sigma_eff
    
    return eps_values, sigma_effs


def tune_L(sigma_eff, eps_opt, d, N, fn, algorithm, iterations=10, debug=False, cauchy=False):

    L_values = torch.zeros(iterations)
    checkpoint(f"\nRunning {iterations} iterations of {algorithm} with {N} steps, updating L")

    for i in range(iterations):
        
        L_init = sigma_eff * np.sqrt(d)
        
        if algorithm == MCHMC.MCHMC_bounces:
          X, *_ = MCHMC.MCHMC_bounces(d, N, L_init, eps_opt, fn, debug=False)
        else:
          X, *_ = MCLMC.MCLMC(d, N, L_init, eps_opt, fn, debug=False)
    
        n_eff_values = utils.effective_sample_size(X, d, cauchy=cauchy, L_tuning=True)
    
        # Computing the dechoerence scale L
        L = 0.4 * eps_opt * d * N / (torch.sum(n_eff_values))
        L_values[i] = L
          
    return L_values