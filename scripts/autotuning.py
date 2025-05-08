import torch
import numpy as np  

from utils import checkpoint, warning
import MCHMC

def tune_eps(d, L, fn, N, iterations=10, debug=False):
    checkpoint("Tuning epsilon..", debug=debug)
    checkpoint(f"\nParameters\n\td={d}\n\tL={L}\n\tNn={N}")

    epsilon = 0.5 # initial value
    checkpoint(f"Initial value: epsilon = {epsilon}")

    eps_values = torch.zeros(iterations)
    #Change to an "until convergence" criterion, could track ((eps_i+1 - eps_i)/eps_i)^2 and get it lower
    #than a threshold
    checkpoint(f"\nRunning multiple iterations of MCHMC_bounces with {N} steps, updating epsilon")
    for i in range(iterations):
        checkpoint(f"\n---------------------------\nIteration {i}")
        X, E, *_ = MCHMC.MCHMC_bounces(d, N, L, epsilon, fn, debug=debug)
        checkpoint(f"X.shape: {X.shape}, E.shape: {E.shape}")
        varE = E.var()
        checkpoint(f"varE: {varE}")
        epsilon *= (0.0005 * d / varE)**(1/4)
        checkpoint(f"New epsilon: {epsilon}")
        eps_values[i]=epsilon
    checkpoint(f"\tOptimized epsilon: {epsilon}")

    return eps_values, X, E

def s_eff(X):
  return np.sqrt((X**2).var(axis=0).mean())

def tune_L(d, sigma_ef, epsilon_optimized, N_prerun, fn, debug=False):
    checkpoint("\nTuning L..", debug=debug)
    #initial value
    L_init = sigma_ef * np.sqrt(d)

    # Run for n steps to estimate the effective sample size for each dimension
    X, *_ = MCHMC.MCHMC_bounces(d, N_prerun, L_init, epsilon_optimized, fn, debug=False)

    n_eff_values = utils.effective_sample_size(X, d, cauchy=True, autotune_mode=True)

    # Computing the dechoerence scale L
    L = 0.4 * epsilon_optimized * d * N_prerun / (torch.sum(n_eff_values))
    checkpoint(f"\tOptimized L: {L}")

    return L