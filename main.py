import torch
import numpy as np  
import sys

from utils import checkpoint, warning
import utils
import MCHMC, MCLMC
import autotuning as aut
import integration_schemes as integ
import functions as funct
import visualization as vis

debug = True

print("Cuda is available?", torch.cuda.is_available())
print("Device in using: ", torch.cuda.get_device_name(0))

# ALGORITHM PARAMETERS
d = 1000  # dimension of the problem
N = 10000 # number of steps

# AUTOTUNING PARAMETERS
L_init = 10
N_prerun_eps = 100
iterations_eps = 10
N_prerun_L = 100
eps, sigma_eff = aut.tune_eps(d, N_prerun_eps, L_init, funct.standard_cauchy, iterations=iterations_eps, debug=debug)
epsilon_opt = eps[-1]

L_opt = aut.tune_L(d, sigma_eff, epsilon_opt, N_prerun_L, funct.standard_cauchy, debug)

K = int(L_opt//epsilon_opt) #  steps between bounces

checkpoint(f"Number of steps N = {N}")
checkpoint(f"Steps between bounces K = {K}")
checkpoint(f"Number of bounces: {int(N//K)}")


X, E, ESS, B_squared = MCHMC.MCHMC_bounces(d, N, L_opt, epsilon_opt, funct.standard_cauchy, integ.leapfrog, debug=debug)

#plotting the results
vis.plot_cauchy(X)