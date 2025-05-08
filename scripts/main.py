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

################################################ PARAMETERS ##########################################################

debug = True

# ALGORITHM PARAMETERS
d = 1000  # dimension of the problem
N = 10000 # number of steps

# AUTOTUNING PARAMETERS
L_init = 10
N_prerun_eps = 100
iterations_eps = 10
N_prerun_L = 100

################################################ AUTOTUNING ##########################################################


eps, X, E = aut.tune_eps(d, L_init, funct.standard_cauchy, N_prerun_eps, iterations=iterations_eps, debug=debug)
epsilon_opt = eps[-1]

sigma_eff = aut.s_eff(X)
L_opt = aut.tune_L(d, sigma_eff, epsilon_opt, N_prerun_L, funct.standard_cauchy, debug)

K = int(L_opt//epsilon_opt) #  steps between bounces

checkpoint(f"Number of steps N = {N}")
checkpoint(f"Steps between bounces K = {K}")
checkpoint(f"Number of bounces: {int(N//K)}")

################################################ SIMULATION ##########################################################

X, E, ESS, B_squared = MCHMC.MCHMC_bounces(d, N, L_opt, epsilon_opt, funct.standard_cauchy, integ.leapfrog, debug=debug)

#plotting the results
vis.plot_cauchy(X)

