import torch
import numpy as np  
from blackjax.diagnostics import effective_sample_size
import blackjax
import jax
from jax import default_device
from tqdm import tqdm 
import itertools
import csv
from utils import checkpoint, warning, choose_device
import utils
import MCHMC
import MCLMC
import integration_schemes as integ
import functions as funct


def tune_eps(d, N, L, fn, algorithm, int_scheme, iterations=10, debug=False, **kwargs):

    eps_values = np.zeros(iterations)
    sigma_effs = np.zeros(iterations)
    target = np.zeros(iterations)
    
    checkpoint(f"\nRunning {iterations} iterations of {algorithm.__name__} with {N} steps, updating epsilon")

    epsilon = 0.5 # initial value
    
    for i in tqdm(range(iterations), desc="Running iterations"):
        
        if algorithm == MCHMC.MCHMC_bounces:
            X, E = MCHMC.MCHMC_bounces(d, N, L, epsilon, fn, int_scheme=int_scheme, debug=debug, **kwargs)
        else:
            X, E = MCLMC.MCLMC(d, N, L, epsilon, fn, int_scheme=int_scheme, debug=debug, **kwargs)
            
        varE = E.var()
        epsilon *= (0.0005 * d / varE)**(1/4)
        
        eps_values[i] = epsilon

        sigma_eff = torch.sqrt((X**2).var(axis=0).mean())
        sigma_effs[i] = sigma_eff
        target[i] = varE / d

    sigma_eff = np.mean(sigma_effs)
    eps_opt = np.mean(eps_values)
    
    return eps_opt, sigma_eff, target


def tune_L(sigma_eff, eps_opt, d, N, fn, algorithm, int_scheme, iterations=10, debug=False, **kwargs):

    L_values = np.zeros(iterations)
    checkpoint(f"\nRunning {iterations} iterations of {algorithm.__name__} with {N} steps, updating L")

    for i in tqdm(range(iterations), desc="Running iterations"):   
        L = sigma_eff * np.sqrt(d)
        if algorithm == MCHMC.MCHMC_bounces:
          X, *_ = MCHMC.MCHMC_bounces(d, N, L, eps_opt, fn, int_scheme=int_scheme, debug=debug, **kwargs)
        else:
          X, *_ = MCLMC.MCLMC(d, N, L, eps_opt, fn, int_scheme=int_scheme, debug=debug, **kwargs)
        
        #Using the library
        Xt = np.expand_dims(X, 0) #(chain_axis, sample_axis, dim_axis)

        ###################################################################### SOLUZIONE TEMPORANEA
        #n_eff_values = np.array(effective_sample_size(Xt)) # NON PIU COSì
        # Force data onto CPU
        Xt_cpu = jax.device_put(Xt, device=jax.devices("cpu")[0])
        # Evaluate on CPU
        with default_device(jax.devices("cpu")[0]):
            n_eff_values = jax.numpy.array(effective_sample_size(Xt_cpu))
        ######################################################################
        # Computing the dechoerence scale L
        n_eff_values = torch.tensor(n_eff_values, device=utils.choose_device())
        L = 0.4 * eps_opt * d * N / (n_eff_values.sum())
        L_values[i] = L
        
    L_opt = np.mean(L_values)
          
    return L_opt

def get_hyperparams(fn, d, L_init, output_csv, **kwargs):

    algorithms = [MCHMC.MCHMC_bounces, MCLMC.MCLMC]
    int_schemes = [integ.leapfrog, integ.minimal_norm]

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['algorithm', 'integ_scheme', 'eps', 'L'])
        
        # try all possible combinations
        for algorithm, int_scheme in itertools.product(algorithms, int_schemes):

            L_arg = torch.tensor(L_init, device=utils.choose_device()) if algorithm == MCLMC.MCLMC else L_init
            eps_opt, sigma_eff, _ = tune_eps(d=d, N=200, L=L_arg, fn=fn, algorithm=algorithm, 
                                                     int_scheme=int_scheme, iterations=5, **kwargs)
            
            eps_arg = torch.tensor(eps_opt, device=utils.choose_device()) if algorithm == MCLMC.MCLMC else eps_opt
            L_opt = tune_L(sigma_eff=sigma_eff, eps_opt=eps_arg, d=d, N=200, fn=fn, 
                                  algorithm=algorithm, int_scheme=int_scheme, iterations=5, **kwargs)

            writer.writerow([algorithm.__name__, int_scheme.__name__, eps_opt, L_opt])

'''
def blackjax_tuner(logdensity_fn, initial_position, key, desired_energy_variance= 5e-4, num_steps=350):
    """
    Tune (epsilon, L) using the autotuner from the Google blackjax package, for consistency checks.
    Inputs:
    - 
    Outputs:
    - 
    """
    init_key, tune_key, run_key = jax.random.split(key, 3)

    # create an initial state for the sampler
    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
    )

    # build the kernel
    kernel = lambda inverse_mass_matrix : blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    # find values for L and step_size
    (
        _,
        blackjax_mclmc_sampler_params,
        _
    ) = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        diagonal_preconditioning=False,
        desired_energy_var=desired_energy_variance
    )

    return blackjax_mclmc_sampler_params
'''