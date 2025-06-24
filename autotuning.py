import torch
import numpy as np  
from utils import checkpoint, warning, choose_device
import utils
import MCHMC
import MCLMC
from blackjax.diagnostics import effective_sample_size
import blackjax
import jax
from jax import default_device
from tqdm import tqdm 

def sigma_eff(d, N, L, fn, algorithm, int_scheme, debug=False, **kwargs):
    epsilon = 0.5 # initial value
    
    if algorithm == MCHMC.MCHMC_bounces:
        X, E = MCHMC.MCHMC_bounces(d, N, L, epsilon, fn, int_scheme=int_scheme, debug=debug, **kwargs)
    else:
        X, E = MCLMC.MCLMC(d, N, L, epsilon, fn, debug=debug, **kwargs)

    sigma_eff = torch.sqrt((X**2).var(axis=0).mean())
    return sigma_eff


def tune_eps(d, N, L, fn, algorithm, int_scheme, iterations=10, debug=False, **kwargs):

    eps_values = np.zeros(iterations)
    sigma_effs = np.zeros(iterations)
    target = np.zeros(iterations)
    
    checkpoint(f"\nRunning {iterations} iterations of {algorithm} with {N} steps, updating epsilon")

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
    
    return eps_values, sigma_effs, target


def tune_L(sigma_eff, eps_opt, d, N, fn, algorithm, int_scheme, iterations=10, debug=False, cauchy=False, **kwargs):

    L_values = np.zeros(iterations)
    checkpoint(f"\nRunning {iterations} iterations of {algorithm} with {N} steps, updating L")

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
          
    return L_values


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