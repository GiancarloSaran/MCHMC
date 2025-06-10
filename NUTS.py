import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist_npr
from jax import random
import blackjax
import random as pyrandom
import numpy as np
import matplotlib.pyplot as plt

def utility_function(d, w1, sep):
    weights = jnp.array([w1, 1 - w1])
    component_dist = dist_npr.Categorical(probs=weights)

    mu1 = jnp.zeros(d)
    mu2 = jnp.zeros(d).at[0].set(sep)

    cov = jnp.eye(d)

    means = jnp.stack([mu1, mu2])
    covs = jnp.stack([cov, cov])

    return component_dist, means, covs



def sample_bimodal_init_positions(num_chains, d, w1, sep, rng_key, standard_gaussian_mode):

    if not standard_gaussian_mode:
        print("\nMODALITY IN USE: *two modes*")
        weights = jnp.array([w1, 1 - w1])
        component_dist = dist_npr.Categorical(probs=weights)

        mu1 = jnp.zeros(d)
        mu2 = jnp.zeros(d).at[0].set(sep)

        cov = jnp.eye(d)

        means = jnp.stack([mu1, mu2])
        covs = jnp.stack([cov, cov])

        rng_keys = jax.random.split(rng_key, num_chains * 2) #to get new values if you use multiple times the function jax.random, you must split the key: It splits one key into two new, independent keys.
        init_positions = []
        modes = []
        for i in range(num_chains):

            comp_key = rng_keys[2 * i]
            sample_key = rng_keys[2 * i + 1]

            #sample the mode to consider
            comp = component_dist.sample(key=comp_key)
            mu = means[comp]
            cov = covs[comp]
            modes.append(int(comp))

            # SAMPLE FROM THE GAUSSIAN ASSOCIATED WITH THAT MODE
            rng_key, subkey = jax.random.split(rng_key) # another key
            sample = dist_npr.MultivariateNormal(mu, cov).sample(key=sample_key)
            init_positions.append(sample)

        return jnp.stack(init_positions), modes

    else:
        print("\nMODALITY IN USE: *standard gaussian*")
        keys = jax.random.split(rng_key, num_chains)

        init_positions = [dist_npr.MultivariateNormal(loc=jnp.zeros(d), covariance_matrix=jnp.eye(d)).sample(key=k) for k in keys]

        return jnp.stack(init_positions), None



def bimodal_model(d, w1, sep):
    component_dist, means, covs = utility_function(d, w1, sep)

    mix = dist_npr.MixtureSameFamily(
        component_dist,
        dist_npr.MultivariateNormal(loc=means, covariance_matrix=covs)
    )

    x = numpyro.sample("x", mix)
    return x