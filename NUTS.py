import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def run_nuts_numpyro(log_prob_fn, init_params, num_warmup=1000, num_samples=1000, num_chains=4):
    def model():
        # Define your model here
        params = numpyro.sample('params', dist.MultivariateNormal(
            loc=jnp.zeros(len(init_params)), 
            covariance_matrix=jnp.eye(len(init_params))
        ))
        # Use your log_prob_fn as the likelihood
        numpyro.factor('custom_logp', log_prob_fn(params))
    
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(jax.random.PRNGKey(0))
    return mcmc.get_samples()