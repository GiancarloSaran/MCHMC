import torch
import numpy as np  
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from utils import checkpoint, warning
import utils
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal
from scipy.stats import ortho_group

np.random.seed(0)

def standard_cauchy(x):
    """
    This function takes an array of d-dimensional points as input
        x = [x_1, x_2, ..., x_d]

    For each point x_i, it computes the value of the cauchy distribution as:
        p(x_i) = ( 1/(1+x_{i,1}**2) )*( 1/(1+x_{i,2}**2) )*...*( 1/(1+x_{i,d}**2) )

    The output is a tensor containing the values of the distribution for each point.
    """
    # serve che sia tensore di torch per implementazione del gradiente
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if x.dim() == 1:
        x = x.unsqueeze(1)
    return torch.prod(1 / (np.pi * (1 + x**2)), axis=1)

def bimodal(x, w1=torch.tensor(0.8), sep=8):
    '''
    This function takes an array of d-dimensional points as input
        x = [x_1, x_2, ..., x_d]

    For each point x_i, it computes the value of the bimodal distribution as:
        p(x_i) = w_1 * p(x_i) + w_2 * p(x_i)
      where p(x_i) is a standard 50-dimensional normal distribution.

    The output is a tensor containing the values of the distribution for each point.
    '''
    device = utils.choose_device()
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if x.dim() == 1:
        x = x.unsqueeze(1)
    x = x.to(device)
    x = torch.transpose(x, 0, 1)
    d = x.shape[1]
    mu1 = torch.zeros(d, device=device)
    mu2 = torch.zeros(d, device=device) 
    mu2[0] = sep
    log_pdf1 = MultivariateNormal(mu1, torch.eye(d, device=device)).log_prob(x)
    log_pdf2 = MultivariateNormal(mu2, torch.eye(d, device=device)).log_prob(x)
    log_pdf = torch.logaddexp(torch.log(w1) + log_pdf1, torch.log(1-w1) + log_pdf2)

    return log_pdf

def get_ill_cov(d, k=100):
    eig = torch.logspace(-np.log(np.sqrt(k)), np.log(np.sqrt(k)), d, base=np.e)
    O = np.diag(eig).astype(float)
    R = ortho_group.rvs(d).astype(float)
    RO = np.matmul(R,O)
    cov = np.matmul(RO, R.T)
    return cov

def ill_cond_gaussian(x, k=100, **kwargs):
    device = utils.choose_device()
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if x.dim() == 1:
        x = x.unsqueeze(1)
    x = x.to(device)
    x = torch.transpose(x, 0, 1)
    d = x.shape[1]
    mu = torch.zeros(d, device=device)
    cov = kwargs.get('cov')
    cov = torch.tensor(cov, device = device)
    log_pdf = MultivariateNormal(mu, cov).log_prob(x) #returns log pdf, not the actual pdf
    return log_pdf

    
def rosenbrock(x, Q=0.1):
    device = utils.choose_device()
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if x.dim() == 1:
        x = x.unsqueeze(1)
    x = x.to(device)
    x = torch.transpose(x, 0, 1)
    d = x.shape[1]
    if d % 2 != 0:
        print(f'Alert!! Rosenbrock allows even dimensionality only')
        return None
    idx_even = torch.arange(0, d, step=2)
    idx_odd = torch.arange(1, d, step=2)
    X = x[:, idx_even]  # Shape: (1, d/2)
    Y = x[:, idx_odd]   # Shape: (1, d/2)
    dist1 = Normal(1, 1)  # N(x|1,1)
    dist2 = Normal(0, np.sqrt(Q))  # N(y-x^2|0, sqrt(Q))
    log_prob_x = dist1.log_prob(X)
    log_prob_y = dist2.log_prob(Y - X**2)
    log_pdf = torch.sum(log_prob_x + log_prob_y)
    return log_pdf

def neals_funnel(x):
    #Same as rosenbrock, it's sum of Gaussian logls
    device = utils.choose_device()
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if x.dim() == 1:
        x = x.unsqueeze(1)
    x = x.to(device)
    x = torch.transpose(x, 0, 1)
    d = x.shape[1]
    if d < 2:
        print(f'Alert!! No hidden variables')
        return None
    theta = x[0, 0]
    z = x[:, 1:]
    dist_theta = Normal(0, 3)
    dist_z = Normal(0, torch.exp(theta/2))
    log_prob_theta = dist_theta.log_prob(theta)
    log_prob_z = dist_z.log_prob(z)
    return log_prob_theta + log_prob_z.sum()

'''
def jax_bimodal(x, w1=0.8, off=8):
    x = jnp.asarray(x, dtype=jnp.float32)
    if x.ndim == 1:
        x = x[None, :]
    
    d = x.shape[1]
    
    mu1 = jnp.zeros(d)
    mu2 = jnp.zeros(d).at[0].set(off)
    cov = jnp.eye(d)
    
    pdf1 = multivariate_normal.pdf(x, mu1, cov)
    pdf2 = multivariate_normal.pdf(x, mu2, cov)
    
    pdf = w1 * pdf1 + (1 - w1) * pdf2
    return pdf
'''