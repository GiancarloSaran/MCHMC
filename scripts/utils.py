import torch
import sys
import numpy as np  


def checkpoint(message = "", debug=True):
    if debug: print(message)


def warning(message = ""):
    print("WARNING: "+ message)


def choose_device():
  
  if torch.cuda.is_available():
      device = torch.device("cuda")
      #print('GPU available')

  else:
      device = torch.device("cpu")
      #print('GPU not available')

  return device

def energy(x, w, d, fn):
    """
    This function computes the energy of the system
    """
    L = -torch.log(fn(x)).sum() # Negative log posterior
    E = d * torch.log(w) + L
    return E


def effective_sample_size(X, d, cauchy=False, debug=False, autotune_mode=False):
    """
    This function computes the ESS used to evaluate the sampling algorithm used. In this
    case we are considering the MCHMC with q=0.

    NOTE:
    -The cauchy distribution case is special. We do not compute b_2 but we compute
    a coefficient b_L, function of the coordinate and momentum of a given
    simulation step. This should decrease during evolution. As it reaches a
    threshold value computed from theory, the simulation is considered converged
    and the ESS can be computed.

    -d should be very large for the validity of the approximations from which the
    formula are derived
    -----------------------------------------------------------------------------
    Args:
        X: vector of points sampled with MCHMC
        d: dimension of the problem
        cauchy: if True, the ESS is caluculated for the cauchy distribution
        autotune_mode: if True, it returns a list with the n_eff^{i} values,
                    needed to tune the step length
    """

    convergence_reached = False
    if cauchy:
        def cauchy_1D(x_i):
            return 1/(np.pi*(1+x_i**2))

        # Retrieving the number of samples simulated
        n_samples = X.shape[0]

        # Computing and storing the expectation values of L over each dimension
        device = choose_device()
        E_sampler_values = torch.zeros(d, device=device)
        for dim in range(d):
            x_i = X[:,dim] # sampled points over a single dimension
            C_i = cauchy_1D(x_i)
            E_sampler_i = torch.mean(-torch.log(C_i))
            E_sampler_values[dim] = E_sampler_i

        # Constants
        E_truth = torch.tensor(  np.log(4*np.pi)  , device=device)  # from paper
        const =   torch.tensor( (np.pi**2) / 3    , device=device) # a generic constant

        if autotune_mode: # compute the n_eff_i values

            n_eff_i_values = const / (E_sampler_values-E_truth)**2 # CHECK: if it is okay
            #this should be a vector with all the n_{eff}^{i} values

            return n_eff_i_values


        else: # compute the ESS

            threshold = 0.0165 # from paper

            # Computing the value of b_L and checking for convergence
            b_squared = (1/d)*torch.sum((E_sampler_values-E_truth)**2)
            b = torch.sqrt(b_squared)

            # Computing the Effective Sample Size
            n_eff = ( const ) / b_squared
            ESS = n_eff / n_samples
            #checkpoint(f"\tb_L: {np.sqrt(b_L_squared)}, ESS: {ESS}", debug)

            if  b < threshold:
                convergence_reached = True

            return ESS, b_squared

    else:
        sys.exit("Distribution not implemented yet")