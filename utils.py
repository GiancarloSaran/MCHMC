import torch
import sys
import numpy as np  
import statsmodels.api as sm
import blackjax.diagnostics as bxd
from blackjax.diagnostics import effective_sample_size
from tqdm import tqdm
import jax
from jax import default_device


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

def energy(x, w, d, fn, **kwargs):
    #L = -torch.log(fn(x, **kwargs)).sum() # Negative log posterior
    L = -fn(x, **kwargs).sum()
    E = d * torch.log(w) + L
    return E

def find_ess(X, threshold=0.1, grad_evaluations=2, min_samples=100, pbar=True, take_minimum=False):
   
    X = X.to("cpu").detach().numpy()
    low, high = min_samples, len(X)

    num_iterations = int(np.ceil(np.log2(high - low + 1)))

    for _ in range(num_iterations):
        if low > high:
            break
        mid = (low + high) // 2
        
        Xt = np.expand_dims(X[:mid], 0)
        
        ##################################################################### TEMPORARY SOLUTION
        # Force data onto CPU
        Xt = jax.device_put(Xt, device=jax.devices("cpu")[0])
        # Evaluate on CPU
        with default_device(jax.devices("cpu")[0]):
            n_eff_values = jax.numpy.array(effective_sample_size(Xt))
        ######################################################################
        
        if not take_minimum:
            # Evaluate on CPU
            n_eff = n_eff_values.mean()
        else:
            n_eff = n_eff_values.min()

        b_2 = np.sqrt(2/n_eff)
        
        if threshold - 0.01 <= b_2 <= threshold + 0.01:
            best_n = mid
            return 200 / (best_n * grad_evaluations)
            
        elif b_2 < threshold - 0.01:
            high = mid - 1
        else:
            low = mid + 1
          
    return None

def ESS_bimodal(X, threshold=0.1, grad_evaluations=2, min_samples=100):
    
    X = X.to("cpu").detach().numpy()
    X2 = X**2

    #binary search
    low, high = min_samples, len(X) 
    num_iterations = int(np.ceil(np.log2(high - low + 1)))
    for _ in range(num_iterations):
        if low > high:
            break
        mid = (low + high) // 2

        z2 = []
    
        for i in range(X.shape[1]):
            if i==0:
                E_t = 13.8 # dimension 0
                E_s = np.mean(X2[:mid,i])
                z2.append(((E_s - E_t) / E_t)**2)
            else: 
                E_t = 1.0 # other dimensions
                E_s = np.mean(X2[:mid,i])
                z2.append(((E_s - E_t) / E_t)**2)
                
        z2_mean = np.mean(z2)
        b_2 = np.sqrt(z2_mean)
            
        if threshold - 0.05 <= b_2 <= threshold + 0.05:
            best_n = mid
            return 200 / (best_n * grad_evaluations)

        #update boundaries 
        elif b_2 < threshold - 0.01:
            high = mid - 1
        else:
            low = mid + 1
  
    return None #if no convergence

def ESS_gaussian(X, cov, threshold=0.1, grad_evaluations=2, min_samples=100):

    X = X.to("cpu").detach().numpy()
    X2 = X**2
    
    #binary search
    low, high = min_samples, len(X) 
    num_iterations = int(np.ceil(np.log2(high - low + 1)))
    for _ in range(num_iterations):
        if low > high:
            break
        mid = (low + high) // 2

        z2 = []
    
        for i in range(X.shape[1]):
                E_t = cov[i,i] 
                E_s = np.mean(X2[:mid,i])
                z2.append(((E_s - E_t) / E_t)**2)
                
        z2_mean = np.mean(z2)
        b_2 = np.sqrt(z2_mean)
            
        if threshold - 0.05 <= b_2 <= threshold + 0.05:
            best_n = mid
            return 200 / (best_n * grad_evaluations)

        #update boundaries 
        elif b_2 < threshold - 0.01:
            high = mid - 1
        else:
            low = mid + 1
  
    return None #if no convergence


def check_two_modes(samples, no_print=False):
    """
    Calcola la media dei samples sulla dimensione 0 e controlla che sia fra le due mode.
    Ritorna un booleano e eventualmente printa il risultato.
    """
    mean0 = samples[:,0].mean()
    if mean0 < 5 and mean0 > 3:
        two_modes=False
        if not no_print: print("(Probably) ONE mode sampled")
    else:
        two_modes=True
        if not no_print: print("(Probably) TWO mode sampled")
        
    return two_modes
    

'''
def find_ess(X, threshold=200, grad_evaluations=2, min_samples=100, pbar=True, take_minimum=False):
    
    X = X.to("cpu").detach().numpy()
    low, high = min_samples, len(X)

    num_iterations = int(np.ceil(np.log2(high - low + 1)))
    iterator = range(num_iterations)

    for _ in iterator:
        if low > high:
            break
        mid = (low + high) // 2
        
        Xt = np.expand_dims(X[:mid], 0)
        ##################################################################### TEMPORARY SOLUTION
        # Force data onto CPU
        Xt_cpu = jax.device_put(Xt, device=jax.devices("cpu")[0])
        # Evaluate on CPU
        with default_device(jax.devices("cpu")[0]):
            n_eff_values = jax.numpy.array(effective_sample_size(Xt_cpu))
        ######################################################################
        if not take_minimum:
            # Evaluate on CPU
            n_eff = n_eff_values.mean()
        else:
            n_eff = n_eff_values.min()
        #print(mid, n_eff)
        
        if threshold - 1 <= n_eff <= threshold + 1:
            best_n = mid
            high = mid - 1  # Try smaller n
            return threshold / (best_n * grad_evaluations)
            
        elif n_eff < threshold - 1:
            low = mid + 1
        else:
            high = mid - 1
            
    return None      
'''