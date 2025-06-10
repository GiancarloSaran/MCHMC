import torch
import sys
import numpy as np  
import statsmodels.api as sm
import blackjax.diagnostics as bxd
from blackjax.diagnostics import effective_sample_size
from tqdm import tqdm


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
        if not take_minimum:
            n_eff = np.array(effective_sample_size(Xt)).mean()
        else:
            n_eff = np.array(effective_sample_size(Xt)).min()
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