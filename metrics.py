import torch
import numpy as np  
import statsmodels.api as sm
import blackjax.diagnostics as bxd
from blackjax.diagnostics import effective_sample_size
from tqdm import tqdm
import jax
from jax import default_device
import functions as funct

def ESS(X, threshold=0.1, grad_evaluations=2, pbar=True, take_minimum=False):
   
    X = X.to("cpu").detach().numpy()

    # binary search
    low, high = 100, len(X)
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
            n_eff = n_eff_values.mean()
        else:
            n_eff = n_eff_values.min()

        b_2 = np.sqrt(2/n_eff)
        
        if threshold - 0.03 <= b_2 <= threshold + 0.03:
            best_n = mid
            return 200 / (best_n * grad_evaluations)
            
        elif b_2 < threshold - 0.01:
            high = mid - 1
        else:
            low = mid + 1
          
    return None

def ESS_truth(X, fn, threshold=0.1, grad_evaluations=2, **kwargs):
    
    X = X.to("cpu").detach().numpy()
    X2 = X**2
    
    #binary search
    low, high = 100, len(X) 
    num_iterations = int(np.ceil(np.log2(high - low + 1)))
    
    for _ in range(num_iterations):
        if low > high:
            break
        mid = (low + high) // 2

        z2 = []
    
        for i in range(X.shape[1]):
            
            if fn == funct.bimodal:
                if i==0:
                    E_t = 13.8 # dimension 0
                    E_s = np.mean(X2[:mid,i])
                    z2.append(((E_s - E_t) / E_t)**2)
                else: 
                    E_t = 1.0 # other dimensions
                    E_s = np.mean(X2[:mid,i])
                    z2.append(((E_s - E_t) / E_t)**2)
                    
            elif fn == funct.ill_cond_gaussian:
                cov = kwargs.get("cov", None)
                E_t = cov[i,i] 
                E_s = np.mean(X2[:mid,i])
                z2.append(((E_s - E_t) / E_t)**2)
        
        z2_mean = np.mean(z2)
        b_2 = np.sqrt(z2_mean)
            
        if threshold - 0.03 <= b_2 <= threshold + 0.03:
            best_n = mid
            return 200 / (best_n * grad_evaluations)

        #update boundaries 
        elif b_2 < threshold - 0.01:
            high = mid - 1
        else:
            low = mid + 1

    return None

# metodo alternativo, esce uguale
'''
def ESS_truth(X, fn, threshold=0.1, grad_evaluations=2, **kwargs):
    
    X = X.to("cpu").detach().numpy()
    X2 = X[100:]**2
    
    tol = 0.05
    
    if fn == funct.bimodal:
        f = np.array([[13.8]+[1]*99]) #ground truth value of 2nd momentum
    elif fn == funct.ill_cond_gaussian:
        f = np.diag(cov)
    
    z_i = ((X2).cumsum(axis=0)/(np.arange(len(X2))[:,None]+1) / f - 1)
    b_2 = (z_i**2).mean(axis=1)**0.5
    n_values = np.argwhere(np.logical_and((b_2 < threshold+tol), (threshold-tol < b_2)))
    
    if n_values.size == 0:
        return None
        
    n = n_values.min()
    return 200 / (n * grad_evaluations)
'''