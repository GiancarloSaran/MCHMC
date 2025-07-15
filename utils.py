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
    L = -fn(x, **kwargs).sum()
    E = d * torch.log(w) + L
    return E


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