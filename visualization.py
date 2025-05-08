import numpy as np
import matplotlib.pyplot as plt
import torch

import functions as funct

def plot_cauchy(X:torch.Tensor):

    X = X.flatten().to("cpu").detach().numpy()

    x = np.linspace(-20, 20, 10000)
    
    plt.hist(X, density=True, alpha=0.5, bins=50, color='blue', label='MCHMC')

    plt.scatter(x, funct.standard_cauchy(x).numpy(), color='orange', s=1, label='exact')

    plt.legend()
    plt.show()