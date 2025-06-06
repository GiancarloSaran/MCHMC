from argparse import ArgumentParser
import torch
from tqdm import tqdm
import numpy as np
import sys
import matplotlib.pyplot as plt
from utils import checkpoint, warning
from utils import effective_sample_size
import utils
import MCHMC, MCLMC
import autotuning as aut
import integration_schemes as integ
import functions as funct
import visualization as vis
import importlib
import pandas as pd
np.random.seed(0)

def main():
    """
    Autotune all the algorithms for distributions specified by the user. The results are stored in .csv format
    """
    functs = [funct.bimodal, funct.ill_cond_gaussian, funct.rosenbrock, funct.neals_funnel]
    dist_params = [[], [], [], []]
    parser = ArgumentParser()
    return 0

if __name__ == "__main__":
    main()