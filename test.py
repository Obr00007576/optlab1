import numpy as np
from matplotlib import pyplot as plt
import scipy
import time

import optimization
import oracles

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

import oracles
import numpy as np
import random
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from inspect import getfullargspec
from itertools import product
from optimization import *
from plot_trajectory_2d import plot_levels, plot_trajectory
import matplotlib.pyplot as plt
from tqdm import tqdm

random.seed(45)
np.random.seed(46)
n_dim = 20
data_num = 10000
A = np.array([np.random.rand(n_dim)*10-5 for _ in range(data_num)])
b = np.array([np.random.rand()*30-15 for _ in range(data_num)])
regcoef = 1.0 / data_num

oracle = oracles.create_log_reg_oracle(A, b, regcoef)
