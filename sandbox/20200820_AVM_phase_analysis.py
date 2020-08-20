import sys
import os

vor_path = "/home/ubuntu/git/active_vertex"
# vor_path = 'C:\\Users\\Pranav\\git\\active_vertex'
sys.path.append(vor_path)

import voronoi_model.voronoi_model_periodic as avm
from lattice_oop import *
import numpy as np
import tqdm
import datetime

########################




f = 200
t0 = 0
tmax = 4
dt = 0.05
n_t = int((tmax - t0) * f / dt) + 1  # calculates the n_t to get the desired dt

a = 0.4
k = 2

to_dir = "2020-08-19_avm_phase_sims/"



for fname, arr in results:
    np.save(fname, arr, allow_pickle=False)
