import os
from lsig_phase_run_one import ex # We import the experiment here. It knows what are the default values, how to run the code, and where to store its results
import numpy as np
import pandas as pd

# Number of replicates of each param set
n_reps = 2

# Parameter values to scan
g_space       = np.linspace(0, 2.4, 3)[1:]
rho_0_space   = np.linspace(0, 6.0, 3)[1:]
rho_max_space = np.linspace(0, 6.0, 3)[1:]

# Make matrix of all combinations of params
param_space = np.meshgrid(
    g_space, 
    rho_0_space, 
    rho_max_space,
)
param_space = np.array(param_space).T.reshape(-1, len(param_space))

for *_, g, rho_0, rho_max in param_space:  # Over what parameters do we loop
    config_updates = { # Update the default variables (all others are still the same)
        "n_reps": n_reps,
        "g": g,
        "rho_0": rho_0,
        "rho_max": rho_max,
    }
    ex.run(config_updates=config_updates)  # Run with the updated parameters
