import os
from lsig_phase_run_one import ex # We import the experiment here. It knows what are the default values, how to run the code, and where to store its results
import numpy as np
import pandas as pd

# Number of replicates of each param set
n_reps = 5

# Parameter values to scan
g_space       = np.linspace(0, 2.4, 25)[1:].tolist()
rho_0_space   = np.linspace(0, 6.0, 25)[1:]

for rho_0 in rho_0_space:  # Over what parameters do we loop
    config_updates = { # Update the default variable s (all others are still the same)
        "n_reps": n_reps,
        "g_space": g_space,
        "rho_0": rho_0,
    }
    ex.run(config_updates=config_updates)  # Run with the updated parameters
