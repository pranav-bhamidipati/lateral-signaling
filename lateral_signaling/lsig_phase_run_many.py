import os
from lsig_phase_run_one import ex # We import the experiment here. It knows what are the default values, how to run the code, and where to store its results
import numpy as np
import pandas as pd

# Parameter values to scan
rep_space     = np.arange(1)
g_space       = np.linspace(0, 2.4, 5)[1:]
rho_0_space   = np.linspace(0, 6, 5)[1:]
rho_max_space = np.linspace(0, 6, 5)[1:]

# Make matrix of all combinations of params
param_space = np.meshgrid(
    rep_space, 
    g_space, 
    rho_0_space, 
    rho_max_space,
)
param_space = np.array(param_space).T.reshape(-1, len(param_space))

for _, g, rho_0, rho_max in param_space:  # Over what parameters do we loop
    config_updates = { # Update the default variables (all others are still the same)
        "g": g,
        "rho_0": rho_0,
        "rho_max": rho_max,
    }
    ex.run(config_updates=config_updates)  # Run with the updated parameters
