import os
from lsig_cisinhib_run_one import ex # We import the experiment here. It knows what are the default values, how to run the code, and where to store its results
import numpy as np
import pandas as pd

# Parameter sets to try
delta_space = np.linspace(0, 4, 9)

# Make matrix of all combinations of params
param_space = np.meshgrid(
    delta_space, 
)
param_space = np.array(param_space).T.reshape(-1, len(param_space))

for delta, *_ in param_space:  # Over what parameters do we loop
    config_updates = { # Update the default variables (all others are still the same)
        "delta": delta,
    }
    ex.run(config_updates=config_updates)  # Run with the updated parameters
