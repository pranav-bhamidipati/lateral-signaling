import os
from lsig_basicsim_run_one import ex # We import the experiment here. It knows what are the default values, how to run the code, and where to store its results
import numpy as np
import pandas as pd

## NOTE: This run uses the default parameter set (so not necessary to supply config options)

# Lattice size
# rows = cols = 50

# Make matrix of all combinations of params
# param_space = np.meshgrid(
#     delta_space, 
# )
# param_space = np.array(param_space).T.reshape(-1, len(param_space))

config_updates = { # Update the default variables (all others are still the same)
    "tmax_days": 7.,
    # "rows": rows,
    # "cols": cols,
    # "animate": True,
    # "n_frames": 51,
}
ex.run(config_updates=config_updates)  # Run with the updated parameters
    