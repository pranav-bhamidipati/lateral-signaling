import os
from lsig_basicsim_run_one import ex 
import numpy as np
import pandas as pd


# Densities to scan
rhos = [1., 2., 4.]

# Perform parameter scan with constant density
for rho in rhos:
    config_updates = { 
        "tmax_days": 7.,
        "rho_0": rho,
        "g": 0,
    }
    ex.run(config_updates=config_updates)  # Run with the updated parameters
    
