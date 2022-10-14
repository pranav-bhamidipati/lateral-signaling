# import os
from simulate_basicsim_run_one import ex 
import numpy as np
# import pandas as pd
import lateral_signaling as lsig

gen_time = np.log(2)
tmax = 3 * gen_time
tmax_days = lsig.t_to_units(tmax)

# Perform parameter scan with constant density
rhos = [1., 2., 4.]
for rho in rhos:
    config_updates = { 
        "tmax_days": tmax_days,
        "rho_0": rho,
        "g": 0,
    }
    ex.run(config_updates=config_updates)  # Run with the updated parameters
    
