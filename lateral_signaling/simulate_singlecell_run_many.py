import os
from simulate_singlecell_run_one import ex
import numpy as np
import pandas as pd
import lateral_signaling as lsig

# Set time-span
tmax = 2.25
tmax_days = lsig.t_to_units(tmax)

# Change default configuration
config_updates = dict(
    tmax_days=tmax_days, nt_t=500
)

# Run with updated configuration
ex.run(config_updates=config_updates)
 
