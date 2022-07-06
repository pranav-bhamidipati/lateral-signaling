import os
from lsig_wholewell_run_one import ex
import numpy as np
import pandas as pd

# Read in growth parameters
mle_data_dir    = os.path.abspath("../data/growth_curves_MLE")
mle_params_path = os.path.join(mle_data_dir, "growth_parameters_MLE__.csv")
mle_params_df   = pd.read_csv(mle_params_path, index_col=0)

save_frames = [round(i) for i in np.linspace(0, 801, 11)]

# Get MLE of carrying capacity
conds, gs, rho_maxs = mle_params_df.loc[:, ["treatment", "g_ratio", "rho_max_ratio"]].values.T

# Try different init densities
rho_0s = [1.0, 2.0, 4.0]

for cond, g, rho_max in zip(conds, gs, rho_maxs):
    for rho_0 in rho_0s:

        # Change default configuration
        config_updates = {
            "drug_condition": cond,
            "rho_0": rho_0,
            "g": g,
            "rho_max": rho_max,
            "save_frames": save_frames,
        }

        # Run with updated configuration
        ex.run(config_updates=config_updates)
    
