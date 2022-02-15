import os
from lsig_wholewell_run_one import ex
import numpy as np
import pandas as pd

# Read in growth parameters
mle_data_dir    = os.path.abspath("../data/MLE")
mle_params_path = os.path.join(mle_data_dir, "growth_parameters_MLE.csv")
mle_params_df   = pd.read_csv(mle_params_path, index_col                  = 0)

save_frames = [round(i) for i in np.linspace(0, 801, 11)]

# Get MLE of carrying capacity
conds, gs = mle_params_df.loc[:, ["condition", "g_ratio"]].values.T

for cond, g in zip(conds, gs):

    # Change default configuration
    config_updates = {
        "drug_condition": cond,
        "g": g,
        "save_frames": save_frames,
    }

    # Run with updated configuration
    ex.run(config_updates=config_updates)
    
