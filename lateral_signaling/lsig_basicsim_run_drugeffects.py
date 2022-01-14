import os
from lsig_basicsim_run_one import ex 
import numpy as np
import pandas as pd

# Add a drug_condition tag to the experiment
ex.add_config(dict(drug_condition=""))

# Read in MLE growth parameters
data_dir  = os.path.abspath("../data/MLE")
mle_fpath = os.path.join(data_dir, "growth_parameters_MLE.csv")
mle_df    = pd.read_csv(mle_fpath, index_col=0)

# Get drug conditions and intrinsic prolif rates
idx   = np.arange(mle_df.shape[0])
conds = mle_df.condition.values
gs    = mle_df.g_inv_days.values

# Initial densities
rho_0s = [1., 2., 4.]

# Make matrix of all combinations of params
param_space = np.meshgrid(
    idx, 
    rho_0s,
)
param_space = np.array(param_space).T.reshape(-1, len(param_space))

for i, rho_0 in param_space:
    
    # Get condition and intrinsic prolif rate
    cond = conds[int(i)]
    g    = gs[int(i)]

    # Update non-default variables and run
    config_updates = {
        "tmax_days": 8.,
        "g": g,
        "rho_0": rho_0,
        "drug_condition": cond,
    }
    ex.run(config_updates=config_updates)
    
