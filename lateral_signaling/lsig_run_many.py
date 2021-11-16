from lsig_run_one import ex # We import the experiment here. It knows what are the default values, how to run the code, and where to store it's results
import numpy as np
import pandas as pd

# Read in growth parameters
mle_data_dir = os.path.abspath("../../data/sim_data")
mle_params_fname = "growth_parameters_MLE.csv"
mle_params_path = os.path.join(data_dir, mle_params_fname)
mle_params_df = pd.read_csv(mle_params_file, index_col=0)

# Get carrying capacity of untreated condition (in dimensionless units)
rho_max = mle_params_df.loc[
    mle_params_df.condition == "untreated", ["rho_max_ratio"]i
].values.ravel()

# Parameter values to scan
rep_space     = np.arange(5)
g_space       = np.linspace(0, 2.4, 5)[1:]
rho_0_space   = np.linspace(0, 6, 5)[1:]
# rho_max_space = np.linspace(0, 6, 5)[1:]

# Make matrix of all combinations of params
param_space = np.meshgrid(
    rep_space, 
    g_space, 
    rho_0_space, 
#     rho_max_space,
)
param_space = np.array(param_space).T.reshape(-1, len(free_params))

for rep, g, rho_0 in param_space:  # Over what parameters do we loop
    config_updates = { # Update the default variables (all others are still the same)
        "rep": rep,
        "g": g,
        "rho_0": rho_0,
        "rho_max": rho_max,
    }
    ex.run(config_updates=config_updates)  # Run with the updated parameters
