from simulate_basicsim_run_one import ex
import numpy as np
import pandas as pd

from lateral_signaling import analysis_dir

# Add a drug_condition tag to the experiment
ex.add_config(dict(drug_condition=""))

# Read in MLE growth parameters
# mle_fpath = analysis_dir.joinpath("growth_parameters_MLE.csv")
mle_fpath = analysis_dir.joinpath("240401_growth_parameters_MLE_fixed_rhomax.csv")
mle_df = pd.read_csv(mle_fpath, index_col=0)

# Get drug treatments and their intrinsic proliferation rates
idx = np.arange(mle_df.shape[0])
conds = mle_df["treatment"].values
gs = mle_df["g_ratio"].values

# Initial densities
rho_0s = [1.0, 2.0, 4.0]

# Make matrix of all combinations of params
param_space = np.meshgrid(
    idx,
    rho_0s,
)
param_space = np.array(param_space).T.reshape(-1, len(param_space))

for i, _rho_0 in param_space:

    # Get params to update
    cond = conds[int(i)]
    g = float(gs[int(i)])
    rho_0 = float(_rho_0)

    # Update non-default variables and run
    config_updates = {
        "g": g,
        "rho_0": rho_0,
        "drug_condition": cond,
    }
    ex.run(config_updates=config_updates)
