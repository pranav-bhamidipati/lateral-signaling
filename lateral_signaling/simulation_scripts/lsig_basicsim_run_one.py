import os
import json
import sacred
from sacred.observers import FileStorageObserver
import pandas as pd
from lsig_basicsim_simulation_logic import do_one_simulation

import lateral_signaling

# Set up Sacred experiment
ex = sacred.Experiment("lateral_signaling_examples")

# Set storage dir for all Sacred results. Could be made locally
#   on a local machine or elsewhere high-performance computing cluster
res_dir = "./sacred"  # Store locally
# res_dir = "/home/pbhamidi/scratch/lateral_signaling/sacred"  # Store in a fast read-write dir (scratch on Caltech HPC)

# Use this dir for storage
sacred_storage_dir = os.path.abspath(res_dir)
# os.makedirs(sacred_storage_dir)   # Make dir if it doesn't exist
ex.observers.append(FileStorageObserver(sacred_storage_dir))

# Get path to simulation parameters
data_dir = os.path.abspath("../data/simulations")
params_json_path = os.path.join(data_dir, "sim_parameters.json")

# Read from JSON file
with open(params_json_path, "r") as f:
    params = json.load(f)

# Unpack
_alpha = float(params["alpha"])
_k = float(params["k"])
_p = float(params["p"])
_delta = float(params["delta"])
_lambda_ = float(params["lambda_"])
_g = float(params["g"])
_rho_0 = float(params["rho_0"])
_delay = float(params["delay"])
_r_int = float(params["r_int"])
_gamma_R = float(params["gamma_R"])
_beta_args = tuple(
    [float(v) for k, v in sorted(params.items()) if k.startswith("beta_")]
)

# Read in growth parameters
mle_data_dir = os.path.abspath("../data/growth_curves_MLE")
mle_params_path = os.path.join(mle_data_dir, "growth_parameters_MLE.csv")
mle_params_df = pd.read_csv(mle_params_path, index_col=0)

# Get MLE of carrying capacity
_rho_max = mle_params_df.loc[
    mle_params_df.treatment == "untreated", ["rho_max_ratio"]
].values.ravel()[0]

# Variables here are handled magically by the provenance system
@ex.config
def cfg():
    tmax_days = 8
    nt_t = 500
    rows = 80
    cols = 80
    beta_function = "exponential"
    alpha = _alpha
    k = _k
    p = _p
    delta = _delta
    lambda_ = _lambda_
    delay = _delay
    g = _g
    rho_0 = _rho_0
    rho_max = _rho_max
    r_int = _r_int
    beta_args = _beta_args
    gamma_R = _gamma_R


@ex.main  # Use ex as our provenance system and call this function as __main__()
def run_one_simulation(_config, _run, seed):
    """Performs a simulation given a single parameter configuration"""
    # _config contains all the variables you define in cfg()
    # _run contains data about the run

    do_one_simulation(save=True, ex=ex, **_config)
