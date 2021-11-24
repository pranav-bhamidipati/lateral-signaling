import os
import json
import sacred
from sacred.observers import FileStorageObserver
import pandas as pd
from lsig_phase_simulation_logic import do_one_simulation

# Set up Sacred experiment
ex = sacred.Experiment("lateral_signaling_phase")

# Set results storage dir. Varies depending on if you're working
#   on a local machine or high-performance computing cluster
# res_dir = "./sacred"                            # Store locally
res_dir = "~/scratch/lateral_signaling/sacred"  # Store in scratch (compute cluster)

# Use this dir for storage
sacred_storage_dir = os.path.abspath(res_dir)
# os.makedirs(sacred_storage_dir)   # Make dir if it doesn't exist
ex.observers.append(
    FileStorageObserver(sacred_storage_dir)
)

# Get path to simulation parameters
data_dir  = os.path.abspath("../data/sim_data")
params_json_path = os.path.join(data_dir, "sim_parameters.json")

# Read from JSON file
with open(params_json_path, "r") as f:
    params = json.load(f)

# Unpack
_alpha     = float(params["alpha"])
_k         = float(params["k"])
_p         = float(params["p"])
_delta     = float(params["delta"])
_lambda_   = float(params["lambda_"])
_g         = float(params["g"])
_rho_0     = float(params["rho_0"])
_delay     = float(params["delay"])
_r_int     = float(params["r_int"])
_gamma_R   = float(params["gamma_R"])
_beta_args = tuple([float(params[k]) for k in params.keys() if k.startswith("beta_")])

# Read in growth parameters
mle_data_dir = os.path.abspath("../data")
mle_params_path = os.path.join(mle_data_dir, "growth_parameters_MLE.csv")
mle_params_df = pd.read_csv(mle_params_path, index_col=0)

# Get MLE of carrying capacity
_rho_max = mle_params_df.loc[
    mle_params_df.condition == "untreated", ["rho_max_ratio"]
].values.ravel()[0]

# Variables here are handled magically by the provenance system
@ex.config
def cfg():
    tmax_days = 8 
    nt_t      = 500
    nt_t_save = 100
    rows      = 80
    cols      = 80
    alpha     = _alpha
    k         = _k
    p         = _p
    delta     = _delta
    lambda_   = _lambda_
    delay     = _delay
    # rep       = -1
    g         = _g
    rho_0     = _rho_0
    rho_max   = _rho_max
    r_int     = _r_int
    beta_args = _beta_args
    gamma_R   = _gamma_R


@ex.main  # Use ex as our provenance system and call this function as __main__()
def run_one_simulation(_config, _run, seed):
    """Simulates SPV given a single parameter configuration"""
    # _config contains all the variables you define in cfg()
    # _run contains data about the run
    do_one_simulation(
        tmax_days = _config["tmax_days"],
        nt_t      = _config["nt_t"],
        nt_t_save = _config["nt_t_save"],
        rows      = _config["rows"],
        cols      = _config["cols"],
        r_int     = _config["r_int"],
        alpha     = _config["alpha"],
        k         = _config["k"],
        p         = _config["p"],
        delta     = _config["delta"],
        lambda_   = _config["lambda_"],
        delay     = _config["delay"],
     #    rep       = rep,
        g         = _config["g"],
        rho_0     = _config["rho_0"],
        rho_max   = _config["rho_max"],
        beta_args = _config["beta_args"],
        gamma_R   = _config["gamma_R"],
        save      = True,
        ex        = ex,  ## Pass over the experiment handler ex
    )
