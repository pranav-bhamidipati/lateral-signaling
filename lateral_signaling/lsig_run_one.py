import os
import json
import sacred
from sacred.observers import FileStorageObserver
from lsig_simulation_logic import do_one_simulation

# Set up Sacred experiment
ex = sacred.Experiment("lateral_signaling")
ex.observers.append(FileStorageObserver("./sacred"))  # Dir for storing results

# Get path to simulation parameters
data_dir  = os.path.abspath("../data/sim_data")
params_json_path = os.path.join(data_dir, "sim_parameters.json")

# Read from JSON file
with open(params_json_path, "r") as f:
    params = json.load(f)

# Unpack
_alpha     = params["alpha"]
_k         = params["k"]
_p         = params["p"]
_delta     = params["delta"]
_lambda_   = params["lambda_"]
_g         = params["g"]
_rho_0     = params["rho_0"]
_delay     = params["delay"]
_r_int     = params["r_int"]
_gamma_R   = params["gamma_R"]
_beta_args = tuple([params[k] for k in params.keys() if k.startswith("beta_")])

# Variables here are handled magically by the provenance system
@ex.config
def cfg():
    tmax_days = 8 
    nt_t      = 250
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
    rho_max   = 5.6
    r_int     = _r_int
    beta_args = _beta_args
    gamma_R   = _gamma_R


@ex.automain  # Tells python to use ex as our provenance system and call this function as the main function
def run_one_simulation(_config, _run, seed):
    """Simulates SPV given a single parameter configuration"""
    # _config contains all the variables you define in cfg()
    # _run contains data about the run
    do_one_simulation(
        seed      = seed,
        tmax_days = _config["tmax_days"],
        nt_t      = _config["nt_t"],
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
