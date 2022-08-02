from pathlib import Path
import sacred
from sacred.observers import FileStorageObserver
import pandas as pd
from lsig_steadystate_simulation_logic import do_one_simulation

# Set up Sacred experiment
ex = sacred.Experiment("lateral_signaling_steady_state")
sacred_storage_dir = Path(
    "./sacred"  # Store locally
    # "/home/pbhamidi/scratch/lateral_signaling/sacred"  # Caltech HPC scratch (fast read-write)
)
sacred_storage_dir.mkdir(exist_ok=True)
ex.observers.append(FileStorageObserver(sacred_storage_dir))

# Growth parameter(s) from experiments
mle_data_dir = Path("../data/growth_curves_MLE")
mle_params_path = mle_data_dir.joinpath("growth_parameters_MLE.csv")
mle_params_df = pd.read_csv(mle_params_path, index_col=0)
_rho_max, *_ = mle_params_df.loc[
    mle_params_df.treatment == "untreated", ["rho_max_ratio"]
].values.flat
_rho_max = float(_rho_max)

# Set default simulation parameters
data_dir = Path("../data/simulations")
params_json = data_dir.joinpath("sim_parameters_steadystate.json")
ex.add_config(str(params_json.resolve()))
ex.add_config(rho_max=_rho_max)


@ex.main  # Use ex as our provenance system and call this function as __main__()
def run_one_simulation(_config, _run, seed):
    """Run simulation with a single parameter configuration"""
    do_one_simulation(save=True, ex=ex, **_config)
