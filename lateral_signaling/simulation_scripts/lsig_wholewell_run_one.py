import os
import sacred
from sacred.observers import FileStorageObserver
from lsig_wholewell_simulation_logic import do_one_simulation

# Set up Sacred experiment
ex = sacred.Experiment("lateral_signaling_whole_well")

# Set storage dir for all Sacred results
res_dir = "./sacred"  # Store locally
# res_dir = "/home/pbhamidi/scratch/lateral_signaling/sacred"  # On Caltech HPC, store in scratch (fast read-write)

# Use this dir for storage
sacred_storage_dir = os.path.abspath(res_dir)
# os.makedirs(sacred_storage_dir)   # Make dir if it doesn't exist
ex.observers.append(FileStorageObserver(sacred_storage_dir))

# Get path to simulation parameters
data_dir = os.path.abspath("../data/simulations")
params_json_path = os.path.join(data_dir, "sim_parameters_wholewell.json")

# Read in growth parameters
from lateral_signaling import mle_params

_rho_max = float(mle_params.rho_max_ratio)

# mle_data_dir = os.path.abspath("../data/growth_curves_MLE")
# mle_params_path = os.path.join(mle_data_dir, "growth_parameters_MLE.csv")
# mle_params_df = pd.read_csv(mle_params_path, index_col=0)
# _rho_max = mle_params_df.loc[
#     mle_params_df.treatment == "untreated", ["rho_max_ratio"]
# ].values.ravel()[0]
# _rho_max = float(_rho_max)

# Set default experimental configuration
ex.add_config(params_json_path)
ex.add_config(rho_max=_rho_max)


@ex.main  # Use ex as our provenance system and call this function as __main__()
def run_one_simulation(_config, _run, seed):
    """Run simulation with a single parameter configuration"""
    # Any file running this should be added as a source
    import sys

    ex.add_source_file(sys.argv[0])

    do_one_simulation(save=True, ex=ex, **_config)
