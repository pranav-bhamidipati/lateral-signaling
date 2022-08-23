from pathlib import Path
import sacred
from sacred.observers import FileStorageObserver
from lsig_phase_simulation_logic import do_one_simulation
from lateral_signaling import mle_params

# Growth parameter(s)
_rho_max = float(mle_params.rho_max_ratio)

# Set up Sacred experiment
ex = sacred.Experiment("lateral_signaling_phase")
sacred_storage_dir = Path(
    "./sacred"  # Store locally
    # "/home/pbhamidi/scratch/lateral_signaling/sacred"  # Caltech HPC scratch (fast read-write)
)
sacred_storage_dir.mkdir(exist_ok=True)
ex.observers.append(FileStorageObserver(sacred_storage_dir))

# Set default simulation parameters
data_dir = Path("../data/simulations")
params_json = data_dir.joinpath("sim_parameters.json")
ex.add_config(str(params_json.resolve()))
ex.add_config(
    rho_max=_rho_max,
    n_reps=5,
    nt_t_save=100,
)


@ex.automain
def run_one_simulation(_config, _run, seed):
    """Run simulation with a single parameter configuration"""

    # Any file running this should be added as a source
    import sys

    ex.add_source_file(sys.argv[0])

    do_one_simulation(save=True, ex=ex, **_config)
