from pathlib import Path
import sacred
from sacred.observers import FileStorageObserver
from simulate_basicsim_simulation_logic import do_one_simulation
import lateral_signaling as lsig

lsig.set_growth_params()

# Growth parameter(s)
_rho_max = float(lsig.mle_params.rho_max_ratio)

# Set up Sacred experiment
ex = sacred.Experiment("lateral_signaling_basic")
sacred_storage_dir = Path(
    "./sacred"  # Store locally
    # "/home/pbhamidi/scratch/lateral_signaling/sacred"  # Caltech HPC scratch (fast read-write)
)
sacred_storage_dir.mkdir(exist_ok=True)
ex.observers.append(FileStorageObserver(sacred_storage_dir))

# Set default simulation parameters
params_json = lsig.simulation_dir.joinpath("sim_parameters.json")
ex.add_config(str(params_json.resolve()))
ex.add_config(rho_max=_rho_max)


@ex.automain
def run_one_simulation(_config, _run, seed):
    """Run simulation with a single parameter configuration"""

    # Any file running this should be added as a source
    import sys

    ex.add_source_file(sys.argv[0])

    do_one_simulation(save=True, ex=ex, **_config)
