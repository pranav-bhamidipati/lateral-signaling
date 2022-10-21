import json
from pathlib import Path
import sacred
from sacred.observers import FileStorageObserver
from simulate_steadystate_simulation_logic import do_one_simulation
from lateral_signaling import mle_params, simulation_dir

# Set up Sacred experiment
ex = sacred.Experiment("lateral_signaling_steady_state")
sacred_storage_dir = Path(
    "./sacred"  # Store locally
    # "/home/pbhamidi/scratch/lateral_signaling/sacred"  # Caltech HPC scratch (fast read-write)
)
sacred_storage_dir.mkdir(exist_ok=True)
ex.observers.append(FileStorageObserver(str(sacred_storage_dir)))

# Set default simulation parameters, modified for the steady state case
default_params_json = simulation_dir.joinpath("sim_parameters.json")
steady_state_params_json = simulation_dir.joinpath("steadystate_parameters.json")

steady_state_config = json.load(default_params_json.open("r")).update(
    json.load(steady_state_params_json.open("r"))
)
ex.add_config(**steady_state_config)
ex.add_config(rho_max=float(mle_params.rho_max_ratio))


@ex.automain
def run_one_simulation(_config, _run, seed):
    """Run simulation with a single parameter configuration"""
    # Any file running this should be added as a source
    import sys

    ex.add_source_file(sys.argv[0])

    do_one_simulation(save=True, ex=ex, **_config)
