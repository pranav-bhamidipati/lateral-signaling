import json
from pathlib import Path
from unittest import defaultTestLoader
import sacred
from sacred.observers import FileStorageObserver
from simulate_phase_simulation_logic import do_one_simulation
from lateral_signaling import mle_params, simulation_dir

# Set up Sacred experiment
ex = sacred.Experiment("lateral_signaling_phase")
sacred_storage_dir = Path(
    "./sacred"  # Store locally
    # "/home/pbhamidi/scratch/lateral_signaling/sacred"  # Caltech HPC scratch (fast read-write)
)
sacred_storage_dir.mkdir(exist_ok=True)
ex.observers.append(FileStorageObserver(sacred_storage_dir))

# Set default simulation parameters, modified for phase calculation
default_params_json = simulation_dir.joinpath("sim_parameters.json")
phase_params_json = simulation_dir.joinpath("phase_parameters.json")
phase_config = json.load(default_params_json.open("r"))
phase_config.update(json.load(phase_params_json).open("r"))
ex.add_config(**phase_config)
ex.add_config(rho_max=float(mle_params.rho_max_ratio))


@ex.automain
def run_one_simulation(_config, _run, seed):
    """Run simulation with a single parameter configuration"""

    # Any file running this should be added as a source
    import sys

    ex.add_source_file(sys.argv[0])

    do_one_simulation(save=True, ex=ex, **_config)
