import json
from pathlib import Path
import sacred
from sacred.observers import FileStorageObserver
from simulate_wholewell_simulation_logic import do_one_simulation
import lateral_signaling as lsig

lsig.set_growth_params()

# Set up Sacred experiment
ex = sacred.Experiment("lateral_signaling_whole_well")

# Use this dir for storing results
sacred_storage_dir = Path(
    "./sacred"  # Store locally
    # "/home/pbhamidi/scratch/lateral_signaling/sacred"  # Caltech HPC scratch (fast read-write)
)
sacred_storage_dir.mkdir(exist_ok=True)
ex.observers.append(FileStorageObserver(str(sacred_storage_dir)))

# Get simulation parameters
default_params_json = lsig.simulation_dir.joinpath("sim_parameters.json")
whole_well_params_json = lsig.simulation_dir.joinpath("wholewell_parameters.json")

_rho_max = float(lsig.mle_params.rho_max_ratio)

# Set default experimental configuration, modified for the whole well case
whole_well_config = json.load(default_params_json.open("r"))
whole_well_config.update(json.load(whole_well_params_json.open("r")))
ex.add_config(**whole_well_config)
ex.add_config(rho_max=_rho_max)


@ex.main  # Use ex as our provenance system and call this function as __main__()
def run_one_simulation(_config, _run, seed):
    """Run simulation with a single parameter configuration"""
    # Any file running this should be added as a source
    import sys

    ex.add_source_file(sys.argv[0])

    do_one_simulation(save=True, ex=ex, **_config)
