import psutil
from pathlib import Path
from typing import Literal, Union
import dask
import dask.distributed

# Set dir for Dask to use (spill to disk, etc.)
# local_dir = Path("/home/pbhamidi/scratch/lateral_signaling/dask-worker-space")
local_dir = Path("/tmp/dask-worker-space")


@dask.delayed
def run_one(config_updates):
    """Run single simulation - executed independently in every thread"""

    # Experiment should happen independently in each thread
    from lsig_phase_run_one import ex

    ex.run(config_updates=config_updates)


def main(
    start: Union[int, float],
    end: Union[int, float],
    size: int = 51,
    scale: Literal["log", "lin"] = "log",
    n_reps: int = 5,
):

    import numpy as np

    if not isinstance(scale, str):
        raise TypeError("Argument `scale` must be a string.")
    elif scale == "log":
        space_fun = np.geomspace
    elif scale == "lin":
        space_fun = np.linspace
    else:
        raise ValueError(
            f"Argument invalid: scale={scale}. Allowed values are: log, lin. "
        )
    rho_space = space_fun(start, end, size)

    # How many threads
    # n_workers = n_runs  # For smaller runs
    n_workers = psutil.cpu_count(logical=True)  # Available threads on local machine
    # n_workers = int(os.environ["SLURM_NPROCS"])  # Available threads on Slurm

    # Memory for each worker
    memory_limit = "auto"  # Default (change if memory errors)
    # memory_limit = "3 GiB"  # Custom
    # mb_per_cpu = int(int(os.environ["SLURM_MEM_PER_CPU"]) * 0.7)
    # memory_limit = f"{mb_per_cpu} MiB"  # For Slurm tasks

    client = dask.distributed.Client(
        threads_per_worker=1,
        n_workers=n_workers,
        memory_limit=memory_limit,
        # interface="ib0",
        timeout=600,
        local_directory=local_dir,
    )

    lazy_results = []
    for rho_0 in rho_space:
        config_updates = dict(
            n_reps=n_reps,
            g_space=[0.0],
            rho_0=float(rho_0),
        )
        lazy_results.append(run_one(config_updates))

    dask.compute(*lazy_results)


# Below is only executed by the master node
if __name__ == "__main__":
    main(
        start=0.01,
        end=1.0,
        size=51,
        scale="lin",
    )
