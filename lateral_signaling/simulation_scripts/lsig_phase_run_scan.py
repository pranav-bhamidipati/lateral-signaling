import os
from typing import List, Literal, Optional, Tuple, Union
import dask
import dask.distributed

# Set dir for Dask to use (spill to disk, etc.)
# local_dir = os.path.abspath("/home/pbhamidi/scratch/lateral_signaling/dask-worker-space")
local_dir = os.path.abspath("/tmp/dask-worker-space")


@dask.delayed
def run_one(config_updates):
    """Run single simulation - executed independently in every thread"""

    # Experiment should happen independently in each thread
    from lsig_phase_run_one import ex

    ex.add_source_file(__file__)
    ex.run(config_updates=config_updates)


def main(
    start: Union[int, float],
    end: Union[int, float],
    size: int = 49,
    scale: Literal["log", "lin", "geom"] = "geom",
    g_space: Optional[List[float]] = None,
    **kwargs,
):

    import numpy as np

    if g_space is None:
        g_space = [1.0]

    if not isinstance(scale, str):
        raise TypeError("Argument `scale` must be a string.")
    elif scale == "log":
        space_fun = np.logspace
    elif scale == "lin":
        space_fun = np.linspace
    elif scale == "geom":
        space_fun = np.geomspace
    else:
        raise ValueError(
            f"Argument invalid: scale={scale}. Allowed values are: log, lin. "
        )
    rho_space = space_fun(start, end, size)

    ## How many threads
    n_workers = 8
    # n_workers = n_runs  # For smaller runs
    # n_workers = psutil.cpu_count(logical=True)  # Available threads on local machine
    # n_workers = int(os.environ["SLURM_NPROCS"])  # Available threads on Slurm

    ## Memory for each worker
    # memory_limit = "auto"  # Default (change if memory errors)
    memory_limit = "5 GiB"  # Custom
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
            g_space=g_space,
            rho_0=float(rho_0),
        )
        lazy_results.append(run_one(config_updates))

    dask.compute(*lazy_results)


# Below is only executed by the master node
if __name__ == "__main__":

    from lateral_signaling import mle_params
    import numpy as np

    main(
        start=0.01,
        end=mle_params.rho_max_ratio,
        size=25,
        scale="geom",
        g_space=np.linspace(0.1, 2.5, 25).tolist(),
        tmax_days=12.0,
        progress=True,
    )
