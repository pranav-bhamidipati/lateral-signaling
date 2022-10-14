from pathlib import Path
import dask
import dask.distributed


# Set dir for Dask to use (spill to disk, etc.)
local_dir = Path("/tmp/dask-worker-space")


@dask.delayed
def run_one(config_updates):
    """Run single simulation - executed independently in every thread"""

    # Experiment should happen independently in each thread
    from simulate_steadystate_run_one import ex

    ex.run(config_updates=config_updates)


def main(
    nrho=5000,
    scan_minrho=0.001,
    scan_maxrho=10.0,
    memory_allocation_percentage=0.85,
    threads_per_worker=1,
    memory_limit="auto",
    interface="lo",
    timeout=600,
    local_directory=local_dir,
):
    ## Below is only executed by the master node

    import os
    import numpy as np
    import psutil

    rho_scan = np.linspace(scan_minrho, scan_maxrho, nrho).tolist()
    n_runs = nrho

    # Set options based on whether this is being run in a SLURM environment or locally
    if slurm_ID := os.environ.get("SLURM_JOB_ID"):
        n_workers = os.environ["SLURM_NPROCS"]  # Number of available threads
        mb_per_cpu = int(
            int(os.environ["SLURM_MEM_PER_CPU"]) * memory_allocation_percentage
        )
        memory_limit = f"{mb_per_cpu} MiB"
        interface = "ib0"
    else:
        n_threads = psutil.cpu_count(logical=True)
        n_workers = min(n_threads, n_runs)

    # Configure a Client that will spawn a local cluster of workers.
    #   Each task gets one worker and one worker gets one thread.
    #   Threads are allocated to workers as they become available
    client = dask.distributed.Client(
        n_workers=n_workers,
        memory_limit=memory_limit,
        threads_per_worker=threads_per_worker,
        interface=interface,
        timeout=timeout,
        local_directory=local_directory,
    )

    print("Building list of tasks to execute asynchronously")
    lazy_results = []
    for rho in rho_scan:
        config_updates = {
            "rho_0": float(rho),
        }
        lazy_results.append(run_one(config_updates))

    print("Executing tasks...")
    dask.compute(*lazy_results)


if __name__ == "__main__":
    main()
