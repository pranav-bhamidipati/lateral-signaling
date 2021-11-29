import os
import dask
import dask.distributed

@dask.delayed
def run_one(config_updates):
    """Run single simulation - executed independently in every thread"""
    
    # Experiment should happen independently in each thread
    from lsig_phase_run_one import ex 

    ex.run(config_updates=config_updates)


# Below is only executed by the master node
if __name__ == "__main__":

    # Import dependencies
    import numpy as np

    # Parameter values to scan
    n_reps        = 5                            # Number of replicates per condition
    g_space       = np.linspace(0, 2.4, 25)[1:]  # Proliferation rate
    rho_0_space   = np.linspace(0,   6, 25)[1:]  # Initial density
    rho_max_space = np.linspace(0,   6, 25)[1:]  # Carrying capacity
    
    # Make matrix of all combinations of params
    param_space = np.asarray(np.meshgrid(
        g_space, 
        rho_0_space, 
        rho_max_space,
    ))
    param_space = param_space.T.reshape(-1, len(param_space))
#     n_runs = param_space.shape[0]
    
    # Decide how many workers to create in the Dask cluster
    #   In general, you can make n_workers = n_runs until n_runs gets
    #   so big that this causes issues (like having too many files open).
    #   At that point, you should try capping n_workers at the number of threads.
#     n_workers = n_runs                      # For smaller runs
    n_workers = int(os.environ["SLURM_NPROCS"])  # Number of available threads (on Slurm)

    # Memory for each worker
#     memory_limit = "auto"   # Default (change if memory errors)
#     memory_limit = "3 GiB"  # Custom
    mb_per_mem = int(int(os.environ["SLURM_MEM_PER_CPU"]) * 0.7)
    memory_limit = f"{mb_per_mem} MiB"  # For Slurm tasks 

    # Configure a Client that will spawn a local cluster of workers.
    #   Each task gets one worker and one worker gets one thread.
    #   Threads are allocated to workers as they become available
    client = dask.distributed.Client(
        threads_per_worker=1, 
        n_workers=n_workers,
        memory_limit=memory_limit,
    )

    # Make a list of tasks to execute (populated asynchronously)
    lazy_results = [] 
    for *_, g, rho_0, rho_max in param_space:
        config_updates = dict(
            n_reps  = n_reps,
            g       = float(g),
            rho_0   = float(rho_0),
            rho_max = float(rho_max),
        ) 
        lazy_results.append(run_one(config_updates)) 

    # Compute tasks lazily - tasks in list are assigned to workers
    #   on demand and the list is populated with results asynchronously.
    
    dask.compute(*lazy_results)


