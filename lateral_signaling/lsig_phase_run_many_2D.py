import os
import dask
import dask.distributed

# from lsig_phase_run_one import ex # We import the experiment here. It knows what are the default values, how to run the code, and where to store its results

@dask.delayed
def run_one(config_updates):
    """Run single simulation - executed independently in every thread"""
    
    # Experiment should happen independently in each thread
    from lsig_phase_run_one import ex 

    ex.run(config_updates=config_updates)


# Number of replicates of each param set
n_reps = 5

# Below is only executed by the master node
if __name__ == "__main__":

    # Import dependencies
    import numpy as np

    # Parameter values to scan
    g_space       = np.linspace(0, 2.4, 25)[1:].tolist()
    rho_0_space   = np.linspace(0, 6.0, 25)[1:]
    
    n_runs = rho_0_space.size
    
    # Decide how many workers to create in the Dask cluster
    #   In general, you can make n_workers = n_runs until n_runs gets
    #   so big that this causes issues (like having too many files open).
    #   At that point, you should try capping n_workers at the number of threads.
    n_workers = n_runs                      # For smaller runs
#    n_workers = int(os.environ["SLURM_NPROCS"])  # Number of available threads (on Slurm)

    # Memory for each worker
    memory_limit = "auto"   # Default (change if memory errors)
#    memory_limit = "3 GiB"  # Custom
#    mb_per_cpu = int(int(os.environ["SLURM_MEM_PER_CPU"]) * 0.7)
#    memory_limit = f"{mb_per_cpu} MiB"  # For Slurm tasks 
    
    # Specify interface (can run `ifconfig` to see what's available)
    interface="lo"   # Localhost (available on all machines)
    interface="ib0"  # Infiniband (faster, available on Caltech HPC Cluster)
   
    # Timeout time for Client
    timeout=600

    # Configure a Client that will spawn a local cluster of workers.
    #   Each task gets one worker and one worker gets one thread.
    #   Threads are allocated to workers as they become available
    client = dask.distributed.Client(
        threads_per_worker=1, 
        n_workers=n_workers,
        memory_limit=memory_limit,
        interface=interface,
        timeout=timeout,
        local_directory=local_dir,
    )

    # Make a list of tasks to execute (populated asynchronously)
    lazy_results = [] 
    print("building lazy results")
    for rho_0 in rho_0_space:
        config_updates = dict(
            n_reps  = n_reps,
            g_space = g_space,
            rho_0   = float(rho_0),
        ) 
        lazy_results.append(run_one(config_updates)) 
        
    print("built lazy results")

    # Compute tasks lazily - tasks in list are assigned to workers
    #   on demand and the list is populated with results asynchronously.
    
    dask.compute(*lazy_results)

