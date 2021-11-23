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
    rep_space     = np.arange(1)                # Number of replicates per condition
    g_space       = np.linspace(0, 2.4, 3)[1:]  # Proliferation rate
    rho_0_space   = np.linspace(0,   6, 3)[1:]  # Initial density
    rho_max_space = np.linspace(0,   6, 3)[1:]  # Carrying capacity

    # Make matrix of all combinations of params
    param_space = np.asarray(np.meshgrid(
        rep_space, 
        g_space, 
        rho_0_space, 
        rho_max_space,
    ))
    param_space = param_space.T.reshape(-1, len(param_space))
    n_runs = param_space.shape[0]
    
    # Holds simulation results (populated asynchronously)
    lazy_results = []  
    
    # Configure a Client that will spawn a local cluster of workers.
    #   Each task gets one worker and one worker gets one thread.
    #   Threads are allocated to workers as they become available
    client = dask.distributed.Client(
        threads_per_worker=1, n_workers=n_runs
    )

    # Assemble list of tasks to execute
    for *_, g, rho_0, rho_max in param_space:
        config_updates = dict(
            g       = float(g),
            rho_0   = float(rho_0),
            rho_max = float(rho_max),
        ) 
        lazy_results.append(run_one(config_updates)) 

    # Compute tasks lazily - tasks in list are assigned to workers
    #   on demand and the list is populated with results asynchronously.
    
    dask.compute(*lazy_results)


