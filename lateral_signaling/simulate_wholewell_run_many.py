from pathlib import Path
import dask
import dask.distributed
from typing import Union

from lateral_signaling import _dask_client_default_kwargs

# Set dir for Dask to use (spill to disk, etc.)
local_dir = Path("/home/pbhamidi/scratch/lateral_signaling/dask-worker-space")


@dask.delayed
def run_one_task(config_updates):
    """Run single simulation - executed independently in every thread"""
    from simulate_wholewell_run_one import ex

    ex.run(config_updates=config_updates)


def main(
    mle_csvs: list[Path],
    treatment_names: list[Union[list[str], None]],
    rho_0s=[1.0],
    save_skip=10,
    local_dir=local_dir,
    memory_allocation_percentage=0.85,
    client_kwargs=_dask_client_default_kwargs,
    **kw,
):

    import os
    import psutil
    import pandas as pd

    # Read in growth parameters
    conds = []
    gs = []
    rho_maxs = []
    for mle_csv, t_names in zip(mle_csvs, treatment_names):
        mle_params_df = pd.read_csv(mle_csv, index_col=0)
        if t_names is not None:
            mle_params_df = mle_params_df.loc[mle_params_df.treatment.isin(t_names)]
            mle_params_df["treatment"] = pd.Categorical(
                mle_params_df["treatment"], categories=t_names, ordered=True
            )
            mle_params_df = mle_params_df.sort_values("treatment")
        conds.extend(mle_params_df["treatment"])
        gs.extend(mle_params_df["g_ratio"])
        rho_maxs.extend(mle_params_df["rho_max_ratio"])

    n_runs = len(conds) * len(rho_0s)

    # Set options based on whether this is being run in a SLURM environment or locally
    kwargs = client_kwargs.copy()
    if slurm_ID := os.environ.get("SLURM_JOB_ID"):
        kwargs["n_workers"] = os.environ["SLURM_NPROCS"]  # Number of available threads
        mb_per_cpu = int(
            int(os.environ["SLURM_MEM_PER_CPU"]) * memory_allocation_percentage
        )
        kwargs["memory_limit"] = f"{mb_per_cpu} MiB"
        kwargs["interface"] = "ib0"
    else:
        n_threads = psutil.cpu_count(logical=True)
        kwargs["n_workers"] = min(n_threads, n_runs)

    kwargs.update(kw)

    # Configure a Client that will spawn a local cluster of workers.
    #   Each task gets one worker and one worker gets one thread.
    #   Threads are allocated to workers as they become available
    client = dask.distributed.Client(**kwargs, local_directory=local_dir)

    print("Building list of tasks to execute asynchronously")

    lazy_results = []
    for cond, g, rho_max in zip(conds, gs, rho_maxs):
        for rho_0 in rho_0s:
            config_updates = {
                "drug_condition": cond,
                "rho_0": rho_0,
                "g": g,
                "rho_max": rho_max,
                "save_skip": save_skip,
            }
            lazy_results.append(run_one_task(config_updates))

    print("Executing tasks...")
    dask.compute(*lazy_results)


if __name__ == "__main__":
    from lateral_signaling import analysis_dir

    # mle_csv = analysis_dir.joinpath("growth_parameters_MLE.csv")
    mle_csvs = [
        analysis_dir.joinpath("240327_growth_parameters_MLE.csv"),
        analysis_dir.joinpath("240402_growth_parameters_MLE_fixed_rhomax.csv"),
    ]
    treatment_names = [["10% FBS"], None]

    # Uncomment to run locally
    local_dir = Path("/tmp/dask-worker-space")
    local_dir.mkdir(exist_ok=True)

    main(
        mle_csvs,
        treatment_names,
        n_workers=3,
        memory_limit="18 GiB",
        local_dir=local_dir,
    )
