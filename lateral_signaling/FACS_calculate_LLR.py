import os

import pandas as pd
import numpy as np
from tqdm import tqdm

import lateral_signaling as lsig


# Fix RNG seed
seed = 2021
rng = np.random.default_rng(seed)

# Define functions to perform statistics
def draw_bs_sample(data, samplesize=None):
    """Draws a bootstrap sample from data"""
    if samplesize is None:
        samplesize = len(data)
    return rng.choice(data, size=samplesize)

# Options/files/directories for reading
data_dir       = os.path.abspath("../data/FACS")
metadata_fname = os.path.join(data_dir, "metadata.csv")

# Options/files/directories for writing
metadata_res_fname = os.path.join(data_dir, "metadata_with_LLR.csv")

# Local directory for fast read-write (used during parallelization)
local_dir = os.path.abspath("/tmp/dask-worker-space")

# Read in metadata
metadata = pd.read_csv(metadata_fname)

# REad in data
data = [pd.read_csv(os.path.join(data_dir, f)).squeeze() for f in metadata.filename]

# Get indices of reference distributions
OFF_idx = metadata.State.str.contains("OFF").values.nonzero()[0][0]
ON_idx  = metadata.State.str.contains("ON").values.nonzero()[0][0]
refdata_idx = np.array([OFF_idx, ON_idx])

# Extract reference data
OFF, ON = [data[i].values for i in refdata_idx]

## Calculate empirical PDF and log-PDF of reference distributions by binning
##   observations into histograms

# Number of bins in histogram
nbins = 1000

# Nubmer of bootstrap replicates
n_bs_reps = 1000000

# Get smallest sample size in dataset
min_sample_size = np.min([d.size for d in data])

# Number and size of bootstrap replicates
bs_rep_size = min_sample_size

# Get histogram (# observations in each bin) for each sample
data_hists = np.array([lsig.data_to_hist(d.values, nbins)[0] for d in data])

# Add 1 to every bin to avoid div by 0. Then normalize and take the logarithm
data_hists_pdf    = (data_hists + 1) / np.sum(data_hists + 1, axis=1, keepdims=True)
data_hists_logpdf = np.log10(data_hists_pdf)

# Get reference and experimental sample histograms
OFF_hist_logpdf, ON_hist_logpdf = data_hists_logpdf[refdata_idx]

## Define functions to compute many bootstrap replicates of 
##   the log-likelihood ratio calculation. The `dask` package
##   is used for distributed computation but this can be run
##   without `dask` using multithreading or simply as a loop
##   on a single thread.
import dask
import dask.delayed
import dask.distributed

def get_one_bs_llr(d, samplesize, OFF_logpdf, ON_logpdf):
    """Draws one boostrap sample from `d` and calculates log-likelihood(ON/OFF)"""
    
    # Draw a bootstrap sample
    bs = draw_bs_sample(data=d, samplesize=bs_rep_size)

    # Bin into histogram
    bs_hist = lsig.data_to_hist(bs, nbins)[0]

    # Calculate log-likelihood ratio (LLR)
    log_like_OFF = np.sum(bs_hist * OFF_hist_logpdf)
    log_like_ON  = np.sum(bs_hist * ON_hist_logpdf)
    
    return log_like_ON - log_like_OFF

@dask.delayed
def get_bs_llr_mean_CI(d, samplesize, n_bs_reps, OFF_logpdf, ON_logpdf, conf=0.95):
    """
    Draws many bootstrap samples, calculates LLR, and returns 
    mean and confidence intervals
    """
    
    # Get bootstrap replicates of LLR
    llr_bs = np.zeros((n_bs_reps,), dtype=float)
    for j in range(n_bs_reps):
        llr = get_one_bs_llr(d, samplesize, OFF_logpdf, ON_logpdf)
        llr_bs[j] = llr
    
    # Compute mean and confidence interval
    return np.mean(llr_bs), *np.quantile(llr_bs, [0.5 - conf / 2, 0.5 + conf / 2])


def main(
    metadata_res_fname=metadata_res_fname,
    local_dir=local_dir,
    bs_rep_size=bs_rep_size,
    n_bs_reps=n_bs_reps,
    save=False,
):
    # Make a client 
    #   The client monitors workers to make sure tasks are allocated efficiently 
    #   (Click on URL to monitor tasks using the dashboard.)
    client = dask.distributed.Client(
        threads_per_worker=1, 
        # n_workers=n_workers,
        # memory_limit=memory_limit,
        # interface="ib0",
        # timeout=600,
        local_directory=local_dir,
    )
    print("Dask dashboard running at: ", client.dashboard_link, sep="\n")

    # Assemble list of tasks
    lazy_results = [
        get_bs_llr_mean_CI(
            d=d, 
            samplesize=bs_rep_size, 
            n_bs_reps=n_bs_reps,
            OFF_logpdf=OFF_hist_logpdf,
            ON_logpdf=ON_hist_logpdf,
            conf=0.95,
        )
        for d in data
    ]

    # Execute tasks 
    llr_bs_means_95CI = np.array(dask.compute(*lazy_results))

    # Unpack results
    llr_bs_means = llr_bs_means_95CI[:, 0]
    llr_bs_95CI  = llr_bs_means_95CI[:, 1:]

    # Make a new metadata table with results
    metadata_res = metadata.copy()
    metadata_res["LLR_mean"]    = llr_bs_means
    metadata_res["LLR_95CI_lo"] = llr_bs_95CI[:, 0]
    metadata_res["LLR_95CI_hi"] = llr_bs_95CI[:, 1]

    # Save
    if save:
        metadata_res.to_csv(metadata_res_fname, index=False)

if __name__ == "__main__":
    main(
        save=True,
    )