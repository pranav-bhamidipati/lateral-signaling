from functools import partial
import pandas as pd
import numpy as np

from numba import njit, prange
from tqdm import tqdm

from lateral_signaling import data_dir, analysis_dir


# FACS_dir = data_dir.joinpath("FACS", "perturbations")
# metadata_fname = FACS_dir.joinpath("metadata.csv")
# metadata_res_fname = analysis_dir.joinpath("FACS_perturbations_LLR_results.csv")

# FACS_dir = data_dir.joinpath("FACS/2024_mESC_and_L929")
# metadata_fname = FACS_dir.joinpath("metadata_mESC.csv")
# metadata_res_fname = analysis_dir.joinpath("2024_mESC_LLR_results.csv")

# FACS_dir = data_dir.joinpath("FACS/2024_mESC_and_L929")
# metadata_fname = FACS_dir.joinpath("metadata_L929.csv")
# metadata_res_fname = analysis_dir.joinpath("2024_L929_LLR_results.csv")


@njit
def draw_bootstrap_sample(arr: np.ndarray, seed: int, size: int = -1) -> np.ndarray:
    """Draws a bootstrap sample from data"""
    if size == -1:
        size = arr.size

    np.random.seed(seed)
    indices = np.random.randint(0, arr.size, size=size)
    return arr[indices]


@njit
def get_hist(arr: np.ndarray, nbins: int, min_val: float, max_val: float) -> np.ndarray:
    """Get histogram of data"""
    return np.histogram(arr, nbins, range=(min_val, max_val))[0]


@njit
def get_one_bs_llr(
    arr: np.ndarray,
    seed: int,
    bootstrap_size: int,
    OFF_logpdf: np.ndarray,
    ON_logpdf: np.ndarray,
    nbins: int,
    min_val: float,
    max_val: float,
) -> float:
    """Draws one boostrap sample from a data array `arr` and calculates log-likelihood(ON/OFF)"""

    # Draw a bootstrap sample
    bs = draw_bootstrap_sample(arr, seed, bootstrap_size)

    # Bin into histogram
    bs_hist = get_hist(bs, nbins, min_val, max_val)

    # Calculate log-likelihood ratio (LLR)
    log_like_OFF = np.sum(bs_hist * OFF_logpdf)
    log_like_ON = np.sum(bs_hist * ON_logpdf)

    return log_like_ON - log_like_OFF


@njit(parallel=True, fastmath=True)
# @njit(fastmath=True)
def get_bs_llr_mean_CI(
    arr: np.ndarray,
    seeds: np.ndarray,
    bootstrap_size: int,
    n_bs_reps: int,
    OFF_logpdf: np.ndarray,
    ON_logpdf: np.ndarray,
    nbins: int,
    min_val: float,
    max_val: float,
    conf: float = 0.95,
) -> tuple[float, float, float]:
    """
    Draws many bootstrap samples, calculates LLR, and returns
    mean and confidence intervals
    """

    # Get bootstrap replicates of LLR
    llr_bs = np.zeros((n_bs_reps,), dtype=float)
    for i in prange(n_bs_reps):
        llr_bs[i] = get_one_bs_llr(
            arr,
            seeds[i],
            bootstrap_size,
            OFF_logpdf,
            ON_logpdf,
            nbins,
            min_val,
            max_val,
        )

    # Compute mean and confidence interval
    conf_min = 0.5 - conf / 2
    conf_max = 0.5 + conf / 2
    quantiles = np.quantile(llr_bs, [conf_min, conf_max])

    return llr_bs.mean(), quantiles[0], quantiles[1]


def do_one_bootstrap_llr_analysis(
    idx_arr_and_seed: tuple[int, np.ndarray, int],
    bootstrap_size: int,
    n_bs_reps: int,
    OFF_logpdf: np.ndarray,
    ON_logpdf: np.ndarray,
    nbins: int,
    min_val: float,
    max_val: float,
    conf: float,
) -> tuple[int, float, float, float]:
    """Wrapper for get_bs_llr_mean_CI to allow parallelization"""
    idx, arr, seed = idx_arr_and_seed

    # Draw a SeedSequence of random number generators as a tuple
    seeds: np.ndarray = np.random.SeedSequence(seed).generate_state(n_bs_reps)

    bs_llr_mean_CI = get_bs_llr_mean_CI(
        arr,
        seeds,
        bootstrap_size,
        n_bs_reps,
        OFF_logpdf,
        ON_logpdf,
        nbins,
        min_val,
        max_val,
        conf,
    )

    return idx, *bs_llr_mean_CI


def main(
    metadata_fname,
    FACS_dir,
    n_bs_reps=1_000_000,
    confidence=0.95,
    master_seed=2021,
    nbins=1000,
    min_val=None,
    max_val=None,
    undo_log=True,
    reg=1e-2,
    progress=True,
    save=False,
):
    # Read in metadata
    metadata = pd.read_csv(metadata_fname)

    # Read in data
    data = []
    for f in metadata["filename"]:
        d: pd.Series = pd.read_csv(FACS_dir.joinpath(f))
        if d.shape[1] == 2:
            # Data may have been saved with an extra index column
            d = d.set_index(d.columns[0])
        d = d.squeeze()
        d = d[~np.isinf(d)]
        if undo_log:
            d = d[d > 0]
            d = 10**d
            d = d.round().astype(int)
        data.append(d)

    ## Calculate empirical PDF and log-PDF of reference distributions by binning
    ##   observations into histogram
    if min_val is None:
        min_val = np.min([d.min() for d in data])
    if max_val is None:
        max_val = np.max([d.max() for d in data])
    data_hists = np.array(
        [np.histogram(d.values, nbins, range=(min_val, max_val))[0] for d in data]
    )

    # Get reference sample histograms
    OFF_idx = metadata["State"].str.contains("OFF").values.nonzero()[0][0]
    ON_idx = metadata["State"].str.contains("ON").values.nonzero()[0][0]
    OFF_hist = data_hists[OFF_idx].astype(float)
    ON_hist = data_hists[ON_idx].astype(float)

    # Regularize so that no bin has 0 count, and the same probability mass
    #  is added to each bin
    regularization = np.ones(nbins, dtype=np.float64) / nbins
    OFF_pdf = OFF_hist / np.sum(OFF_hist)
    OFF_pdf = (1 - reg) * OFF_pdf + reg * regularization
    OFF_logpdf = np.log10(OFF_pdf)

    ON_pdf = ON_hist / np.sum(ON_hist)
    ON_pdf = (1 - reg) * ON_pdf + reg * regularization
    ON_logpdf = np.log10(ON_pdf)

    min_sample_size = np.min([d.size for d in data])
    get_one_result = partial(
        do_one_bootstrap_llr_analysis,
        bootstrap_size=min_sample_size,
        n_bs_reps=n_bs_reps,
        OFF_logpdf=OFF_logpdf,
        ON_logpdf=ON_logpdf,
        nbins=nbins,
        min_val=min_val,
        max_val=max_val,
        conf=confidence,
    )

    # Generate random number generators for each task
    master_ss = np.random.SeedSequence(master_seed)
    input_seeds = master_ss.generate_state(len(data))
    inputs = [(i, d.values, s) for i, (d, s) in enumerate(zip(data, input_seeds))]
    if progress:
        pbar = tqdm(total=len(data), desc="Calculating LLR")

    # Execute tasks
    indices = []
    means = []
    CI_los = []
    CI_his = []

    for idx, mean, ci_lo, ci_hi in map(get_one_result, inputs):
        if progress:
            pbar.update()
        indices.append(idx)
        means.append(mean)
        CI_los.append(ci_lo)
        CI_his.append(ci_hi)
    pbar.close()

    llr_bs_means = np.array(means)[np.argsort(indices)]
    llr_bs_CI_lo = np.array(CI_los)[np.argsort(indices)]
    llr_bs_CI_hi = np.array(CI_his)[np.argsort(indices)]

    # Make a new metadata table with results
    metadata_res = metadata.copy()
    metadata_res["LLR_mean"] = llr_bs_means
    metadata_res["LLR_95CI_lo"] = llr_bs_CI_lo
    metadata_res["LLR_95CI_hi"] = llr_bs_CI_hi

    # Save
    if save:
        from datetime import datetime

        today = datetime.today().strftime("%y%m%d")
        fname = analysis_dir.joinpath(f"{today}_{metadata_fname.stem}_LLR_results.csv")
        print(f"Writing to: {fname.resolve().absolute()}")
        metadata_res.to_csv(fname, index=False)


if __name__ == "__main__":

    # FACS_dir = data_dir.joinpath("FACS/perturbations")
    # metadata_fname = FACS_dir.joinpath("metadata.csv")
    # metadata_res_fname = analysis_dir.joinpath("FACS_perturbations_LLR_results.csv")

    FACS_dir = data_dir.joinpath("FACS/2024_mESC_and_L929")
    # metadata_fname = data_dir.joinpath("FACS/2024_mESC_and_L929/metadata_L929.csv")
    metadata_fname = data_dir.joinpath("FACS/2024_mESC_and_L929/metadata_mESC.csv")

    main(
        metadata_fname=metadata_fname,
        FACS_dir=FACS_dir,
        save=True,
        undo_log=False,
    )
