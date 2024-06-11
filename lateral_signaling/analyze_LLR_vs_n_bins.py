import h5py
from functools import partial
from pathlib import Path
import pandas as pd
import numpy as np
from numba import njit, prange
from tqdm import tqdm
import lateral_signaling as lsig

from analyze_FACS_LLR import get_one_bs_llr


def get_data_and_reference_idx(metadata_csv: Path, FACS_dir: Path, undo_log: bool):

    mdata = pd.read_csv(metadata_csv)

    # Read in data
    data = []
    for f in mdata["filename"]:
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

    OFF_idx = mdata.State.str.contains("OFF").values.nonzero()[0][0]
    OFF_arr = data[OFF_idx].values
    ON_idx = mdata.State.str.contains("ON").values.nonzero()[0][0]
    ON_arr = data[ON_idx].values

    return data, OFF_arr, ON_arr


@njit(parallel=True, fastmath=True)
def get_bs_llr_mean_vs_nbins(
    arr: np.ndarray,
    seeds: np.ndarray,
    bootstrap_size: int,
    nbins_range: np.ndarray,
    n_bs_reps: int,
    OFF_arr: np.ndarray,
    ON_arr: np.ndarray,
    min_val: float,
    max_val: float,
    reg: float = 1e-2,
) -> np.ndarray:
    """
    Draws a few bootstrap samples, calculates LLR, and returns
    mean LLR for various number of histogram bins.
    """

    # Get bootstrap replicates of LLR
    n_nbins = nbins_range.size
    llr_bs = np.zeros((n_nbins, n_bs_reps), dtype=float)
    for i in prange(n_nbins):

        # Bin reference data into histograms
        nbins = nbins_range[i]
        OFF_hist = np.histogram(OFF_arr, nbins, range=(min_val, max_val))[0]
        ON_hist = np.histogram(ON_arr, nbins, range=(min_val, max_val))[0]

        # Regularize so that no bin has 0 count, and the same probability mass
        #  is added to each bin
        regularization = np.ones(nbins, dtype=np.float64) / nbins
        OFF_pdf = OFF_hist / np.sum(OFF_hist)
        OFF_pdf = (1 - reg) * OFF_pdf + reg * regularization
        OFF_logpdf = np.log10(OFF_pdf)

        ON_pdf = ON_hist / np.sum(ON_hist)
        ON_pdf = (1 - reg) * ON_pdf + reg * regularization
        ON_logpdf = np.log10(ON_pdf)

        for j in prange(n_bs_reps):
            llr_bs[i, j] = get_one_bs_llr(
                arr,
                seeds[i, j],
                bootstrap_size,
                OFF_logpdf,
                ON_logpdf,
                nbins,
                min_val,
                max_val,
            )

            if np.isnan(llr_bs[i, j]):
                print(f"NaN found for seed: {seeds[i, j]}")

    return llr_bs.sum(axis=1) / n_bs_reps


def calculate_mean_LLR(
    arr_and_seed: tuple[np.ndarray, int],
    bootstrap_size: int,
    nbins_range: np.ndarray,
    n_bs_reps: int,
    OFF_arr: np.ndarray,
    ON_arr: np.ndarray,
    min_val: float,
    max_val: float,
    reg: float = 1e-2,
):
    arr, seed = arr_and_seed
    n_nbins = nbins_range.size
    seeds = (
        np.random.SeedSequence(seed)
        .generate_state(n_nbins * n_bs_reps)
        .reshape(n_nbins, -1)
    )
    bs_llr_mean = get_bs_llr_mean_vs_nbins(
        arr,
        seeds,
        bootstrap_size,
        nbins_range,
        n_bs_reps,
        OFF_arr,
        ON_arr,
        min_val,
        max_val,
        reg=reg,
    )

    if np.isnan(bs_llr_mean).any():
        print(f"NaNs found for seed: {seed}")

    return bs_llr_mean


def main(
    metadata_with_results_csv,
    FACS_dir,
    n_nbins=500,
    max_nbins=int(1e6),
    n_bs_reps=20,
    min_val=None,
    max_val=None,
    undo_log=True,
    master_seed=2021,
    progress=True,
    regularize=1e-2,
    save_dir=lsig.analysis_dir,
    save=False,
):

    data, OFF_arr, ON_arr = get_data_and_reference_idx(
        metadata_with_results_csv, FACS_dir, undo_log
    )
    if min_val is None:
        min_val = np.min([d.min() for d in data])
    if max_val is None:
        max_val = np.max([d.max() for d in data])

    min_sample_size = np.min([d.size for d in data])
    nbins_range = np.unique(np.geomspace(1, max_nbins, n_nbins).astype(int))

    get_LLRs_given_nbins = partial(
        calculate_mean_LLR,
        bootstrap_size=min_sample_size,
        nbins_range=nbins_range,
        n_bs_reps=n_bs_reps,
        OFF_arr=OFF_arr,
        ON_arr=ON_arr,
        min_val=min_val,
        max_val=max_val,
        reg=regularize,
    )

    seeds = np.random.SeedSequence(master_seed).generate_state(len(data))
    iterator = [(d.values, s) for d, s in zip(data, seeds)]
    if progress:
        iterator = tqdm(iterator, total=len(data), desc="Calculating LLRs")

    results = []
    for inputs in iterator:
        results.append(get_LLRs_given_nbins(inputs))

    llrs_v_nbins = np.array(results).T

    if save:
        save_path = save_dir.joinpath(f"{metadata_res_csv.stem}_vs_nbins.hdf5")
        print(f"Writing data to: {save_path.resolve().absolute()}")
        with h5py.File(save_path, "w") as f:
            f.create_dataset("nbins_range", data=nbins_range)
            f.create_dataset("llrs_v_nbins", data=llrs_v_nbins)


if __name__ == "__main__":
    # FACS_dir = lsig.data_dir.joinpath("FACS", "perturbations")
    # metadata_res_csv = lsig.analysis_dir.joinpath("FACS_perturbations_LLR_results.csv")

    FACS_dir = lsig.data_dir.joinpath("FACS/2024_mESC_and_L929")
    # metadata_res_csv = lsig.analysis_dir.joinpath(
    #     "240402_metadata_L929_LLR_results.csv"
    # )
    metadata_res_csv = lsig.analysis_dir.joinpath(
        "240402_metadata_mESC_LLR_results.csv"
    )

    main(
        metadata_with_results_csv=metadata_res_csv,
        FACS_dir=FACS_dir,
        undo_log=False,
        save=True,
        # n_nbins=5,
    )
