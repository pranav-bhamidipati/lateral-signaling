import h5py
from functools import partial
import multiprocessing as mp
import psutil

import pandas as pd
import numpy as np
import holoviews as hv

hv.extension("matplotlib")

import lateral_signaling as lsig

lsig.default_rcParams()

FACS_dir = lsig.data_dir.joinpath("FACS", "perturbations")
metadata_res_csv = lsig.analysis_dir.joinpath("FACS_perturbations_LLR_results.csv")


def get_data_and_reference_idx(metadata_with_results_csv):

    metadata_res = pd.read_csv(metadata_with_results_csv)

    data = [pd.read_csv(FACS_dir.joinpath(f)).squeeze() for f in metadata_res.filename]
    OFF_idx = metadata_res.State.str.contains("OFF").values.nonzero()[0][0]
    ON_idx = metadata_res.State.str.contains("ON").values.nonzero()[0][0]

    return data, OFF_idx, ON_idx


def calculate_LLR(data, OFF_idx, ON_idx, _nbins):

    _data_hists = np.array([lsig.data_to_hist(d.values, _nbins)[0] for d in data])

    # Add 1 to each bin to avoid div by 0. (regularization)
    _data_hists = _data_hists + 1
    _data_hists_pdf = _data_hists / np.sum(_data_hists, axis=1, keepdims=True)
    _data_hists_logpdf = np.log10(_data_hists_pdf)

    # Compare to reference log-PDFs
    _OFF_hist_logpdf = _data_hists_logpdf[OFF_idx]
    _ON_hist_logpdf = _data_hists_logpdf[ON_idx]
    log_like_OFF = np.sum(_data_hists * _OFF_hist_logpdf, axis=1)
    log_like_ON = np.sum(_data_hists * _ON_hist_logpdf, axis=1)

    return log_like_ON - log_like_OFF


def get_nbins_range(n_nbins, max_nbins):
    """Returns a range of integers to use as the number of histogram bins"""


def main(
    metadata_with_results_csv=metadata_res_csv,
    n_nbins=500,
    max_nbins=int(1e6),
    save_dir=lsig.analysis_dir,
    save=False,
):

    get_LLRs_given_nbins = partial(
        calculate_LLR, *get_data_and_reference_idx(metadata_with_results_csv)
    )

    nbins_range = np.unique(np.geomspace(1, max_nbins, n_nbins).astype(int))

    n_threads = psutil.cpu_count(logical=True)
    print(f"Assembling thread pool ({n_threads} workers)")

    pool = mp.Pool(n_threads)
    results = pool.map(get_LLRs_given_nbins, nbins_range, chunksize=5)
    pool.close()
    pool.join()
    print("Complete")

    llrs_v_nbins = np.array(results)

    if save:
        save_path = save_dir.joinpath("FACS_perturbations_LLR_vs_nbins.hdf5")
        print(f"Writing data to: {save_path.resolve().absolute()}")
        with h5py.File(save_path, "w") as f:
            f.create_dataset("nbins_range", data=nbins_range)
            f.create_dataset("llrs_v_nbins", data=llrs_v_nbins)


if __name__ == "__main__":
    main(
        save=True,
        # n_nbins=5,
    )
