import json
import h5py
import numpy as np
from pathlib import Path

# Locate simulations of steady-state expression
# ss_sacred_dir = Path("./sacred")
ss_sacred_dir = Path("../data/simulations/20220726_steadystate/sacred")
ss_data_dirs = sorted(list(ss_sacred_dir.glob("[0-9]*")))

# Extract some metadata first
def get_metadata(results_file):
    nscan = len(ss_data_dirs)
    with h5py.File(results_file, "r") as h:
        nreps, nc = np.asarray(h["S_final_rep"]).shape
        nsenders = np.asarray(h["sender_idx_rep"]).shape[1]
        ntc = nc - nsenders
        rep_idx = np.repeat(np.arange(nreps), nsenders)
    return nscan, nreps, nc, nsenders, ntc, rep_idx


# Extract data from a parameter scan across density (rho)
def _initialize():
    """Must be run before get_steady_state and get_steady_state_vector can be used."""
    nscan, nreps, nc, nsenders, ntc, rep_idx = get_metadata(
        ss_data_dirs[0].joinpath("results.hdf5")
    )
    rho_scan_unsorted = []
    S_tcmean_rep_scan_unsorted = []

    ss_data_files = [
        (d.joinpath("config.json"), d.joinpath("results.hdf5"))
        for d in ss_data_dirs
        if d.joinpath("config.json").exists()
    ]
    for i, (config_file, data_file) in enumerate(ss_data_files):
        with config_file.open("r") as f:
            j = json.load(f)
            rho_0 = j["rho_0"]

        with h5py.File(data_file, "r") as h:
            tc_mask_rep = np.asarray(h["tc_mask_rep"])
            S_final_rep = np.asarray(h["S_final_rep"])

        S_tc_final_rep = S_final_rep[tc_mask_rep].reshape(nreps, ntc)
        S_tcmean_rep = S_tc_final_rep.mean(axis=1)

        # Save the density and mean transceiver expression at the final time-point
        rho_scan_unsorted.append(rho_0)
        S_tcmean_rep_scan_unsorted.append(S_tcmean_rep)

    rho_scan_unsorted = np.asarray(rho_scan_unsorted)
    S_tcmean_rep_scan_unsorted = np.asarray(S_tcmean_rep_scan_unsorted)
    sort_rho = np.argsort(rho_scan_unsorted)
    rho_scan = rho_scan_unsorted[sort_rho]

    # Mean and std. dev. of expression will be used as estimate of steady-state
    S_tcmean_rep_scan = S_tcmean_rep_scan_unsorted[sort_rho]
    S_tcmean_scan = S_tcmean_rep_scan.mean(axis=1)
    S_tcmean_scan_std = S_tcmean_rep_scan.std(axis=1)

    minval = rho_scan.min()
    maxval = rho_scan.max()

    return rho_scan, minval, maxval, nscan, S_tcmean_scan, S_tcmean_scan_std


def _get_steady_state(
    rho_scan,
    minval,
    maxval,
    nscan,
    S_tcmean_scan,
    S_tcmean_scan_std,
    rho,
    method="nearest",
):
    """
    Get approximate steady-state ligand expression of Transceivers at density `rho`.
    Returns mean and standard deviation of 10 replicate simulations.
    """

    if not (minval <= rho <= maxval):
        raise ValueError(
            f"Argument `rho` must be within the range: `{minval:.3f} <= rho <= {maxval:.3f}`"
        )

    if method in ("left", "right"):
        idx = np.searchsorted(rho_scan, rho, side=method)
        mean = S_tcmean_scan[idx]
        std = S_tcmean_scan_std[idx]
    elif method == "nearest":
        idx = round((nscan - 1) * (rho - minval) / (maxval - minval))
        mean = S_tcmean_scan[idx]
        std = S_tcmean_scan_std[idx]
    else:
        raise ValueError

    return mean, std
