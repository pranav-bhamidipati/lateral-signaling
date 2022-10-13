import json
import h5py
import numpy as np


def get_metadata(results_file):
    """Extract some metadata"""
    with h5py.File(results_file, "r") as h:
        nreps, nc = np.asarray(h["S_final_rep"]).shape
        nsenders = np.asarray(h["sender_idx_rep"]).shape[1]
        ntc = nc - nsenders
        rep_idx = np.repeat(np.arange(nreps), nsenders)
    return nreps, nc, nsenders, ntc, rep_idx


def _find_linear_root(x1, y1, x2, y2, c=0.0):
    """Solve for the root of `(y2 - y1) - m * (x2 - x1) = c`, where `m` is the
    slope of the line between (x1, y1) and (x2, y2). The root should be known to
    lie between `x1` and `x2`.
    """

    dy = y2 - y1
    dx = x2 - x1
    return x1 + (c - y1) / dy * dx


# Extract data from a parameter scan across density (rho)
def _initialize(_ss_sacred_dir):
    """Must be run before steady state functions."""

    from lateral_signaling import simulation_params

    _ss_data_dirs = sorted(list(_ss_sacred_dir.glob("[0-9]*")))
    nscan = len(_ss_data_dirs)

    nreps, nc, nsenders, ntc, rep_idx = get_metadata(
        _ss_data_dirs[0].joinpath("results.hdf5")
    )
    rho_scan_unsorted = []
    S_tcmean_rep_scan_unsorted = []

    ss_data_files = [
        (d.joinpath("config.json"), d.joinpath("results.hdf5"))
        for d in _ss_data_dirs
        if d.joinpath("config.json").exists()
    ]

    for config_file, data_file in ss_data_files:

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
    S_tcmean_scan_mean = S_tcmean_rep_scan.mean(axis=1)
    S_tcmean_scan_std = S_tcmean_rep_scan.std(axis=1)

    minval = rho_scan.min()
    maxval = rho_scan.max()

    def _get_steady_state_mean(rho):
        """
        Get approximate steady-state ligand expression of Transceivers at density `rho`.
        Returns mean of 10 replicate simulations.
        """

        idx = round((nscan - 1) * (rho - minval) / (maxval - minval))
        mean = S_tcmean_scan_mean[idx]
        return mean

    def _get_steady_state_std(rho):
        """
        Get approximate steady-state ligand expression of Transceivers at density `rho`.
        Returns standard deviation of 10 replicate simulations.
        """

        idx = round((nscan - 1) * (rho - minval) / (maxval - minval))
        std = S_tcmean_scan_std[idx]
        return std

    def _get_steady_state_replicates(rho):
        """
        Get approximate steady-state ligand expression of Transceivers at density `rho`.
        Returns all replicate simulations.
        """

        idx = round((nscan - 1) * (rho - minval) / (maxval - minval))
        reps = S_tcmean_rep_scan[idx]
        return reps

    def _get_steady_state_ci_lo(rho, conf_int):
        """
        Get approximate steady-state ligand expression of Transceivers at density `rho`.
        Returns confidence intervals over replicate simulations.
        """

        idx = round((nscan - 1) * (rho - minval) / (maxval - minval))
        reps = S_tcmean_rep_scan[idx]

        lower_quantile = 1 / 2 - conf_int / 2

        return np.quantile(reps, lower_quantile)

    def _get_steady_state_ci_hi(rho, conf_int):
        """
        Get approximate steady-state ligand expression of Transceivers at density `rho`.
        Returns confidence intervals over replicate simulations.
        """

        idx = round((nscan - 1) * (rho - minval) / (maxval - minval))
        reps = S_tcmean_rep_scan[idx]

        upper_quantile = 1 / 2 + conf_int / 2

        return np.quantile(reps, upper_quantile)

    ## Simulated GFP conc. can cross the threshold many times (back and forth) due
    ## to stochasticity. We take the most extreme values as the low and high ones
    critical_idx = np.diff(np.sign(S_tcmean_scan_mean - simulation_params.k)).nonzero()[
        0
    ]

    crit_idx_lo = critical_idx[0]
    crit_rho_lo = _find_linear_root(
        rho_scan[crit_idx_lo],
        S_tcmean_scan_mean[crit_idx_lo],
        rho_scan[crit_idx_lo + 1],
        S_tcmean_scan_mean[crit_idx_lo + 1],
        c=simulation_params.k,
    )

    crit_idx_hi = critical_idx[-1]
    crit_rho_hi = _find_linear_root(
        rho_scan[crit_idx_hi],
        S_tcmean_scan_mean[crit_idx_hi],
        rho_scan[crit_idx_hi + 1],
        S_tcmean_scan_mean[crit_idx_hi + 1],
        c=simulation_params.k,
    )

    return (
        _get_steady_state_mean,
        _get_steady_state_std,
        _get_steady_state_replicates,
        _get_steady_state_ci_lo,
        _get_steady_state_ci_hi,
    ), {"rho_ON": crit_rho_lo, "rho_OFF": crit_rho_hi}
