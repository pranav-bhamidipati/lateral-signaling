from datetime import datetime
from typing import Optional
from functools import partial
import h5py
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit
import scipy.optimize
from tqdm import tqdm

from lateral_signaling import data_dir, analysis_dir, logistic


@njit
def logistic_resid(params, t, rhos, rho_0s, which_treatment: np.ndarray):
    """
    Residual for a logistic growth model as a function of
    carrying capacity and intrinsic prolif. rate, given
    initial populations and time-points.
    Used to compute estimates for growth parameters.
    """
    g = params[:-1][which_treatment]
    rho_max = params[-1]
    means = logistic(t, g, rho_0s[which_treatment], rho_max)
    return rhos - means


@njit
def rms(vals):
    return np.sqrt(np.mean(vals**2))


def compute_least_squares_logistic_mle(t, rho, rho_0, which_treatment, initial_guesses):
    """Compute MLE for carrying capacity, intrinsic prolif. rate, and RMSD of the
    logistic growth model."""

    # Get the maximum likelihood estimate (MLE) parameters
    res = scipy.optimize.least_squares(
        fun=logistic_resid,
        x0=initial_guesses,
        bounds=(
            [0, 0, 0, 0],  # Lower bounds for g and rho_max
            [12.0, 12.0, 12.0, 25_000],  # Upper bounds for g and rho_max
        ),
        args=(t, rho, rho_0, which_treatment),
    )

    # Get residuals using MLE parameters
    resid = logistic_resid(res.x, t, rho, rho_0, which_treatment)

    # Compute root-mean-squared deviation (RMSD)
    sigma_mles = np.zeros((3,), dtype=np.float64)
    for i in range(3):
        sigma_mles[i] = rms(resid[which_treatment == i])

    # Return MLE parameters and RMSD
    *gs, rho_max = res.x

    mles = np.zeros((3, 3), dtype=np.float64)
    mles[:, 0] = gs
    mles[:, 1] = rho_max
    mles[:, 2] = sigma_mles
    return mles


#### NOTE: THe below functions were modified from the bebi103 package
####  developed by Justin Bois, which has now been deprecated.


@njit
def bootstrap_by_residuals(y_fit, residuals, rg: np.random.Generator):
    """Produces a bootsrap sample of dependent variables using the best-fit values and residuals
    from a model. Resamples the residuals from the best fit."""
    bs_indices = rg.integers(0, len(residuals), size=len(residuals))
    return y_fit + residuals[bs_indices]


def bootstrap_logistic_mle(
    rho_fit,
    residuals,
    t,
    rho_0,
    which_treatment,
    initial_guesses,
    seed_and_task_id=(None, 0),
):
    """Draw a bootstrap replicate of maximum likelihood estimator for logistic growth. Uses
    `bootstrap_by_residuals()` to resample residuals from the best-fit model.

    Parameters
    ----------
    rho_fit : array-like
        Best-fit values of the dependent variable.
    residuals : array-like
        Residuals from the best-fit model.
    t : array-like
        Independent variable.
    rho_0 : array-like
        Initial values of the dependent variable.
    initial_guesses : array-like
        Initial guesses for the parameters of the model.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    g : float
        Best-fit value of the intrinsic proliferation rate.
    rho_max : float
        Best-fit value of the carrying capacity.
    sigma : float
        Best-fit value of the RMSD (i.e. the standard deviation of the residuals)
    """
    seed, task_id = seed_and_task_id
    if seed is None:
        seed = np.random.SeedSequence().entropy
    rg = np.random.default_rng([seed, task_id])
    rho_bs = bootstrap_by_residuals(rho_fit, residuals, rg)
    return compute_least_squares_logistic_mle(
        t, rho_bs, rho_0, which_treatment, initial_guesses=initial_guesses
    )


def draw_logistic_mle_bootstraps(
    rho_fit,
    residuals,
    t,
    rho_0,
    which_treatment,
    initial_guesses,
    n_bootstrap=1,
    n_procs=1,
    seed=None,
    progress_bar=False,
    chunksize=1,
):
    draw_one_bootstrap = partial(
        bootstrap_logistic_mle,
        rho_fit,
        residuals,
        t,
        rho_0,
        which_treatment,
        initial_guesses,
    )

    seeds = np.random.SeedSequence(seed).generate_state(n_bootstrap)

    # The (seed, task_id) pair is used to seed the random number generator
    # for each bootstrap replicate.
    task_ids = range(n_bootstrap)
    ins = zip(seeds, task_ids)
    if n_procs == 1:
        if progress_bar:
            ins = tqdm(ins, total=n_bootstrap, desc="Drawing bootstrap replicates")
        return np.array([draw_one_bootstrap(input) for input in ins])

    else:
        bootstrap_mles = []
        if progress_bar:
            pbar = tqdm(total=n_bootstrap, desc="Drawing bootstrap replicates")

        with Pool(n_procs) as pool:
            for mle in pool.imap_unordered(
                draw_one_bootstrap, ins, chunksize=chunksize
            ):
                bootstrap_mles.append(mle)
                if progress_bar:
                    pbar.update()
            pbar.close()

        return np.array(bootstrap_mles)


def main(
    data_fname,
    seed=2022,
    n_bootstrap=1_000_000,
    n_procs=1,
    CI_pct=90,
    initial_guesses=np.array(
        [
            1.0,  # Intrinsic proliferation rate (days ^ -1)
            1.0,
            1.0,
            5000,  # Carrying capacity (mm ^ -2)
        ]
    ),
    reference_treatment="10% FBS",
    reference_density_inv_mm2=1250.0,
    chunksize=100,
    treatment_names: list[str] = ["10% FBS", "50 µM Y-27632", "250 ng/mL FGF2"],
    progress_bar=True,
    save=False,
    save_dir=Path(analysis_dir),
):
    # Read in data and sort rows
    print("Reading in data from:", Path(data_fname).resolve())
    df = pd.read_csv(data_fname)
    df = df.loc[df["treatment"].isin(treatment_names)]
    df["treatment"] = pd.Categorical(df["treatment"], categories=treatment_names)
    df["time (days)"] = df["time (days)"].astype(int)

    ## Perform MLE jointly for multiple treatment conditions, keeping rho_max fixed
    # Get data for MLE fitting
    t = df["time (days)"].values
    rho = df["cell density (mm^-2)"].values
    rho_0s = df["initial cell density (mm^-2)"].values
    which_treatment = df["treatment"].cat.codes.values

    # Get least-squares estimate of MLE
    mle_results = compute_least_squares_logistic_mle(
        t, rho, rho_0s, which_treatment, initial_guesses=initial_guesses
    )
    for i in range(3):
        g, rho_max, sigma = mle_results[i]
        print(f"MLE results for {treatment_names[i]}:")
        print(f"\tg (days^-1): {g:.3e}")
        print(f"\trho_max (mm^-2): {rho_max:.3e}")
        print(f"\tsigma (mm^-2): {sigma:.3e}")
        print()

    # Bootstrap replicates of maximum likelihood estimation
    print(f"Drawing {n_bootstrap} bootstrap replicates of MLE...")
    g_mles = mle_results[:, 0]
    rho_max_mle = mle_results[0, 1]
    rho_fit = logistic(t, g_mles[which_treatment], rho_0s[which_treatment], rho_max_mle)
    residuals = rho - rho_fit
    bootstrap_mle = draw_logistic_mle_bootstraps(
        rho_fit,
        residuals,
        t,
        rho_0s,
        which_treatment,
        initial_guesses=initial_guesses,
        n_bootstrap=n_bootstrap,
        n_procs=n_procs,
        seed=seed,
        progress_bar=progress_bar,
        chunksize=chunksize,
    )
    print()

    ## Package results of MLE into dataframe
    # Compute confidence intervals
    CI_bounds = 50 - CI_pct / 2, 50 + CI_pct / 2
    conf_ints = np.percentile(bootstrap_mle, CI_bounds, axis=0)
    conf_ints = conf_ints.transpose((0, 2, 1)).reshape(-1, len(treatment_names))

    # Package MLE of params
    g_inv_days, rho_max_inv_mm2, sigma_inv_mm2 = mle_results.T

    # MLE in dimensionless units
    which_reference = treatment_names.index(reference_treatment)
    g_inv_days = np.array(g_inv_days)
    g_ratio = g_inv_days / g_inv_days[which_reference]
    rho_max_inv_mm2 = np.array(rho_max_inv_mm2)
    rho_max_ratio = rho_max_inv_mm2 / reference_density_inv_mm2

    # MLE of doubling time
    doubling_time_days = np.log(2) / g_inv_days
    doubling_time_hours = doubling_time_days * 24

    # Package into dataframe
    mle_data = {
        "treatment": treatment_names,
        "g_inv_days": g_inv_days,
        "rho_max_inv_mm2": rho_max_inv_mm2,
        "sigma_inv_mm2": sigma_inv_mm2,
        "g_ratio": g_ratio,
        "rho_max_ratio": rho_max_ratio,
        "doubling_time_days": doubling_time_days,
        "doubling_time_hours": doubling_time_hours,
    } | dict(
        zip(
            [
                f"g_inv_days_{int(CI_pct)}CI_lo",
                f"rho_max_inv_mm2_{int(CI_pct)}CI_lo",
                f"sigma_inv_mm2_{int(CI_pct)}CI_lo",
                f"g_inv_days_{int(CI_pct)}CI_hi",
                f"rho_max_inv_mm2_{int(CI_pct)}CI_hi",
                f"sigma_inv_mm2_{int(CI_pct)}CI_hi",
            ],
            conf_ints,
        )
    )
    mle_df = pd.DataFrame(mle_data)

    if save:
        today = datetime.today().strftime("%y%m%d")
        bs_reps_dump_fpath = save_dir.joinpath(
            f"{today}_growth_curve_bootstrap_replicates_fixed_rhomax.hdf5"
        )
        mle_df_fpath = save_dir.joinpath(
            f"{today}_growth_parameters_MLE_fixed_rhomax.csv"
        )

        # Save MLE of parameters
        print("Writing to:", mle_df_fpath)
        mle_df.to_csv(mle_df_fpath)

        # Save bootstrap replicates
        print("Writing to:", bs_reps_dump_fpath)
        with h5py.File(bs_reps_dump_fpath, "w") as f:
            for i in range(3):
                f.create_dataset(
                    f"bs_reps_{treatment_names[i]}", data=bootstrap_mle[:, i]
                )


if __name__ == "__main__":
    # data_fname = data_dir.joinpath("growth_curves_MLE/growth_curves.csv")
    data_fname = Path(data_dir.joinpath("growth_curves_MLE/231219_growth_curves.csv"))

    main(
        data_fname,
        n_procs=13,
        seed=2023,
        # n_bootstrap=10_000,
        chunksize=1000,
        save=True,
    )
