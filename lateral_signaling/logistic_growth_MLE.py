#!/usr/bin/env python
# coding: utf-8

import os
import h5py

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats as st

import bebi103

import lateral_signaling as lsig


# Locs for reading
data_dir   = os.path.abspath("../data/growth_curves_MLE")
data_fname = os.path.join(data_dir, "growth_curves.csv")

# Locs for writing
save_dir           = os.path.abspath("../plots")
bs_reps_dump_fpath = os.path.join(data_dir, "growth_curve_bs_reps.hdf5")
mle_df_fpath       = os.path.join(data_dir, "growth_parameters_MLE__.csv")

#dens_curve_pfx     = os.path.join(save_dir, "growth_curves_")
#corner_plot_pfx    = os.path.join(save_dir, "MLE_corner_plot_")

## Define functions for MLE of parameters and bootstrapping

## Set initial guesses for param vals 
initial_guesses = np.array([
    1.0,   # Intrinsic proliferation rate (days ^ -1)
    5000,  # Carrying capacity (mm ^ -2)
])

## Set explicit bounds on parameter values
# Proliferation rate bounds (days^-1)
prolif_min = 0.
prolif_max = 20.

# Carrying capcity bounds
cc_min = 1e2
cc_max = 2e5

# Package for scipy least_squares function
arg_bounds = [
    [prolif_min, cc_min],
    [prolif_max, cc_max],
]

# Define functions for MLE procedure on logistic equation

def logistic_resid(params, t, rhos, rho_0s):
    """
    Residual for a logistic growth model as a function of 
    carrying capacity and intrinsic prolif. rate, given 
    initial populations and time-points.
    Used to compute estimates for growth parameters.
    """
    g, rho_max = params
    means = lsig.logistic(t, g, rho_0s, rho_max)
    return rhos - means


def logistic_mle_lstq(data, method="trf"):
    """
    Compute MLE for carrying capacity, intrinsic 
    prolif. rate, and RMSD of the logistic growth model.
    """
    
    t, rhos, rho_0s = data
    
    # Get the maximum likelihood estimate (MLE) parameters
    res = scipy.optimize.least_squares(
        logistic_resid, 
        initial_guesses, 
        args=(t, rhos, rho_0s), 
        method=method,
        bounds=arg_bounds,
    )
    
    # Get residuals using MLE parameters
    resid = logistic_resid(res.x, t, rhos, rho_0s)
    
    # Compute root-mean-squared deviation (RMSD)
    sigma_mle = np.sqrt(np.mean(resid ** 2))
    
    return tuple([*res.x, sigma_mle])


def gen_logistic_data(params, t, rho_0s, size, rg):
    """Generate a new logistic growth data set given parameters."""
    g, rho_max, sigma = params
    mus = lsig.logistic(t, g, rho_0s, rho_max)
    gen_rho = np.maximum(rg.normal(mus, sigma), 0)

    return [t, gen_rho, rho_0s]


def main(
    param_names=["untreated", "FGF2", "RI"],
    param_names_plot=["g (days^-1)", "ρ_max (mm^-2)", "σ (mm^-2)"],
    CI_pct=90,
    n_bs_reps=1000000,
    n_jobs=32,
    progress_bar=True,
    save=False,
    fmt="png",
    dpi=300,
):

    # Set RNG seed
    rg = np.random.default_rng(seed=seed)

    # Read in data and sort rows
    df = pd.read_csv(data_fname)
    df.treatment = pd.Categorical(df.treatment, categories=param_names)
    df = df.sort_values(
        by=["treatment", "initial cell density (mm^-2)", "days_integer", "replicate"]
    ).reset_index(drop=True)

    # Unpack conditions and density data
    conds = []
    rhos = []

    for i, grp in enumerate(df.groupby(["initial cell density (mm^-2)", "treatment"])):
        
        # Unpack 
        _rho_0, _cond = grp[0]
        
        # Store init density and drug condition
        conds.append(grp[0])
        
        # Get density data in chronological order
        d = grp[1].sort_values("time (days)")
        rhos.append(d["cell density (mm^-2)"].values)
        
    rhos            = np.asarray(rhos)

    # Get number of samples, unique conditions, and replicates of conditions
    nsamp  = df.shape[0]
    ncond  = len(conds)
    nrep   = df.replicate.unique().size
    ntreat = df.treatment.unique().size
    ndens  = df["initial cell density (mm^-2)"].unique().size

    ## Perform MLE on samples in each drug treatment
    mle_results_list = []
    bs_reps_list     = []
    
    for treatment in param_names:

        # Isolate samples for this treatment
        data = df.loc[df["treatment"] == treatment].sort_values(
            ["initial cell density (mm^-2)", "days_integer", "replicate"]
        ).pivot(
            index=["initial cell density (mm^-2)"],
            columns=["replicate", "days_integer"],
            values=["cell density (mm^-2)"],
        )

        # Get data for MLE fitting
        t      = np.tile([tup[2] for tup in data.columns], data.shape[0])
        rho_0s = np.repeat(data.index.values, data.shape[1])
        rhos   = data.values.flatten()
        data   = [t, rhos, rho_0s]

        # Get least-squares estimate of MLE
        mle_results = logistic_mle_lstq(data)
        
        # Bootstrap replicates of maximum likelihood estimation
        bs_reps = bebi103.bootstrap.draw_bs_reps_mle(
            logistic_mle_lstq,
            gen_logistic_data,
            data=data,
            mle_args=(),
            gen_args=(t, rho_0s),
            size=n_bs_reps,
            n_jobs=n_jobs,
            progress_bar=progress_bar,
        )
        
        # Store results
        mle_results_list.append(mle_results)
        bs_reps_list.append(bs_reps)

    ## Package results of MLE into dataframe
    # Quantities to save
    mle_columns = [
        "treatment",
        "g_inv_days",
        "rho_max_inv_mm2",
        "sigma_inv_mm2",
        "g_inv_days_90CI_lo",
        "g_inv_days_90CI_hi",
        f"rho_max_inv_mm2_{int(CI_pct)}CI_lo",
        f"rho_max_inv_mm2_{int(CI_pct)}CI_hi",
        "sigma_inv_mm2_90CI_lo",
        "sigma_inv_mm2_90CI_hi",
        "g_ratio",
        "rho_max_ratio",
        "doubling_time_days",
        "doubling_time_hours",
    ]

    # Compute confidence intervals
    CI_bounds = 50 - CI_pct / 2, 50 + CI_pct / 2
    conf_ints = np.array([
        np.percentile(_bsr, CI_bounds, axis=0).flatten()
        for _bsr in bs_reps_list
    ]).T

    # Package MLE of params
    mle_results = np.asarray(mle_results_list).T

    # MLE in dimensionless units
    mle_ratios = mle_results[:2] / np.array([[mle_results[0, 0]], [1250]])

    # MLE of doubling time
    doubling_time_days = np.log(2) / mle_results[:1]
    doubling_time_hours = doubling_time_days * 24

    # Package into dataframe
    conditions = np.array(param_names)[np.newaxis, :]
    mle_data = np.block([
        [conditions],
        [mle_results], 
        [conf_ints], 
        [mle_ratios], 
        [doubling_time_days], 
        [doubling_time_hours],
    ])
    mle_df = pd.DataFrame(
        data=dict(zip(mle_columns, mle_data)), 
    )

    if save:
        
        # Save MLE of parameters
        print("Writing to:", mle_df_fpath)
        mle_df.to_csv(mle_df_fpath)
        
        # Save bootstrap replicates
        print("Writing to:", bs_reps_dump_fpath) 
        with h5py.File(bs_reps_dump_fpath, "w") as f:
            for n, bsr in zip(param_names, bs_reps_list):
                f.create_dataset("bs_reps_" + n, data=bsr)


main(
    seed=2021,
    save=True,
)

