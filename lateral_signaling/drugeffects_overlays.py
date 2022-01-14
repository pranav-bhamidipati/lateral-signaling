import os
from glob import glob
import json
from copy import deepcopy
import h5py

import numpy as np
import pandas as pd
from tqdm import tqdm

import colorcet as cc
import matplotlib.pyplot as plt
import matplotlib as mpl
import holoviews as hv
hv.extension("matplotlib")

import lateral_signaling as lsig

sim_dir    = os.path.abspath("../data/simulations/20220113_drugeffects/sacred")
drug_fname = os.path.abspath("../data/wholewell/")
dens_fname = os.path.abspath("../data/singlespot/singlespot_timeseries.csv")

save_dir   = os.path.abspath("../plots")
drug_fpfx  = os.path.join(save_dir, "drugeffects_sqrtarea_")
dens_fpfx  = os.path.join(save_dir, "denseffects_sqrtarea_")
fmt        = "png"
dpi        = 300

def main(
    sim_dir=sim_dir,
    prop_fname=prop_fname,
    pad=0.05,
    sample_every=1,
    save=False,
    suffix="",
    fmt=fmt,
    dpi=dpi,
):
    
    ## Read invitro data from file
    data = pd.read_csv(prop_fname)
    data = data.loc[data.Condition.str.contains("1250 cell/mm2")]
    data.days = data.days.astype(float)
    tmax_days = data.days.max()

    # Convert area units and take sqrt
    data["Area_mm2"]  = data["area (um2)"] / 1e6
    del data["area (um2)"]
    data["sqrtA_mm"] = np.sqrt(data["Area_mm2"])

    # Get means and standard deviations
    agg_data = data.groupby(
        ["Condition", "days"]
    ).agg(
        [np.mean, np.std]
    ).reset_index()
    agg_data.columns = ['_'.join(col).strip("_") for col in agg_data.columns.values]
    agg_data["sqrtA_mm_mean_norm"] = agg_data["sqrtA_mm_mean"] / data["sqrtA_mm"].max()
    agg_data["sqrtA_mm_std_norm"]  = agg_data["sqrtA_mm_std"]  / data["sqrtA_mm"].max()
    
    ## Read simulated data
    run_dir      = glob(os.path.join(sim_dir, "[0-9]*"))[0]
    config_file  = os.path.join(run_dir, "config.json")
    results_file = os.path.join(run_dir, "results.hdf5")
    
    with open(config_file, "r") as c:
        config = json.load(c)

        # Expression threshold
        k = config["k"]
     
    with h5py.File(results_file, "r") as f:

        # Time-points
        t = np.asarray(f["t"])
        t_days = lsig.t_to_units(t)

        # Index of sender cell
        sender_idx = np.asarray(f["sender_idx"])

        # Density vs. time
        rho_t = np.asarray(f["rho_t"])

        # Signal and reporter expression vs. time
        S_t = np.asarray(f["S_t"])

    # Restrict time-range to match invitro
    tmask  = t_days <= tmax_days
    t      = t[tmask]
    t_days = t_days[tmask]
    rho_t  = rho_t[tmask]
    S_t    = S_t[tmask]
    
    # Calculate the number of activated transceivers
    n_act_t = (S_t > k).sum(axis=-1) - 1

    # Area and sqrt(Area) of activation
    A_t = lsig.ncells_to_area(n_act_t, rho_t)
    sqrtA_t = np.sqrt(A_t)

    ## Make plot
    # Axis limits with padding
    xmin = 0.0
    xmax = tmax_days
    ymin = 0.0
    ymax = 0.45 
    xlim = xmin - pad * (xmax - xmin), xmax + pad * (xmax - xmin)
    ylim = ymin - pad * (ymax - ymin), ymax + pad * (ymax - ymin)

    # Make data
    curve_data = {
        "days"     : t_days[::sample_every],
        "Area_mm"  : A_t[::sample_every],
        "sqrtA_mm" : sqrtA_t[::sample_every],
    } 

    # Plot
    points = hv.Scatter(
        agg_data,
        kdims=["days"],
        vdims=["sqrtA_mm_mean"],
        label=r"$in$ $vitro$",
    ).opts(
        c="k",
        s=20,
        marker="o",
    )

    curve = hv.Curve(
        curve_data,
        kdims=["days"],
        vdims=["sqrtA_mm"],
        label=r"$in$ $silico$",
    ).opts(
        c=cc.gray[100],
        linewidth=2,
        linestyle="dashed",
    )
    
    errors = hv.ErrorBars(
        agg_data,
        kdims=["days"],
        vdims=["sqrtA_mm_mean", "sqrtA_mm_std"],
    ).opts(
        edgecolor="k",
        linewidth=1,
        capsize=1,
    )
    
    overlay = hv.Overlay(
        [curve, points, errors]
    ).opts(
        xlabel="Days",
        xlim=xlim,
        xticks=(0, 2, 4, 6, 8),
        ylabel=r"$\sqrt{Area}$ ($mm$)",
#        ylabel=r"$\sqrt{Area}$ (norm.)",
        ylim=ylim,
        yticks=(0, 0.1, 0.2, 0.3, 0.4),
        aspect=1.3,
        fontscale=1.3,
        legend_position="right",
    )
    
    if save:

        # Print update and save
        _fpath = fpath + suffix + "." + fmt
        print(f"Writing to: {_fpath}")
        hv.save(overlay, fpath + suffix, fmt=fmt, dpi=dpi)

main(
    save=True,
    pad=0.05,
    sample_every=20,
)
