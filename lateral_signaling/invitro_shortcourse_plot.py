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

data_fname = os.path.abspath("../data/single_spots/singlespot_timeseries.csv")
save_dir   = os.path.abspath("../plots")
fpath      = os.path.join(save_dir, "invitro_shortcourse_sqrtarea_")
fmt        = "png"
dpi        = 300

def main(
    fpath=fpath,
    tmax_days=2,
    save=False,
    fmt=fmt,
    dpi=dpi,
):

    # Read DataFrame from file
    data = pd.read_csv(data_fname)

    # Select relevant conditions
    data = data.loc[data.Condition.str.contains("cell/mm2")]
    data = data.loc[data.days <= tmax_days]

    # Extract condition names
    conds = np.unique(data.Condition)
    
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
   
    # Axis limits with padding
    pad  = 0.05

    xmin = 0.0
    xmax = tmax_days
    ymin = 0.0
    ymax = 0.45
    
    xlim = xmin - pad * (xmax - xmin), xmax + pad * (xmax - xmin)
    ylim = ymin - pad * (ymax - ymin), ymax + pad * (ymax - ymin)

    # Set colors/linestyles/markers
    ccycle = lsig.sample_cycle(cc.gray[:150], conds.size)
#    ccycle = hv.Cycle(lsig.cols_blue)
    lstyle = hv.Cycle(["solid", "dashed", "dotted"])
    mcycle = hv.Cycle(["o", "s", "^"])

    points = [
        hv.Scatter(
            agg_data.loc[agg_data.Condition == cond],
            kdims=["days"],
#            vdims=["sqrtA_mm_mean_norm", "Condition"],
            vdims=["sqrtA_mm_mean", "Condition"],
            label=("1x", "2x", "4x")[i],
        ).opts(
            c=ccycle.values[i],
#            c="k",
            s=60,
            marker=mcycle.values[i],
#            facecolors=(0, 0, 0, 0),
            edgecolors=ccycle.values[i],
        )
        for i, cond in enumerate(conds)
    ]

    curves = [
        hv.Curve(
            agg_data.loc[agg_data.Condition == cond],
            kdims=["days"],
#            vdims=["sqrtA_mm_mean_norm", "Condition"],
            vdims=["sqrtA_mm_mean", "Condition"],
            label=("1x", "2x", "4x")[i],
        ).opts(
            c=ccycle.values[i],
#            c="k",
            linewidth=2,
            linestyle=lstyle.values[i],
        )
        for i, cond in enumerate(conds)
    ]

    errors = [
        hv.ErrorBars(
            agg_data.loc[agg_data.Condition == cond],
            kdims=["days"],
#            vdims=["sqrtA_mm_mean_norm", "sqrtA_mm_std_norm"],
            vdims=["sqrtA_mm_mean", "sqrtA_mm_std"],
            label=("1x", "2x", "4x")[i],
        ).opts(
            edgecolor=ccycle.values[i],
#            edgecolor="k",
            linewidth=1,
            capsize=2,
        )
        for i, cond in enumerate(conds)
    ]
    
    overlay = hv.Overlay(
        [c_ * p_ * e_ for e_, c_, p_ in zip(errors, curves, points,)]
    ).opts(
        xlabel="Days",
        xlim=xlim,
        xticks=(0, 1, 2),
        ylabel=r"$\sqrt{Area}$ ($mm$)",
#        ylabel=r"$\sqrt{Area}$ (norm.)",
        ylim=ylim,
        yticks=(0, 0.1, 0.2, 0.3, 0.4),
        fontscale=1.3,
#        show_legend=False,
        legend_position="right",
    )
    
    if save:
        
        # Print update and save
        _fpath = fpath + "." + fmt
        print(f"Writing to: {_fpath}")
        hv.save(overlay, fpath, fmt=fmt, dpi=dpi)


main(
    save=True,
    tmax_days=2,
)
