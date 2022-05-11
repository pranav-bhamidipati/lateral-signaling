import os

import numpy as np
import pandas as pd

import colorcet as cc
import matplotlib.pyplot as plt
import seaborn as sns

from statannotations.Annotator import Annotator

import holoviews as hv
hv.extension("matplotlib")

import lateral_signaling as lsig


data_dir   = os.path.abspath("../data/single_spots")
data_fname = os.path.join(data_dir, "singlespot_timeseries.csv")

save_dir       = os.path.abspath("../plots")
boxplots_fpath = os.path.join(save_dir, "shorttimecourse_velocity_boxplots_")
curves_fpath   = os.path.join(save_dir, "shorttimecourse_curves_")

fmt = "png"
dpi = 300

tmax_days=2
save=False


def main(
    data_fname=data_fname,
    figsize=(6, 4),
    SMALL_SIZE=14, 
    MEDIUM_SIZE=16, 
    BIGGER_SIZE=20,
    tmax_days=2,
    save=False,
    fmt="png",
    dpi=300,
):

    # Read DataFrame from file
    data = pd.read_csv(data_fname)

    # Select relevant conditions
    data = data.loc[data.Condition.str.contains("cell/mm2")]
    data = data.loc[data.days <= tmax_days]

    # Extract condition names
    data["ConditionNum"] = data.Condition.astype("category").cat.codes
    data["ConditionLabel"] = np.array(["1x", "2x", "4x"])[data["ConditionNum"].values]
    conds = np.unique(data.ConditionLabel)

    # Convert area units and take sqrt
    data["Area_mm2"]  = data["area (um2)"] / 1e6
    del data["area (um2)"]
    data["sqrtA_mm"] = np.sqrt(data["Area_mm2"])
    
    
    # Get velocity data
    keep_cols = ["ConditionLabel", "ConditionNum", "SpotNum"]

    multiidx = ()
    day_data = []
    vel_data = []
    for group, _df in data.groupby(keep_cols)[["days", "sqrtA_mm"]]:

        (_, days), (__, lscale) = _df.items()

        vels = []
        for i, day in enumerate(days):
            if day == days.min():
                vels.append(np.nan)
            else:
                i_prev_day = np.where(days == (day - 1.0))[0]
                vel = lscale.values[i] - lscale.values[i_prev_day]
                vels.append(vel[0])

        multiidx += (group,) * len(days)
        day_data += days.to_list()
        vel_data += vels
    
    midx = pd.MultiIndex.from_tuples(multiidx, names=keep_cols)
    vel_df = pd.DataFrame({"days": day_data, "velocity_mmpd": vel_data}, index=midx)
    vel_df = vel_df.reset_index().dropna().sort_values("ConditionNum")
    vel_df = vel_df.groupby(keep_cols)["velocity_mmpd"].agg(np.mean).reset_index()

    # Get summary data
    densities = [g[0] for g in vel_df.groupby("ConditionLabel")]
    summary = vel_df.groupby(["ConditionNum"])["velocity_mmpd"].agg([np.mean, np.std]).round(3).values
    summary_df = pd.DataFrame(
        data=dict(mean_vel_mmpd=summary[:, 0], std_vel_mmpd=summary[:, 1]), 
        index=vel_df["ConditionLabel"].unique(),
    )
    summary_df["mean_pm_std"] = [fr"{m:.3f} $\pm$ {s:.3f}" for d, m, s in zip(densities, *summary_df.values.T)]
    summary_df.index = [s + ":" for s in summary_df.index]

    # Pairs for significance testing
    pairs = [("1x", "2x"), ("2x", "4x"), ("1x", "4x")]

    plotting_data = dict(
        data=vel_df, 
        x="ConditionLabel", 
        y="velocity_mmpd", 
        order=densities, 
        # palette=cmap.colors, 
    )
    
    lsig.default_rcParams(SMALL_SIZE=SMALL_SIZE, MEDIUM_SIZE=MEDIUM_SIZE, BIGGER_SIZE=BIGGER_SIZE)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw=dict(width_ratios=[1.1, 1.]))

    sns.boxplot(ax=ax1, palette=["w"], width=0.5, linewidth=2.5, **plotting_data)
    sns.stripplot(ax=ax1, palette=["k"], facecolors="w", **plotting_data)

    ax1.set(
        xlabel="Density",
        ylabel="Velocity (mm/day)",
    )

    annotator = Annotator(ax=ax1, pairs=pairs, **plotting_data)
    annotator.configure(
        test='Mann-Whitney', 
        text_format='star', 
    )
    annotator.apply_and_annotate()

    plt.sca(ax2)
    plt.axis("off")
    table = plt.table(
        cellText=summary_df.values[:, 2:],
        rowLabels=summary_df.index,
        colLabels=["Mean velocity\n(mm/day)"],
        cellLoc = 'center', 
        rowLoc = 'center',
        # loc='center', 
        bbox=[0.05, 0.3, 1., 0.6],
        edges="open",
    )
    table.set_fontsize(MEDIUM_SIZE)
    
    plt.tight_layout()

    if save:
        _fpath = boxplots_fpath + "." + fmt
        print("Writing to:", _fpath)
        plt.savefig(_fpath, dpi=dpi, format=fmt)

    # Get means and standard deviations
    agg_data = data.groupby(
        ["ConditionLabel", "days"]
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
            agg_data.loc[agg_data["ConditionLabel"] == cond],
            kdims=["days"],
    #            vdims=["sqrtA_mm_mean_norm", "ConditionLabel"],
            vdims=["sqrtA_mm_mean", "ConditionLabel"],
            label=cond,
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
            agg_data.loc[agg_data["ConditionLabel"] == cond],
            kdims=["days"],
    #            vdims=["sqrtA_mm_mean_norm", "ConditionLabel"],
            vdims=["sqrtA_mm_mean", "ConditionLabel"],
            label=cond,
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
            agg_data.loc[agg_data["ConditionLabel"] == cond],
            kdims=["days"],
    #            vdims=["sqrtA_mm_mean_norm", "sqrtA_mm_std_norm"],
            vdims=["sqrtA_mm_mean", "sqrtA_mm_std"],
            label=cond,
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
        _fpath = curves_fpath + "." + fmt
        print(f"Writing to:", _fpath)
        hv.save(overlay, curves_fpath, fmt=fmt, dpi=dpi)


main(
    save=True,
)
