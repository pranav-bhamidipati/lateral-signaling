import os
import json
from pathlib import Path
import h5py

import numpy as np
import pandas as pd

import colorcet as cc
import matplotlib.pyplot as plt
import seaborn as sns

from statannotations.Annotator import Annotator

import holoviews as hv

hv.extension("matplotlib")

import lateral_signaling as lsig


data_file = lsig.data_dir.joinpath("single_spots/singlespot_timeseries.csv")
sim_dir = lsig.simulation_dir.joinpath("20220111_constantdensity")
sacred_dir = sim_dir.joinpath("sacred")
save_dir = lsig.plot_dir


def main(
    data_file=data_file,
    sacred_dir=sacred_dir,
    save_dir=save_dir,
    figsize=(4, 3.2),
    SMALL_SIZE=14,
    MEDIUM_SIZE=16,
    BIGGER_SIZE=20,
    tmax_days=2,
    save=False,
    fmt="png",
    dpi=300,
):

    # Read in data from experiments
    run_dirs = list(sacred_dir.glob("[0-9]*"))
    run_dirs = [rd for rd in run_dirs if rd.joinpath("config.json").exists()]
    rd0 = run_dirs[0]

    with rd0.joinpath("config.json").open() as j:
        config = json.load(j)
        delay = config["delay"]
        k = config["k"]

    with h5py.File(str(rd0.joinpath("results.hdf5")), "r") as f:
        t = np.asarray(f["t"])

    # Define data to read
    rhos = []
    S_ts = []
    R_ts = []

    for rd in run_dirs:

        with h5py.File(str(rd.joinpath("results.hdf5")), "r") as f:
            rho = np.asarray(f["rho_t"])[0]  # Constant density
            S_t = np.asarray(f["S_t"])
            R_t = np.asarray(f["R_t"])

        rhos.append(rho)
        S_ts.append(S_t)
        R_ts.append(R_t)

    sort_rhos = np.argsort(rhos)
    rhos = np.asarray(rhos)[sort_rhos]
    S_ts = np.asarray(S_ts)[sort_rhos]
    R_ts = np.asarray(R_ts)[sort_rhos]

    # Get early time-range for plotting
    tmax = tmax_days / lsig.t_to_units(1)
    tmax_idx = np.searchsorted(t, tmax, side="right")
    tslice = slice(tmax_idx)
    nt = t[tslice].size

    # Calculate the activation size
    n_act_ts = (S_ts[:, tslice] > k).sum(axis=2) - 1
    A_ts = np.array([lsig.ncells_to_area(n, rho) for n, rho in zip(n_act_ts, rhos)])
    A_ts_norm = lsig.normalize(A_ts, 0, A_ts.max())
    r_prop_ts = lsig.area_to_radius(A_ts)
    r_prop_ts_norm = lsig.normalize(r_prop_ts, 0, r_prop_ts.max())

    simdata = pd.DataFrame(
        {
            "t": np.tile(t[tslice], len(rhos)),
            "A_t": A_ts.ravel(),
            "A_t_norm": A_ts_norm.ravel(),
            "r_prop_t": r_prop_ts.ravel(),
            "r_prop_t_norm": r_prop_ts_norm.ravel(),
            "Density": np.repeat([fr"$\rho =$ {int(r)}" for r in rhos], nt),
            "ConditionLabel": np.repeat([f"{int(r)}x" for r in rhos], nt),
        }
    )

    simdata_final = simdata.loc[
        simdata["t"] == simdata["t"].max(),
        ["t", "A_t", "r_prop_t", "ConditionLabel"],
    ]
    simdata_final["t_delay"] = simdata_final["t"] / delay
    simdata_final["velocity_mmptau"] = (
        simdata_final["r_prop_t"] / simdata_final["t_delay"]
    )
    simdata_final["ConditionX"] = (
        simdata_final["ConditionLabel"].astype("category").cat.codes + 0.3
    )

    data = pd.read_csv(str(data_file))

    # Select relevant conditions
    data = data.loc[data.Condition.str.contains("cell/mm2")]
    data = data.loc[data.days <= tmax_days]

    # Extract condition names
    data["ConditionNum"] = data.Condition.astype("category").cat.codes
    data["ConditionLabel"] = np.array(["1x", "2x", "4x"])[data["ConditionNum"].values]
    conds = np.unique(data.ConditionLabel)

    # Convert area units and take sqrt
    data["Area_mm2"] = data["area (um2)"] / 1e6
    del data["area (um2)"]

    # Calculate radius
    data["r_prop_mm"] = lsig.area_to_radius(data["Area_mm2"].values)

    # Get velocity data
    keep_cols = ["ConditionLabel", "ConditionNum", "SpotNum"]

    multiidx = ()
    day_data = []
    vel_data = []
    # for group, _df in data.groupby(keep_cols)[["days", "sqrtA_mm"]]:
    for group, _df in data.groupby(keep_cols)[["days", "r_prop_mm"]]:

        (_, days), (__, dist) = _df.items()

        vels = []
        for i, day in enumerate(days):
            if day == days.min():
                vels.append(np.nan)
            else:
                i_prev_day = np.where(days == (day - 1.0))[0]
                vel = dist.values[i] - dist.values[i_prev_day]
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
    summary = (
        vel_df.groupby(["ConditionNum"])["velocity_mmpd"]
        .agg([np.mean, np.std])
        .round(3)
        .values
    )
    summary_df = pd.DataFrame(
        data=dict(mean_vel_mmpd=summary[:, 0], std_vel_mmpd=summary[:, 1]),
        index=pd.Index(vel_df["ConditionLabel"].unique(), name="ConditionLabel"),
    )
    summary_df["mean_pm_std"] = [
        fr"{m:.3f} $\pm$ {s:.3f}" for d, m, s in zip(densities, *summary_df.values.T)
    ]
    summary_df = summary_df.reset_index()
    summary_df.index = [s + ":" for s in summary_df["ConditionLabel"]]

    # Pairs for significance testing
    pairs = [("1x", "2x"), ("2x", "4x"), ("1x", "4x")]

    data_kw = dict(
        data=vel_df,
        x="ConditionLabel",
        y="velocity_mmpd",
        order=densities,
    )
    summary_kw = dict(
        data=summary_df,
        x="ConditionLabel",
        y="mean_vel_mmpd",
    )
    sim_kw = dict(
        data=simdata_final,
        x="ConditionX",
        y="velocity_mmptau",
    )

    lsig.viz.default_rcParams(
        SMALL_SIZE=SMALL_SIZE, MEDIUM_SIZE=MEDIUM_SIZE, BIGGER_SIZE=BIGGER_SIZE
    )

    fig1 = plt.figure(1, figsize=figsize)
    ax1 = plt.gca()

    #    sns.boxplot(ax=ax1, palette=["w"], width=0.5, linewidth=2.5, **data_kw)
    sns.scatterplot(ax=ax1, c=["k"], s=600, alpha=0.5, marker="_", **summary_kw)
    sns.stripplot(ax=ax1, palette=["k"], facecolors="w", **data_kw)

    ax1.set(xlabel="Density", ylabel="Velocity (mm/day)", ylim=(0, None))

    annotator = Annotator(ax=ax1, pairs=pairs, **data_kw)
    annotator.configure(
        test="Mann-Whitney",
        text_format="star",
    )
    annotator.apply_and_annotate()

    twinax1 = ax1.twinx()
    twin_ylim = (
        0,
        ax1.get_ylim()[1]
        * simdata_final["velocity_mmptau"].mean()
        / summary_df["mean_vel_mmpd"].mean(),
    )
    twinax1.set(
        ylabel=r"Velocity (mm/$\tau$)",
        ylim=twin_ylim,
        # yticks=(0, 0.05, 0.1),
    )
    sns.scatterplot(
        ax=twinax1,
        palette=[cc.palette.glasbey_cool[1]],
        lw=0.0,
        marker="d",
        s=125,
        alpha=0.8,
        **sim_kw,
    )

    plt.tight_layout()
    ax1.set(xlim=(-0.5, 2.5))
    ax1.spines["top"].set_visible(False)
    twinax1.spines["top"].set_visible(False)

    if save:
        _fpath = save_dir.joinpath(f"shorttimecourse_velocity_stripplot.{fmt}")
        print("Writing to:", str(_fpath.resolve().absolute()))
        plt.savefig(_fpath, dpi=dpi)

    fig2 = plt.figure(2, figsize=(2.5, 2.5))
    plt.axis("off")
    table = plt.table(
        cellText=summary_df.loc[:, ["mean_pm_std"]].values,
        rowLabels=summary_df.index,
        colLabels=["Mean velocity\n(mm/day)"],
        cellLoc="center",
        rowLoc="center",
        bbox=[0.05, 0.3, 1.0, 0.6],
        edges="open",
    )
    table.set_fontsize(MEDIUM_SIZE)

    plt.tight_layout()

    if save:
        _fpath = save_dir.joinpath(f"shorttimecourse_velocity_table.{fmt}")
        print("Writing to:", _fpath.resolve().absolute())
        plt.savefig(_fpath, dpi=dpi)

    # Get means and standard deviations
    agg_data = (
        data.groupby(["ConditionLabel", "days"]).agg([np.mean, np.std]).reset_index()
    )
    agg_data.columns = ["_".join(col).strip("_") for col in agg_data.columns.values]
    agg_data["r_prop_mm_mean_norm"] = (
        agg_data["r_prop_mm_mean"] / data["r_prop_mm"].max()
    )
    agg_data["r_prop_mm_std_norm"] = agg_data["r_prop_mm_std"] / data["r_prop_mm"].max()

    # Axis limits with padding
    pad = 0.05

    xmin = 0.0
    xmax = tmax_days
    ymin = 0.0
    ymax = 0.23

    xlim = xmin - pad * (xmax - xmin), xmax + pad * (xmax - xmin)
    ylim = ymin - pad * (ymax - ymin), ymax + pad * (ymax - ymin)

    # Styling
    ccycle = lsig.viz.sample_cycle(cc.gray[:150], conds.size)
    lstyle = hv.Cycle(["solid", "dashed", "dotted"])
    mcycle = hv.Cycle(["o", "s", "^"])

    points = [
        hv.Scatter(
            agg_data.loc[agg_data["ConditionLabel"] == cond],
            kdims=["days"],
            vdims=["r_prop_mm_mean", "ConditionLabel"],
            label=cond,
        ).opts(
            c=ccycle.values[i],
            s=60,
            marker=mcycle.values[i],
            edgecolors=ccycle.values[i],
        )
        for i, cond in enumerate(conds)
    ]

    curves = [
        hv.Curve(
            agg_data.loc[agg_data["ConditionLabel"] == cond],
            kdims=["days"],
            vdims=["r_prop_mm_mean", "ConditionLabel"],
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
            vdims=["r_prop_mm_mean", "r_prop_mm_std"],
            label=cond,
        ).opts(
            edgecolor=ccycle.values[i],
            linewidth=1,
            capsize=2,
        )
        for i, cond in enumerate(conds)
    ]

    overlay = hv.Overlay(
        [
            c_ * p_ * e_
            for e_, c_, p_ in zip(
                errors,
                curves,
                points,
            )
        ]
    ).opts(
        xlabel="Days",
        xlim=xlim,
        xticks=(0, 1, 2),
        ylabel=r"$r_\mathrm{prop}$ ($mm$)",
        ylim=ylim,
        yticks=(0, 0.05, 0.1, 0.15, 0.2),
        fontscale=1.3,
        legend_position="right",
    )

    if save:

        _fpath = save_dir.joinpath(f"shorttimecourse_rprop_invitro.{fmt}")
        print(f"Writing to:", _fpath.resolve().absolute())
        hv.save(overlay, _fpath, dpi=dpi)

    ## Plot in silico timeseries

    xmin = 0.0
    xmax = t[tslice][-1]
    xlim = xmin - pad * (xmax - xmin), xmax + pad * (xmax - xmin)

    # Styling
    lcycle = hv.Cycle(["solid", "dashed", "dotted"])
    xticks = [
        (0 * delay, "0"),
        (1 * delay, "τ"),
        (2 * delay, "2τ"),
        (3 * delay, "3τ"),
        (4 * delay, "4τ"),
    ]

    # Plot curves
    curve_plot = (
        hv.Curve(
            simdata,
            kdims=["t"],
            vdims=["r_prop_t", "Density"],
        )
        .groupby(
            "Density",
        )
        .opts(
            xlabel="Simulation time",
            xlim=xlim,
            xticks=xticks,
            ylabel=r"$r_\mathrm{prop}$ ($mm$)",
            ylim=ylim,
            yticks=[0, 0.05, 0.1, 0.15, 0.2],
            linewidth=2,
            linestyle=lcycle,
            color="k",
            aspect=1,
        )
        .overlay("Density")
        .opts(
            fontscale=1.3,
            legend_position="right",
        )
    )

    if save:

        _fpath = save_dir.joinpath(f"shorttimecourse_rprop_insilico.{fmt}")
        print("Writing to:", _fpath.resolve().absolute())
        hv.save(curve_plot, _fpath, fmt=fmt, dpi=dpi)


if __name__ == "__main__":
    main(
        save=True,
        save_dir=lsig.temp_plot_dir,
    )
