import os
from glob import glob
import json
import h5py

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import lateral_signaling as lsig

# Reading
data_dir = os.path.abspath("../data/simulations")
sacred_dir = os.path.join(data_dir, "20211201_singlespotphase/sacred")
thresh_fpath = os.path.join(data_dir, "phase_threshold.json")

# Writing
save_dir = os.path.abspath("../plots/tmp")
v_init_fname = os.path.join(save_dir, "v_init_histogram")
n_act_fin_fname = os.path.join(save_dir, "n_act_fin_histogram")


def main(
    figsize=(3.5, 2.5),
    nbins=22,
    binmin=1,
    binmax=5000,
    save=False,
    fmt="png",
    dpi=300,
):

    # Get threshold for v_init
    with open(thresh_fpath, "r") as f:
        threshs = json.load(f)
        v_init_thresh = float(threshs["v_init_thresh"])

    # Read in phase metric data
    run_dirs = glob(os.path.join(sacred_dir, "[0-9]*"))

    # Store each run's data in a DataFrame
    dfs = []
    for rd_idx, rd in enumerate(run_dirs):

        _config_file = os.path.join(rd, "config.json")
        _results_file = os.path.join(rd, "results.hdf5")

        if (not os.path.exists(_config_file)) or (not os.path.exists(_results_file)):
            continue

        # Get some info from the run configuration
        with open(_config_file, "r") as c:
            config = json.load(c)

            # Initial density, carrying capacity
            rho_0 = config["rho_0"]
            rho_max = config["rho_max"]

        # Get remaining info from run's data dump
        with h5py.File(_results_file, "r") as f:

            # Phase metrics
            v_init = np.asarray(f["v_init_g"])
            n_act_fin = np.asarray(f["n_act_fin_g"])

            # Proliferation rates and time-points
            if rd_idx == 0:
                g = list(f["g_space"])
                t = np.asarray(f["t"])

        # Assemble dataframe
        _df = pd.DataFrame(
            dict(
                v_init=v_init,
                n_act_fin=n_act_fin,
                g=g,
                rho_0=rho_0,
                rho_max=rho_max,
            )
        )
        dfs.append(_df)

    # Concatenate into one dataset
    df = pd.concat(dfs).reset_index(drop=True)
    nrow = df.shape[0]
    df["activates"] = df["v_init"] > v_init_thresh
    df["deactivates"] = df["n_act_fin"] > 0

    # Get pct of runs
    pct_off = (~df.deactivates).sum() / nrow
    pct_on = df.activates.sum() / nrow

    ## Plot n_act distribution
    fig1 = plt.figure(1, figsize=figsize)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    sns.histplot(
        df,
        x="n_act_fin",
        multiple="stack",
        hue="deactivates",
        palette=plt.get_cmap("Pastel1").colors,
        legend=False,
        bins=nbins,
    )

    plt.xlabel(r"$n_\mathrm{act}$ ($t=8\,\mathrm{days}$)", fontsize=16)
    plt.ylabel("# Simulations", fontsize=16)
    plt.tick_params(labelsize=12)

    # Plot pie chart of runs with zero/nonzero active cells at the end
    # pielabels = (r"$=0$", r"$>0$")
    pielabels = ("", "")
    sizes = (pct_off, 1 - pct_off)
    explode = (0.0, 0.0)

    plt.text(450, 6000, r"$n_\mathrm{act}=0$", fontsize=14, color="red")

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xrange = xmax - xmin
    yrange = ymax - ymin

    # Make inset axis
    axins = ax.inset_axes(
        [xmin + 0.4 * xrange, ymin + 0.3 * yrange, 0.9 * xrange, 0.8 * yrange],
        transform=ax.transData,
    )
    axins.pie(
        sizes,
        explode=explode,
        labels=pielabels,
        colors=plt.get_cmap("Pastel1").colors,
        autopct="%1.1f%%",
    )

    plt.tight_layout()

    if save:

        _fpath = str(n_act_fin_fname)
        if not _fpath.endswith(fmt):
            _fpath += "." + fmt
        print("Writing to:", _fpath)
        plt.savefig(_fpath, dpi=dpi)

    ## Plot histogram of v_init
    fig2 = plt.figure(2, figsize=figsize)
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    sns.histplot(
        df,
        x="v_init",
        multiple="stack",
        hue="activates",
        palette=plt.get_cmap("Pastel1").colors,
        legend=False,
        bins=nbins,
    )

    # Threshold used
    _xlim = ax.get_xlim()
    _xrange = _xlim[1] - _xlim[0]
    plt.vlines(
        v_init_thresh,
        *plt.gca().get_ylim(),
        color=lsig.darker_gray,
        lw=2,
        linestyles="dashed",
    )
    plt.text(
        v_init_thresh + 0.05 * _xrange,
        5000,
        r"$v_{\mathrm{thresh}}$",
        fontsize=16,
        color=lsig.darker_gray,
    )

    plt.xlabel(r"$v_{init}$", fontsize=16)
    plt.ylabel("# Simulations", fontsize=16)
    plt.tick_params(labelsize=12)

    plt.tight_layout()

    # Plot pie chart of runs with zero/nonzero active cells at the end
    # pielabels = (r"$=0$", r"$>0$")
    pielabels = ("", "")
    sizes = (pct_on, 1 - pct_on)
    explode = (0.0, 0.0)

    plt.text(450, 6000, r"$v_\mathrm{init}=0$", fontsize=14, color="red")

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xrange = xmax - xmin
    yrange = ymax - ymin

    # Make inset axis
    axins = ax.inset_axes(
        [xmin + 0.35 * xrange, ymin + 0.25 * yrange, 0.75 * xrange, 0.65 * yrange],
        transform=ax.transData,
    )
    axins.pie(
        sizes,
        explode=explode,
        labels=pielabels,
        colors=plt.get_cmap("Pastel1").colors,
        autopct="%1.1f%%",
    )

    if save:

        _fpath = str(v_init_fname)
        if not _fpath.endswith(fmt):
            _fpath += "." + fmt
        print("Writing to:", _fpath)
        plt.savefig(_fpath, dpi=dpi)


if __name__ == "__main__":
    main(
        save=True,
    )
