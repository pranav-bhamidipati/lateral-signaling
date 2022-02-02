import lateral_signaling as lsig

import os
from glob import glob
import json
import h5py

import numpy as np
import pandas as pd

import colorcet as cc
import holoviews as hv
hv.extension("matplotlib")

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Reading 
data_dir = os.path.abspath("../data/imaging/FACS_brightfield/")
rois_fname = os.path.join(data_dir, "cell_boundary_vertices.csv")

# Writing
save_dir = os.path.abspath("../plots")
plot_fpath = os.path.join(save_dir, "cell_morphology_metrics")

def main(
#    figsize=(8, 3),
    figsize=(9, 9),
    n_bins=5,
    area_cutoff=100,
    save=False,
    fmt="png",
    dpi=300,
):

    # Read cell boundary data
    df = pd.read_csv(rois_fname, index_col=0)

    # Calculate metrics on data
    aggdfs = []
    for (r, w), d in df.groupby(["roi", "window"]):

        # Extract density condition
        dens = d.density.unique()

        # GEt metrics from ROI polygon vertices
        verts = d[["x", "y"]].values
        area = lsig.shoelace_area(verts)
        perimeter = lsig.perimeter(verts)
        circularity = lsig.circularity(verts)

        # Filter out ROIs that are too small (erroneous ROI)
        if area < area_cutoff:
            continue
        else:
            aggdfs.append(
                pd.DataFrame(
                    dict(
                        roi=r,
                        window=w,
                        density=dens,
                        area=area,
                        perimeter=perimeter,
                        circularity=circularity,
                    )
                )
            )

    aggdf = pd.concat(aggdfs)

    # # Set options for scatterplot
    # points_kw = dict(
    #     xlim = (0, None),
    #     ylim = (0, None),
    # )

    # overlay_kw = dict(
    #     show_legend=True,
    #     legend_position="right",
    # )

    # # Plot perimeter vs. area
    # hv.Points(
    #     data=aggdf,
    #     kdims=["area", "perimeter"],
    #     vdims=["density"]
    # ).groupby(
    #     "density"
    # ).opts(
    #     **points_kw
    # ).overlay(
    # ).opts(
    #     **overlay_kw
    # )

    # Get morphology data as a funciton of density
    densities = [g[0] for g in aggdf.groupby("density")]
    circ_data = [g[1].circularity.values for g in aggdf.groupby("density")]
    area_data = [g[1].area.values for g in aggdf.groupby("density")]
    perim_data = [g[1].perimeter.values for g in aggdf.groupby("density")]

    ## Some plotting options
    # Font sizes
    SMALL_SIZE  = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    # Set font sizes
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Line/marker plotting options
    colors = plt.get_cmap("gray")(np.linspace(0, 0.7, 4))[::-1]
    linestyles = np.tile(["solid", "dotted", "dashed", "dashdot"], 2)

    # Package plotting options
    ecdf_kw = dict(
        xlabel="Circularity index",
        xlim=(0, 1),
        xticks=(0, 0.25, 0.5, 0.75, 1.0),
        ylabel="Cumulative distrib.",
        ylim=(-0.05, 1.05),
        yticks=(0, 0.25, 0.5, 0.75, 1.0),
    )
    hist_kw = dict(
        xlabel="Circularity index",
        xlim=(0, 1),
        xticks=np.linspace(0, 1, 6),
        ylabel="Frequency",
        # ylim=(-0.05, 1.05),
        # yticks=(0, 0.25, 0.5, 0.75, 1.0),
    )

    ## Plot circularity
    # Make figure
    prows = 3
    pcols = 2
    fig = plt.figure(figsize=figsize)

    # Get histogram bins
    hist_bins = np.linspace(0, 1, n_bins + 1)

    # Plot histograms
    # ax0 = fig.add_subplot(1, 2, 1)
    ax0 = fig.add_subplot(prows, pcols, 1)
    ax0.set(**hist_kw)

    plt.hist(circ_data, bins=hist_bins, histtype="bar", color=colors, density=False)

    plt.legend(densities, title="Density", loc="upper left")

    # Plot ECDFs
    # ax1 = fig.add_subplot(1, 2, 2)
    ax1 = fig.add_subplot(prows, pcols, 2)
    ax1.set(**ecdf_kw)

    for i, _d in enumerate(circ_data):
        ax1.step(*lsig.ecdf_vals(_d), color="k", linestyle=linestyles[i], linewidth=1.5)

    plt.legend(densities, title="Density")

    # plt.tight_layout()


    ## Plot area
    # Edit plotting options
    ecdf_kw["xlabel"] = hist_kw["xlabel"] = r"Cell area ($\mathrm{px}^2$)"
    ecdf_kw["xlim"] = hist_kw["xlim"] = (0, 6000)
    ecdf_kw["xticks"] = hist_kw["xticks"] = (0, 2000, 4000, 6000)

    hist_bins = n_bins

    # fig = plt.figure(figsize=figsize)

    # ax0 = fig.add_subplot(1, 2, 1)
    ax0 = fig.add_subplot(prows, pcols, 3)
    ax0.set(**hist_kw)

    plt.hist(area_data, bins=hist_bins, histtype="bar", color=colors, density=False)

    plt.legend(densities, title="Density", loc="upper right")

    # ax1 = fig.add_subplot(1, 2, 2)
    ax1 = fig.add_subplot(prows, pcols, 4)
    ax1.set(**ecdf_kw)

    for i, _d in enumerate(area_data):
        ax1.step(*lsig.ecdf_vals(_d), color="k", linestyle=linestyles[i], linewidth=1.5)

    plt.legend(densities, title="Density")

    # plt.tight_layout()


    ## Plot perimeter
    # Edit plotting options
    ecdf_kw["xlabel"] = hist_kw["xlabel"] = r"Cell perimeter ($\mathrm{px}$)"
    # ecdf_kw["xlim"] = hist_kw["xlim"] = (0, 6000)
    # ecdf_kw["xticks"] = hist_kw["xticks"] = (0, 2000, 4000, 6000)
    del ecdf_kw["xlim"]
    del hist_kw["xlim"]
    del ecdf_kw["xticks"]
    del hist_kw["xticks"]

    # fig = plt.figure(figsize=figsize)

    # ax0 = fig.add_subplot(1, 2, 1)
    ax0 = fig.add_subplot(prows, pcols, 5)
    ax0.set(**hist_kw)

    plt.hist(perim_data, bins=hist_bins, histtype="bar", color=colors, density=False)

    plt.legend(densities, title="Density", loc="upper right")

    # ax1 = fig.add_subplot(1, 2, 2)
    ax1 = fig.add_subplot(prows, pcols, 6)
    ax1.set(**ecdf_kw)

    for i, _d in enumerate(perim_data):
        ax1.step(*lsig.ecdf_vals(_d), color="k", linestyle=linestyles[i], linewidth=1.5)

    plt.legend(densities, title="Density")

    plt.tight_layout()
    
    
    # Save
    if save:
        _fpath = plot_fpath + "." + fmt
        print("Writing to:", _fpath)
        plt.savefig(_fpath, dpi=dpi)


main(
    save = True,
)


