import numpy as np
import pandas as pd

import holoviews as hv

from itertools import combinations
from statannotations.Annotator import Annotator

hv.extension("matplotlib")

import matplotlib.pyplot as plt
import seaborn as sns

import lateral_signaling as lsig

lsig.viz.default_rcParams()

data_fname = lsig.analysis_dir.joinpath("FACS_brightfield/cell_shape_metrics.csv")
save_dir = lsig.plot_dir


def main(
    data_fname=data_fname,
    save_dir=save_dir,
    figsize=(9, 5),
    n_bins=5,
    significance_test="Mann-Whitney",
    save=False,
    fmt="png",
    dpi=300,
):

    aggdf = pd.read_csv(data_fname)

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

    # Line/marker plotting options
    colors = plt.get_cmap("gray")(np.linspace(0, 0.7, 4))[::-1]
    linestyles = np.tile(["solid", "dotted", "dashed", "dashdot"], 2)

    # Package plotting options
    ecdf_ax_kw = dict(
        xlim=(0, 1),
        xticks=(0, 0.25, 0.5, 0.75, 1.0),
        ylabel="Cumulative distrib.",
        ylim=(-0.05, 1.05),
        yticks=(0, 0.25, 0.5, 0.75, 1.0),
    )
    hist_ax_kw = dict(
        xlim=(0, 1),
        xticks=np.linspace(0, 1, 6),
        ylabel="Frequency",
        # ylim=(-0.05, 1.05),
        # yticks=(0, 0.25, 0.5, 0.75, 1.0),
    )
    hist_kw = dict(
        histtype="bar",
        color=colors,
        density=False,
    )
    ecdf_kw = dict(
        color="k",
        linewidth=1.5,
    )

    ## Plot circularity
    # Make figure
    prows = 2
    pcols = 3
    fig = plt.figure(figsize=figsize)

    # Get histogram bins
    hist_bins = np.linspace(0, 1, n_bins + 1)

    # Plot histograms
    # ax0 = fig.add_subplot(1, 2, 1)
    ax0 = fig.add_subplot(prows, pcols, 1)
    ax0.set(xlabel="Circularity index", **hist_ax_kw)

    plt.hist(circ_data, bins=hist_bins, **hist_kw)

    plt.legend(densities, title="Density", loc="upper left")

    # Plot ECDFs
    # ax1 = fig.add_subplot(1, 2, 2)
    ax1 = fig.add_subplot(prows, pcols, 4)
    ax1.set(xlabel="Circularity index", **ecdf_ax_kw)

    for i, _d in enumerate(circ_data):
        ax1.step(*lsig.ecdf_vals(_d), linestyle=linestyles[i], **ecdf_kw)

    plt.legend(densities, title="Density")

    # plt.tight_layout()

    ## Plot area
    # Edit plotting options
    #    ecdf_kw["xlabel"] = hist_kw["xlabel"] = r"Cell area ($\mathrm{\mu m}^2$)"
    #    ecdf_kw["xlim"] = hist_kw["xlim"] = (0,850)
    #    ecdf_kw["xticks"] = hist_kw["xticks"] = (0, 200, 400, 600)
    del ecdf_ax_kw["xlim"]
    del hist_ax_kw["xlim"]
    del ecdf_ax_kw["xticks"]
    del hist_ax_kw["xticks"]

    hist_bins = n_bins

    # fig = plt.figure(figsize=figsize)

    # ax0 = fig.add_subplot(1, 2, 1)
    ax0 = fig.add_subplot(prows, pcols, 2)
    ax0.set(xlabel=r"Area ($\mu m^2$)", **hist_ax_kw)

    plt.hist(area_data, bins=hist_bins, **hist_kw)

    plt.legend(densities, title="Density", loc="upper right")

    # ax1 = fig.add_subplot(1, 2, 2)
    ax1 = fig.add_subplot(prows, pcols, 5)
    ax1.set(xlabel=r"Area ($\mu m^2$)", **ecdf_ax_kw)

    for i, _d in enumerate(area_data):
        ax1.step(*lsig.ecdf_vals(_d), linestyle=linestyles[i], **ecdf_kw)

    plt.legend(densities, title="Density")

    # plt.tight_layout()

    ## Plot perimeter
    ax0 = fig.add_subplot(prows, pcols, 3)
    ax0.set(xlabel=r"Perimeter ($\mu m$)", **hist_ax_kw)

    plt.hist(perim_data, bins=hist_bins, **hist_kw)

    plt.legend(densities, title="Density", loc="upper right")

    # ax1 = fig.add_subplot(1, 2, 2)
    ax1 = fig.add_subplot(prows, pcols, 6)
    ax1.set(xlabel=r"Perimeter ($\mu m$)", **ecdf_ax_kw)

    for i, _d in enumerate(perim_data):
        ax1.step(*lsig.ecdf_vals(_d), linestyle=linestyles[i], **ecdf_kw)

    plt.legend(densities, title="Density")

    plt.tight_layout()

    if save:
        _fpath = save_dir.joinpath(f"cell_morphology_metrics.{fmt}")
        print("Writing to:", _fpath.resolve().absolute())
        plt.savefig(_fpath, dpi=dpi)

    ## Plot area and circularity as violins
    # Set options for each plot
    metrics = ("area", "circularity")
    labels = (r"Area ($\mu m^2$)", "Circularity")
    ylims = ((0, 850), (-0.05, 1.05))
    ytickss = ([0, 500, 1000], [0.0, 0.5, 1.0])

    # Set colors
    #    colors  = plt.get_cmap("gray")(np.linspace(0.5, 0.9, 4))[::-1]
    #    colors  = [plt.get_cmap("gray")(0.9)]
    colors = ["w"]

    violin_fpath = lambda metric: save_dir.joinpath(f"{metric}_violin_plots.{fmt}")
    for metric, label, ylim, yticks in zip(metrics, labels, ylims, ytickss):

        # Set up figure
        fig = plt.figure(figsize=(4, 3))
        plt.cla()

        #        # Plot box plot
        #        ax = sns.boxplot(
        #            x="density",
        #            y=metric,
        #            data=aggdf,
        ##            scale="width",
        #            palette=colors,
        ##            size=4,
        ##            jitter=0.2,
        #        )
        #
        #        # Plot strip plot
        #        ax = sns.stripplot(
        #            x="density",
        #            y=metric,
        #            data=aggdf,
        ##            scale="width",
        #            size=4,
        #            jitter=0.2,
        #        )

        # Plot violin plot
        violin_data = dict(
            x="density",
            y=metric,
            data=aggdf,
        )
        ax = sns.violinplot(
            **violin_data,
            bw="silverman",
            scale="area",
            palette=colors,
            # inner="point",
            inner="point",
            cut=0.5,
            linewidth=1,
            edgecolor="k",
            s=3,
        )

        xvals = []
        centers = []
        for i, c in enumerate(ax.collections):
            if i % 2 == 0:
                c.set_edgecolor("k")
            else:
                offsets = np.asarray(c.get_offsets())
                jitter = np.random.normal(0, 0.03, size=offsets.shape[0])
                offsets[:, 0] += jitter
                c.set_color("k")
                c.set_sizes([2])
                c.set_offsets(offsets)

                median = np.median(offsets[:, 1])
                mean = np.mean(offsets[:, 1])

                xvals.append(offsets[0, 0])
                centers.append(mean)

        xvals = np.array(xvals)
        centers = np.array(centers)
        plt.hlines(centers, xvals - 0.3, xvals + 0.3, lw=2, ec="gray")

        # Axis options
        plt.ylabel(label)
        plt.ylim(ylim)
        plt.xlabel("Density")

        #        # Plot median as a point
        #        sns.scatterplot(
        #            ax=ax,
        #            x="Condition",
        #            y="Median",
        #            data=violin_median_df,
        #            color=lsig.viz.black,
        #            s=50,
        #            edgecolor="k",
        #            linewidth=1,
        #        )
        #
        #        # Plot cell-wise activation cutoff
        #        ax.hlines(cutoff, *ax.get_xlim(), lw=2, linestyle="dashed", ec="gray")
        #
        #        # Set axis limits
        #        plt.xlim((-0.75, n_data_idx - 0.25))
        #        plt.ylim((0, 1100))
        #        plt.yticks([0, 250, 500, 750, 1000])
        #
        #        # Keep ticks but remove labels
        #        plt.xlabel("")
        #        ax.tick_params(labelbottom=False)
        #
        #        # Set font sizes
        #        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        #            label.set_fontsize(14)

        # Remove spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        plt.tight_layout()

        if significance_test is not None:

            # Pairs for comparison testing
            densities = np.unique(aggdf["density"])
            pairs = list(combinations(densities, 2))

            print(pairs)

            annotator = Annotator(ax=ax, pairs=pairs, **violin_data)
            annotator.configure(
                test=significance_test,
                text_format="star",
            )
            # annotator.apply_and_annotate()
            annotator.apply_test()

            # Remove non-significant results (reduces clutter)
            annotator.annotations = [a for a in annotator.annotations if "*" in a.text]
            annotator.annotate()

        plt.yticks(yticks)

        if save:

            _fpath = violin_fpath(metric)
            print("Writing to:", _fpath.resolve().absolute())
            plt.savefig(_fpath, dpi=dpi)


if __name__ == "__main__":
    main(
        save=True,
        save_dir=lsig.temp_plot_dir,
    )
