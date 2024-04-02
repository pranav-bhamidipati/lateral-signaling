import h5py
import pandas as pd
import numpy as np
import holoviews as hv

import matplotlib.pyplot as plt

from analyze_LLR_vs_n_bins import get_data_and_reference_idx

hv.extension("matplotlib")

import lateral_signaling as lsig

lsig.viz.default_rcParams()


def main(
    metadata_csvs,
    LLR_v_nbins_hdfs,
    FACS_dir,
    nbins=100,
    reg=1e-2,
    undo_log=True,
    min_val=None,
    max_val=None,
    nbins_used=1000,
    log=True,
    min_decision_bound=-np.inf,
    max_decision_bound=np.inf,
    figsize=(4, 3.5),
    save_dir=lsig.plot_dir,
    transparent=True,
    save=False,
    fmt="png",
    dpi=300,
    suffix="",
):

    metadatas = [pd.read_csv(csv) for csv in metadata_csvs]

    # Select and re-order samples for display
    idx_for_plots = []
    for mdata in metadatas:
        if "subplot" in mdata.columns:
            idx_to_plot = mdata["subplot"].str.contains("d").values.nonzero()[0]
            nplot = idx_to_plot.size
            _permute = np.arange(nplot)
            _permute[1:] += 6
            _permute[-6:] = np.arange(1, 7)
            idx_to_plot = idx_to_plot[_permute]
        else:
            idx_to_plot = np.arange(len(mdata))
        idx_for_plots.append(idx_to_plot)

    for csv, hdf, idx in zip(metadata_csvs, LLR_v_nbins_hdfs, idx_for_plots):
        # Get LLR as a function of number of histogram bins
        with h5py.File(hdf, "r") as f:
            nbins_range = np.asarray(f["nbins_range"])
            llrs_v_nbins = f["llrs_v_nbins"][()]

        # Set plot options
        nbins_opts = [
            hv.opts.Overlay(
                aspect=2,
                legend_position="right",
                # legend_position="bottom",
                logx=True,
                xlabel="# histogram bins",
                ylabel=r"Log-Likelihood$\left[\,\frac{\mathrm{signaling\; ON}}{\mathrm{signaling\; OFF}}\,\right]$",
            ),
            hv.opts.Curve(
                linewidth=1,
            ),
            hv.opts.HLine(
                c="k",
                linestyle="dotted",
                linewidth=1,
            ),
            hv.opts.VLine(
                c="gray",
                linestyle="solid",
                linewidth=1,
            ),
        ]

        # Plot LLR as a function of # bins used for each sample
        curve_data = []
        curve_labels = []
        curve_opts = []
        for i in idx:
            label = mdata["Label"][i]
            llr = llrs_v_nbins[:, i]
            curve_data.append((nbins_range, llr))
            curve_labels.append(label)
            opts = {}
            if "Color_shaded" in mdata.columns:
                opts["c"] = mdata["Color_shaded"][i]
            if "Linestyle" in mdata.columns:
                opts["linestyle"] = mdata["Linestyle"][i]
            curve_opts.append(opts)

        # Add a line showing log-ratio of zero (equal likelihood)
        fig = plt.figure(figsize=figsize)
        llr_nbins_plot = hv.Overlay(
            [
                hv.Curve(data, label=lbl).opts(**copts)
                for data, lbl, copts in zip(curve_data, curve_labels, curve_opts)
            ]
            + [hv.HLine(0)]
            + [hv.VLine(nbins_used)]
        ).opts(*nbins_opts)

        plt.tight_layout()
        if save:
            fpath = save_dir.joinpath(f"{csv.stem}_llr_vs_num_bins{suffix}.{fmt}")
            print("Writing to:", fpath.resolve().absolute())
            hv.save(llr_nbins_plot, fpath, dpi=dpi, transparent=transparent)

        ## Plot the difference between the ON and OFF distributions
        # Read in data
        data, OFF_arr, ON_arr = get_data_and_reference_idx(csv, FACS_dir, undo_log)
        if min_val is None:
            min_val = np.min([d.min() for d in data])
        if max_val is None:
            max_val = np.max([d.max() for d in data])

        OFF_hist = np.histogram(OFF_arr, nbins, range=(min_val, max_val))[0]
        ON_hist = np.histogram(ON_arr, nbins, range=(min_val, max_val))[0]

        regularization = np.ones(nbins, dtype=np.float64) / nbins
        OFF_pdf = OFF_hist / np.sum(OFF_hist)
        OFF_pdf = (1 - reg) * OFF_pdf + reg * regularization
        OFF_logpdf = np.log10(OFF_pdf)

        ON_pdf = ON_hist / np.sum(ON_hist)
        ON_pdf = (1 - reg) * ON_pdf + reg * regularization
        ON_logpdf = np.log10(ON_pdf)

        diff_logpdf = ON_logpdf - OFF_logpdf
        bin_edges = np.linspace(min_val, max_val, nbins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.axhline(0, color="gray", lw=1)
        ax.plot(bin_centers, diff_logpdf, c="k")

        if log:
            ax.set_xlabel(r"Log$_{10}$(mCherry (AU))")
        else:
            ax.set_xlabel("mCherry (AU)")
        ax.set_title("Effect of an observation on LLR")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_ylabel(
            r"$\mathrm{Log}_{10}\left[\frac{\mathrm{ON distribution}}{\mathrm{OFF distribution}}\right]$"
        )

        # Label the decision boundary with a vertical line
        where_min = np.searchsorted(bin_centers, min_decision_bound)
        decision_mask = (bin_centers > min_decision_bound) & (
            bin_centers < max_decision_bound
        )
        where_crosses_zero = np.where(np.diff(np.sign(diff_logpdf[decision_mask])))[0]
        if len(where_crosses_zero) > 0:
            idx_crosses_zero = where_min + where_crosses_zero[0]
            x_before = bin_centers[idx_crosses_zero]
            x_after = bin_centers[idx_crosses_zero + 1]
            y_before = diff_logpdf[idx_crosses_zero]
            y_after = diff_logpdf[idx_crosses_zero + 1]
            x_crossing = x_before - y_before * (x_after - x_before) / (
                y_after - y_before
            )
            xbias = 0.02 * (max_val - min_val)
            ax.axvline(x_crossing)
            ax.text(
                x_crossing - xbias,
                ax.get_ylim()[1] * 0.95,
                "Decision boundary",
                rotation=90,
                va="top",
                ha="right",
                fontsize=10,
            )

        plt.tight_layout()
        if save:
            fpath = save_dir.joinpath(f"{csv.stem}_FACS_diff_logpdf{suffix}.{fmt}")
            print("Writing to:", fpath.resolve().absolute())
            plt.savefig(fpath, dpi=dpi)


if __name__ == "__main__":

    # FACS_dir = lsig.data_dir.joinpath("FACS/perturbations")
    # metadata_csvs = [
    #     lsig.analysis_dir.joinpath("FACS_perturbations_LLR_results.csv"),
    # ]
    # LLR_v_nbins_hdfs = [
    #     lsig.analysis_dir.joinpath("FACS_perturbations_LLR_vs_nbins.hdf5")
    # ]
    # min_decision_bound = 300
    # max_decision_bound = 600
    # log = False

    FACS_dir = lsig.data_dir.joinpath("FACS/2024_mESC_and_L929")
    metadata_csvs = [
        lsig.analysis_dir.joinpath("240326_metadata_L929_LLR_results.csv"),
        lsig.analysis_dir.joinpath("240326_metadata_mESC_LLR_results.csv"),
    ]
    LLR_v_nbins_hdfs = [
        lsig.analysis_dir.joinpath("240326_metadata_L929_LLR_results_vs_nbins.hdf5"),
        lsig.analysis_dir.joinpath("240326_metadata_mESC_LLR_results_vs_nbins.hdf5"),
    ]
    min_decision_bound = 1.5
    max_decision_bound = 4.5
    log = True

    main(
        metadata_csvs=metadata_csvs,
        LLR_v_nbins_hdfs=LLR_v_nbins_hdfs,
        FACS_dir=FACS_dir,
        undo_log=False,
        nbins_used=1000,
        min_decision_bound=min_decision_bound,
        max_decision_bound=max_decision_bound,
        log=log,
        save=True,
        fmt="pdf",
    )
