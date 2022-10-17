import h5py
import pandas as pd
import numpy as np
import holoviews as hv

hv.extension("matplotlib")

import lateral_signaling as lsig

lsig.viz.default_rcParams()


FACS_dir = lsig.data_dir.joinpath("FACS", "perturbations")
metadata_with_results_csv = lsig.analysis_dir.joinpath(
    "FACS_perturbations_LLR_results.csv"
)
LLR_v_nbins_data_file = lsig.analysis_dir.joinpath(
    "FACS_perturbations_LLR_vs_nbins.hdf5"
)


def main(
    metadata_with_results_csv=metadata_with_results_csv,
    LLR_v_nbins_data_file=LLR_v_nbins_data_file,
    transparent=True,
    save_dir=lsig.plot_dir,
    save=False,
    fmt="png",
    dpi=300,
):

    metadata_with_results = pd.read_csv(metadata_with_results_csv)

    # Select and re-order samples for display
    idx_to_plot = metadata_with_results.subplot.str.contains("d").values.nonzero()[0]
    nrows = metadata_with_results.shape[0]
    nplot = idx_to_plot.size

    _permute = np.arange(nplot)
    _permute[1:] += 6
    _permute[-6:] = np.arange(1, 7)
    idx_to_plot = idx_to_plot[_permute]

    # Get LLR as a function of number of histogram bins
    with h5py.File(LLR_v_nbins_data_file, "r") as f:
        nbins_range = np.asarray(f["nbins_range"])
        llrs_v_nbins = np.asarray(f["llrs_v_nbins"])

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
    ]

    # Plot LLR as a function of # bins used for each sample
    llr_nbins_plot = hv.Overlay(
        [
            hv.Curve((nbins_range, llr), label=label).opts(c=clr, linestyle=ls)
            for llr, label, clr, ls in zip(
                llrs_v_nbins.T,
                metadata_with_results.Label,
                metadata_with_results.Color_shaded,
                metadata_with_results.Linestyle,
            )
            # Add a line showing log-ratio of zero (equal likelihood)
        ]
        + [hv.HLine(0)]
    ).opts(*nbins_opts)

    if save:
        fpath = save_dir.joinpath(f"FACS_llr_vs_num_bins.{fmt}")
        print("Writing to:", fpath.resolve().absolute())
        hv.save(llr_nbins_plot, fpath, dpi=dpi, transparent=transparent)


if __name__ == "__main__":
    main(
        # save_dir=lsig.temp_plot_dir,
        # save=True,
    )
