import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import lateral_signaling as lsig

lsig.default_rcParams()


FACS_dir = lsig.data_dir.joinpath("FACS", "perturbations")
metadata_csv = FACS_dir.joinpath("metadata.csv")


def main(
    metadata_csv=metadata_csv,
    FACS_dir=FACS_dir,
    cutoff_figsize=(4.0, 2.5),
    save_dir=lsig.plot_dir,
    save=False,
    fmt="png",
    dpi=300,
):

    # Get DataFrame with info on each FACS sample
    metadata = pd.read_csv(metadata_csv)

    # Read sample data from files
    data = [pd.read_csv(FACS_dir.joinpath(f)).squeeze() for f in metadata.filename]

    # Get indices of reference samples
    ref_states = ("OFF", "ON")
    ref_idx = np.isin(metadata.State, ref_states).nonzero()[0].tolist()

    ## Calculate a fluorescence cutoff using equivalence point of Log-likelihood
    # Number of bins in histogram
    nbins = 100
    data_range = (0, 1000)
    bins = np.linspace(*data_range, nbins + 1)

    # Get indices of reference distributions
    OFF_idx = metadata.State.str.contains("OFF").values.nonzero()[0][0]
    ON_idx = metadata.State.str.contains("ON").values.nonzero()[0][0]
    refdata_idx = np.array([OFF_idx, ON_idx])

    # Get number of observations in each bin
    data_hists = np.array(
        [lsig.data_to_hist(d.values, nbins, data_range)[0] for d in data]
    )

    # Add 1 to each bin to avoid div by 0. Then normalize and take the logarithm
    data_hists_pdf = (data_hists + 1) / np.sum(data_hists + 1, axis=1, keepdims=True)
    data_hists_logpdf = np.log10(data_hists_pdf)

    # Get the difference between (log-)PDFs for reference samples
    OFF_hist_pdf, ON_hist_pdf = data_hists_pdf[refdata_idx]
    LR = ON_hist_pdf / OFF_hist_pdf
    OFF_hist_logpdf, ON_hist_logpdf = data_hists_logpdf[refdata_idx]
    LLR = ON_hist_logpdf - OFF_hist_logpdf

    # To approximate where it crosses from OFF to ON, find where it changes sign from - to +
    where_LR_crosses = np.diff(np.sign(LR - 1)) > 0
    where_LLR_crosses = np.diff(np.sign(LLR)) > 0
    cutoff_idx = where_LLR_crosses.nonzero()[0]
    cutoff = float(bins[cutoff_idx + 1])
    print(f"{cutoff=}")

    ## Make plot showing log-likelihood and cutoff

    # Get center of each bin in distribution
    bin_centers = bins[:-1] + np.diff(bins) / 2

    # Plot Log-likelihood and cutoff line
    fig, ax = plt.subplots(figsize=cutoff_figsize)
    plt.scatter(bin_centers, LR, color="k", marker="o", s=5)
    plt.vlines(
        cutoff,
        *plt.gca().get_ylim(),
        lw=2,
    )

    plt.xlabel("mCherry (AU)")
    plt.ylabel(r"$\frac{\mathrm{Likelihood(ON)}}{\mathrm{Likelihood(OFF)}}$")
    ax.set_yscale("log")
    plt.title("Effect of an observation\non ON/OFF likelihood ratio")
    plt.tight_layout()

    # Save
    if save:
        fpath = save_dir.joinpath(f"LLR_cutoff.{fmt}")
        print("Writing to:", fpath.resolve().absolute())
        plt.savefig(fpath, dpi=dpi)

    ## Make violin plots
    for char in "bcas":

        # Get indices of FACS samples to use in this plot
        data_idx = (
            ref_idx
            + (
                metadata.subplot.str.contains(char)
                & ~np.isin(metadata.State, ref_states)
            )
            .values.nonzero()[0]
            .tolist()
        )
        n_data_idx = len(data_idx)

        # Make DataFrame of individual fluorescence values
        violin_data = [data[j] for j in data_idx]
        data_sizes = [d.size for d in violin_data]
        violin_df = pd.DataFrame(
            dict(
                Condition=np.repeat(metadata.filename[data_idx], data_sizes),
                Fluorescence=np.concatenate(violin_data),
                Color=np.repeat(metadata.Color_lite[data_idx], data_sizes),
            )
        )

        # Make DataFrame with median fluorescence of each sample
        violin_medians = [np.median(d) for d in violin_data]
        violin_median_df = pd.DataFrame(
            dict(
                Condition=metadata.filename[data_idx],
                Median=violin_medians,
                Color=metadata.Color_lite[data_idx],
            )
        )

        # Get color coding
        colors = metadata.Color_lite[data_idx].to_list()

        # Set up figure
        plt.figure(figsize=(0.8 * (n_data_idx + 0.5) + 1.15, 3.5))
        plt.cla()

        # Plot distribution as violin
        ax = sns.violinplot(
            x="Condition",
            y="Fluorescence",
            data=violin_df,
            scale="width",
            palette=colors,
            inner=None,
        )

        # Plot median as a point
        sns.scatterplot(
            ax=ax,
            x="Condition",
            y="Median",
            data=violin_median_df,
            color=lsig.black,
            s=50,
            edgecolor="k",
            linewidth=1,
        )

        # Plot cell-wise activation cutoff
        ax.hlines(cutoff, *ax.get_xlim(), lw=2, linestyle="dashed", ec="gray")

        # Set axis limits
        plt.xlim((-0.75, n_data_idx - 0.25))
        plt.ylim((0, 1100))
        plt.yticks([0, 250, 500, 750, 1000])

        # Keep ticks but remove labels
        plt.xlabel("")
        ax.tick_params(labelbottom=False)

        # Set font sizes
        plt.ylabel("mCherry (AU)", fontsize=16)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(14)

        plt.tight_layout()

        # Remove spines
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        if save:
            _fpath = save_dir.joinpath(f"FACS_violins_{char}.{fmt}")
            print(f"Writing to: {_fpath.resolve().absolute()}")
            plt.savefig(_fpath, dpi=dpi)


if __name__ == "__main__":
    main(
        save=True,
    )
