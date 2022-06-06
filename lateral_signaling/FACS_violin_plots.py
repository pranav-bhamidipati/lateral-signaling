import os

import pandas as pd
import numpy as np

import colorcet as cc
import cmocean.cm as cmo

import holoviews as hv
import matplotlib.pyplot as plt
import seaborn as sns

import lateral_signaling as lsig


# Inputs
data_dir       = os.path.abspath("../data/FACS")
metadata_fpath = os.path.join(data_dir, "metadata.csv")

# Outputs
save_dir   = os.path.abspath("../plots")
violin_pfx = os.path.join(save_dir, "FACS_violins_")
LLR_fname  = os.path.join(save_dir, "LLR_cutoff_")

def main(
    violin_prefix=violin_pfx,
    save=False,
    fmt="png",
    dpi=300,
):

    # Get DataFrame with info on each FACS sample
    metadata = pd.read_csv(metadata_fpath)

    # Read sample data from files
    data = [pd.read_csv(os.path.join(data_dir, f)).squeeze() for f in metadata.filename]

    # Get indices of reference samples
    ref_states = ("OFF", "ON")
    ref_idx = np.isin(metadata.State, ref_states).nonzero()[0].tolist()

    ## Calculate a fluorescence cutoff using equivalence point of Log-likelihood
    # Number of bins in histogram
    nbins      = 100
    data_range = (0, 1000)
    bins       = np.linspace(*data_range, nbins + 1)

    # Get indices of reference distributions
    OFF_idx     = metadata.State.str.contains("OFF").values.nonzero()[0][0]
    ON_idx      = metadata.State.str.contains("ON").values.nonzero()[0][0]
    refdata_idx = np.array([OFF_idx, ON_idx])

    # Get number of observations in each bin
    data_hists = np.array([lsig.data_to_hist(d.values, nbins, data_range)[0] for d in data])

    # Add 1 to each bin to avoid div by 0. Then normalize and take the logarithm
    data_hists_pdf    = (data_hists + 1) / np.sum(data_hists + 1, axis=1, keepdims=True)
    data_hists_logpdf = np.log10(data_hists_pdf)

    # Get the difference between log(PDF)s for reference samples
    OFF_hist_logpdf, ON_hist_logpdf = data_hists_logpdf[refdata_idx]
    LLR = ON_hist_logpdf - OFF_hist_logpdf

    # To approximate where it crosses from OFF to ON, find where it changes sign from - to +
    where_LLR_crosses = np.diff(np.sign(LLR)) > 0
    cutoff_idx = where_LLR_crosses.nonzero()[0]
    cutoff = (bins[cutoff_idx + 1])

    ## Make plot showing log-likelihood and cutoff

    # Get center of each bin in distribution
    bin_centers = bins[:-1] + np.diff(bins) / 2

    # Plot Log-likelihood and cutoff line
    plt.scatter(bin_centers, LLR, color="k", marker="o", s=5)
    plt.vlines(cutoff, *plt.gca().get_ylim(), lw=1,)

    # Options
    plt.xlabel("mCherry (AU)")
    plt.ylabel(r"$\mathrm{Log}_{10}P(\mathrm{ON}) - \mathrm{Log}_{10}P(\mathrm{OFF})$")

    # Save
    if save:
        fpath = LLR_fname + "." + fmt
        print("Writing to:", fpath)
        plt.savefig(fpath, dpi=dpi)

    ## Make violin plots
    
    # Output filenames
    violin_fpath = lambda char: f"{violin_prefix}{char}_.{fmt}"

    for char in "bcas":
        
        # Get indices of FACS samples to use in this plot
        data_idx = ref_idx + (
            metadata.subplot.str.contains(char) &~ np.isin(metadata.State, ref_states)
        ).values.nonzero()[0].tolist()
        n_data_idx = len(data_idx)
        
        # Make DataFrame of individual fluorescence values
        violin_data = [data[j] for j in data_idx]
        data_sizes =  [d.size for d in violin_data]
        violin_df = pd.DataFrame(dict(
            Condition    = np.repeat(metadata.filename[data_idx], data_sizes),
            Fluorescence = np.concatenate(violin_data),
            Color        = np.repeat(metadata.Color_lite[data_idx], data_sizes),
        ))
        
        # Make DataFrame with median fluorescence of each sample
        violin_medians = [np.median(d) for d in violin_data]
        violin_median_df = pd.DataFrame(dict(
            Condition    = metadata.filename[data_idx],
            Median       = violin_medians,
            Color        = metadata.Color_lite[data_idx],
        ))
        
        # Get color coding
        colors = metadata.Color_lite[data_idx].to_list()
        
        # Set up figure
        plt.figure(figsize = (0.8 * (n_data_idx + 0.5) + 1.15, 3.5))    
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
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(14)
        
        plt.tight_layout()
        
        # Remove spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Save
        if save:
            _fpath = violin_fpath(char)
            print("Writing to:", _fpath)
            plt.savefig(_fpath, dpi=dpi)


main(
    save=True,
)
