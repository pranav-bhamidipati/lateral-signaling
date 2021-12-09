import os

import pandas as pd
import numpy as np

import colorcet as cc
import cmocean.cm as cmo

import holoviews as hv
import matplotlib.pyplot as plt
import seaborn as sns

import lateral_signaling as lsig


# Parameters for saving figures/data
save_figs = True
fmt       = "png"
dpi       = 300

# Set directories for I/O
data_dir  = os.path.abspath("../data/FACS_data")
plot_dir  = os.path.abspath("../plots")

# Set filenames for I/O
metadata_fname   = os.path.join(data_dir, "metadata.csv")
violin_fname     = lambda char: os.path.join(plot_dir, f"FACS_violins_{char}." + fmt)

# Get DataFrame with info on each FACS sample
metadata = pd.read_csv(metadata_fname)

# Read sample data from files
data = [pd.read_csv(os.path.join(data_dir, f)).squeeze() for f in metadata.filename]

# Get indices of reference samples
ref_states = ("OFF", "ON")
ref_idx = np.isin(metadata.State, ref_states).nonzero()[0].tolist()

# Make 3 plots
for char in "bca":
    
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
    
    if save_figs:
        fname = violin_fname(char)
        print(fname)
        plt.savefig(fname, dpi=dpi)

