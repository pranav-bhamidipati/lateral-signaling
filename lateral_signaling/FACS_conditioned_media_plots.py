import lateral_signaling as lsig

import os
from glob import glob

import numpy as np
import pandas as pd
import scipy.stats as st

import colorcet as cc
import matplotlib.pyplot as plt
import seaborn as sns

from statannotations.Annotator import Annotator


# I/O
data_dir       = os.path.abspath("../data/FACS/conditioned_media")
metadata_fname = os.path.join(data_dir, "metadata.csv")

save_dir = os.path.abspath("../plots")
fname    = os.path.join(save_dir, "conditioned_media_violins")


def main(
    data_dir=data_dir,
    metadata_fname=metadata_fname,
    fname=fname,
    figsize=(4,4),
    SMALL_SIZE=14,
    MEDIUM_SIZE=16,
    BIGGER_SIZE=20,
    save=False,
    dpi=300,
    fmt="png",
):
    
    metadata = pd.read_csv(metadata_fname)
    mdata_cols = metadata.columns
    dfs = []
    for (cols,) in zip(metadata.values):
        f = glob(os.path.join(data_dir, f"*{cols[0]}*.csv"))[0]
        _dat = {mc: c for mc, c in zip(mdata_cols, cols)}
        _dat["GFP"] = pd.read_csv(f)["FITC-A"]
        _dat["TexasRed"] = pd.read_csv(f)["PE-Texas Red-A"]
        dfs.append(pd.DataFrame(_dat))

    data = pd.concat(dfs)
    
    median_data = data.groupby("id")["TexasRed"].agg(np.median).reset_index()
    median_data.columns = ["id", "MedianTexasRed"]

    # Pairs for significance testing
    pairs = [
        ("d1", "d2"), 
        ("d1", "d3"), 
        ("d1", "d4"), 
        ("d2", "d3"), 
        ("d2", "d4"), 
        ("d3", "d4"), 
    ]

    violin_data = dict(
        data=data, 
        x="id", 
        y="TexasRed", 
        order=metadata.id.values, 
    )
    scatter_data = dict(
        data=median_data, 
        x="id", 
        y="MedianTexasRed", 
        edgecolor="k", 
        facecolor="w",
        linewidth=1.5,
    )

    lsig.default_rcParams(SMALL_SIZE=SMALL_SIZE, MEDIUM_SIZE=MEDIUM_SIZE, BIGGER_SIZE=BIGGER_SIZE)
    
#    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw=dict(width_ratios=[1.1, 1.]))
    fig, ax1 = plt.subplots(figsize=figsize)

    sns.violinplot(ax=ax1, palette=["w"], inner=None, **violin_data)
    for coll in ax1.collections:
        coll.set_edgecolor("k")
    
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()

    sns.scatterplot(ax=ax1, **scatter_data)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

#    sns.boxplot(ax=ax1, palette=["w"], **plotting_data)

    annotator = Annotator(ax=ax1, pairs=pairs, **violin_data)
    annotator.configure(
        test='Mann-Whitney', 
        text_format='star', 
    )
#    annotator.apply_and_annotate()

    ax1.set(
        xlabel="",
        ylabel="TexasRed (AU)",
    )

#    plt.sca(ax2)
#    plt.axis("off")
#    table = plt.table(
#        cellText=data.values[:, 1:],
#        rowLabels=data.index,
#        colLabels=["Mean velocity\n(mm/day)"],
#        cellLoc = 'center', 
#        rowLoc = 'center',
#        # loc='center', 
#        bbox=[0.05, 0.3, 1., 0.6],
#        edges="open",
#    )
#    table.set_fontsize(MEDIUM_SIZE)
    
    plt.tight_layout()

    if save:
        _fname = fname + "." + fmt
        print("Writing to:", _fname)
        plt.savefig(_fname, dpi=dpi, format=fmt)
    
    plt.tight_layout()


main(
    save=True,
)
