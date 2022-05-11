import lateral_signaling as lsig

import os
from glob import glob

import numpy as np
import pandas as pd
import scipy.stats as st

import colorcet as cc
import matplotlib.pyplot as plt
import seaborn as sns

lsig.default_rcParams()

# I/O
data_dir      = os.path.abspath("../data/FACS")
sender_data   = glob(os.path.join(data_dir, "senders", "*.csv"))
receiver_data = glob(os.path.join(data_dir, "receivers", "*.csv"))

save_dir      = os.path.abspath("../plots")
violin_fname  = os.path.join(save_dir, "sender_receiver_violins")
contour_fname = os.path.join(save_dir, "sender_receiver_contours")

densities     = ["0.25x", "0.5x", "1x", "2x", "3x", "4x"]
data_cols     = ["FITC-A", "APC-A", "FSC-A"]
kde_densities = [0, 2, 5]


def main(
    sender_data=sender_data,
    receiver_data=receiver_data,
    densities=densities,
    data_cols=data_cols,
    kde_densities=kde_densities,
    data_dir=data_dir,
    save_dir=save_dir,
    cmap=cc.palette.glasbey_category10,
    save=False,
    dpi=300,
    fmt="png",
):

    num_cols = ["GFP (AU)", "mCherry (AU)", "FSC"]
    trf_cols = [r"Log_10(GFP)", r"Log_10(mCherry)", r"Log_10(FSC^3)"]
    ylims = [
        (2.0,  4.5),
        (None, 3.5),
        (None, None),
        (1.5,  3.5),
        (1.25, None),
        (None, None),
    ]

    # Sort alphabetically (increasing density)
    sender_data.sort()
    receiver_data.sort()

    dfs = []
    for i, csv in enumerate(sender_data):
        df = pd.read_csv(csv)
        df = df[data_cols]
        df.columns = num_cols
        df["density"] = densities[i]
        dfs.append(df)
    sender_df = pd.concat(dfs).reset_index(drop=True)

    dfs = []
    for i, csv in enumerate(receiver_data):
        df = pd.read_csv(csv)
        df = df[data_cols]
        df.columns = num_cols
        df["density"] = densities[i]
        dfs.append(df)
    receiver_df = pd.concat(dfs).reset_index(drop=True)

    # Filter negatives
    sender_df   = sender_df.loc[(sender_df[num_cols] > 0).all(axis=1)]
    receiver_df = receiver_df.loc[(receiver_df[num_cols] > 0).all(axis=1)]

    # Transformations for plotting
    for i, (ncol, tcol) in enumerate(zip(num_cols, trf_cols)):

        if i == (len(trf_cols) - 1):
            # Volume scales with FSC^3
            _tfunc = lambda x: np.log10(np.power(x, 3))
        else:
            _tfunc = np.log10

        sender_df[tcol]   = sender_df[ncol].apply(_tfunc)
        receiver_df[tcol] = receiver_df[ncol].apply(_tfunc)

    # Medians 
    sender_medians = {
        "Median"+col: [
            np.median(sender_df.loc[sender_df["density"] == d, col]) for d in densities
        ]
        for col in num_cols + trf_cols
    }
    sender_medians["density"] = densities
    sender_dfm = pd.DataFrame(sender_medians)

    receiver_medians = {
        "Median"+col: [
            np.median(receiver_df.loc[receiver_df["density"] == d, col]) for d in densities
        ]
        for col in num_cols + trf_cols
    }
    receiver_medians["density"] = densities
    receiver_dfm = pd.DataFrame(receiver_medians)

    violin_opts = dict(
        inner=None, 
        cut=0, 
        bw=0.05,
    )
    scatter_opts = dict(
        color=lsig.black,
        s=50,
        edgecolor="k",
        linewidth=1,
    )

    # Plot distribution as violin
    fig, axs = plt.subplots(2, 3, figsize=(10, 6), sharex=False)

    for i, ax in enumerate(axs.flat):

        _data, _Mdata, _y, _ylim  = (
            (sender_df,   sender_dfm,   trf_cols[0], ylims[0]),
            (sender_df,   sender_dfm,   trf_cols[1], ylims[1]),
            (sender_df,   sender_dfm,   num_cols[2], ylims[2]),
            (receiver_df, receiver_dfm, trf_cols[0], ylims[3]),
            (receiver_df, receiver_dfm, trf_cols[1], ylims[4]),
            (receiver_df, receiver_dfm, num_cols[2], ylims[5]),
        )[i]

        sns.violinplot(ax=ax, x="density", y=_y, data=_data, cmap=cmap, **violin_opts)
        ax.set(ylim=_ylim, xlabel="")
        sns.scatterplot(ax=ax, x="density", y="Median"+_y, data=_Mdata, cmap=cmap, **scatter_opts)

        ax.set_ylabel(_y)
        ax.set_xlim(-0.5, len(densities) - 0.5)

    plt.tight_layout()

    if save:
        _fname = violin_fname + "." + fmt
        print("Writing to:", _fname)
        plt.savefig(_fname, dpi=dpi, format=fmt)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    plot_dens = [densities[i] for i in kde_densities]
    colors    = [cmap[i] for i in kde_densities]
    palette   = {d: c for d, c in zip(plot_dens, colors)}

    samples_mask = np.isin(sender_df["density"], plot_dens)
    _data = sender_df.loc[samples_mask, :]

    for ax in axs:
        for dens, _clr in zip(plot_dens[::-1], colors[::-1]):

            dens_data = _data.loc[_data["density"] == dens]
            sns.kdeplot(
                ax=ax,
                data=dens_data.copy(),
                x=trf_cols[0],
                y=trf_cols[2],
                hue="density",
                # palette={dens: _clr},
                palette=palette,
                levels=20,
                fill=True,
                alpha=0.4,
                thresh=0.2,
                legend=False,
            )
            sns.kdeplot(
                ax=ax,
                data=dens_data.copy(),
                x=trf_cols[0],
                y=trf_cols[2],
                hue="density",
                # palette={dens: _clr},
                palette=palette,
                levels=5,
                legend=False,
            )
    
    plt.tight_layout()

    if save:
        _fname = contour_fname + "." + fmt
        print("Writing to:", _fname)
        plt.savefig(_fname, dpi=dpi, format=fmt)


main(save=True)
