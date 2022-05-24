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
sender_cols   = ["FITC-A", "APC-A", "FSC-A"]
receiver_cols = ["FITC-A", "PE-Texas Red-A", "FSC-A"]


def main(
    kde_densities,
    sender_data=sender_data,
    receiver_data=receiver_data,
    densities=densities,
    sender_cols=sender_cols,
    receiver_cols=receiver_cols,
    data_dir=data_dir,
    save_dir=save_dir,
    cmap=cc.palette.glasbey_category10,
    save=False,
    save_violin=False,
    save_kde=False,
    dpi=300,
    fmt="png",
):

    sender_colnames = ["GFP (AU)", "FRFP (AU)", "FSC (AU)"]
    sender_transformed_colnames = [r"Log_10(GFP)", r"Log_10(FRFP)", r"Log_10(FSC^3)"]
    sender_transformed_colnames = [r"Log_10(GFP)", r"Log_10(FRFP)", r"Log_10(FSC^3)"]
    receiver_colnames = ["GFP (AU)", "mCherry (AU)", "FSC (AU)"]
    receiver_transformed_colnames = [r"Log_10(GFP)", r"Log_10(mCherry)", r"Log_10(FSC^3)"]
    
    # Some samples need a cutoff for fluorescence (extremelly high values)
    sender_fluor_cutoffs = np.array([
        15000,
        np.inf,
        np.inf,
    ])
    receiver_fluor_cutoffs = np.array([
        1200,
        np.inf,
        np.inf,
    ])

    titles = [
        "Sender",
        "Sender",
        "Sender",
        "Receiver",
        "Receiver",
        "Receiver",
    ]

    ylims = [
        (2.0, 4.25),
        (None, 4000),
        (None, None),
        (1.5, 3.25),
        (1.0, 5.0),
        (None, None),
    ]

    # Sort alphabetically (increasing density)
    sender_data.sort()
    receiver_data.sort()

    dfs = []
    for i, csv in enumerate(sender_data):
        df = pd.read_csv(csv)
        df = df[sender_cols]
        df.columns = sender_colnames
        df["density"] = densities[i]
        dfs.append(df)
    sender_df = pd.concat(dfs).reset_index(drop=True)

    dfs = []
    for i, csv in enumerate(receiver_data):
        df = pd.read_csv(csv)
        df = df[receiver_cols]
        df.columns = receiver_colnames
        df["density"] = densities[i]
        dfs.append(df)
    receiver_df = pd.concat(dfs).reset_index(drop=True)

    # Filter negative values
    sender_df   = sender_df.loc[(sender_df[sender_colnames] > 0).all(axis=1)]
    receiver_df = receiver_df.loc[(receiver_df[receiver_colnames] > 0).all(axis=1)]

    # Filter spurious high values
    sender_df = sender_df.loc[(sender_df[sender_colnames] < sender_fluor_cutoffs).all(axis=1), :]
    receiver_df = receiver_df.loc[(receiver_df[receiver_colnames] < receiver_fluor_cutoffs).all(axis=1), :]

    # Transformations for plotting
    def _tfunc(i):
        if i == (len(sender_transformed_colnames) - 1):
            return lambda x: np.log10(np.power(x, 3)) 
        else:
            return np.log10

    for i, (col, tcol) in enumerate(zip(sender_colnames, sender_transformed_colnames)):
        sender_df[tcol] = sender_df[col].apply(_tfunc(i))

    for i, (col, tcol) in enumerate(zip(receiver_colnames, receiver_transformed_colnames)):
        receiver_df[tcol] = receiver_df[col].apply(_tfunc(i))

    # Medians 
    sender_medians = {
        "Median"+col: [
            np.median(sender_df.loc[sender_df["density"] == d, col]) for d in densities
        ]
        for col in sender_colnames + sender_transformed_colnames 
    }
    sender_medians["density"] = densities
    sender_dfm = pd.DataFrame(sender_medians)

    receiver_medians = {
        "Median"+col: [
            np.median(receiver_df.loc[receiver_df["density"] == d, col]) for d in densities
        ]
        for col in receiver_colnames + receiver_transformed_colnames 
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
        
        row = i // 3
        col = i % 3

        _data = (sender_df, receiver_df)[row]
        _Mdata = (sender_dfm, receiver_dfm)[row]
        _y = (
            sender_transformed_colnames[0],
            sender_colnames[1],
            sender_colnames[2],
            receiver_transformed_colnames[0],
            receiver_transformed_colnames[1],
            receiver_colnames[2],
        )[i]

        sns.violinplot(ax=ax, x="density", y=_y, data=_data, cmap=cmap, **violin_opts)
        ax.set(ylim=ylims[i], xlabel="")
        sns.scatterplot(ax=ax, x="density", y="Median"+_y, data=_Mdata, cmap=cmap, **scatter_opts)

        ax.set_ylabel(_y)
        ax.set_xlim(-0.5, len(densities) - 0.5)
        ax.set_title(titles[i])

    plt.tight_layout()

    if save or save_violin:
        _fname = violin_fname + "." + fmt
        print("Writing to:", _fname)
        plt.savefig(_fname, dpi=dpi, format=fmt)
    
    fig, axs = plt.subplots(2, 2, figsize=(5, 5), sharex=False, sharey=False)
    
    plot_dens = [densities[i] for i in kde_densities]
    colors    = [cmap[i] for i in kde_densities]
    palette   = {d: c for d, c in zip(plot_dens, colors)}

    for i, ax in enumerate(axs.flat):
        
        row = i // 2
        col = i % 2
        _data = (sender_df, receiver_df)[row]
        samples_mask = np.isin(_data["density"], plot_dens)
        _data = _data.loc[samples_mask, :]

        trf_cols = (sender_transformed_colnames, receiver_transformed_colnames)[row]
        
        for dens, _clr in zip(plot_dens[::-1], colors[::-1]):

            dens_data = _data.loc[_data["density"] == dens]
            sns.kdeplot(
                ax=ax,
                data=dens_data.copy(),
                x=trf_cols[2],
                y=trf_cols[col],
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
                x=trf_cols[2],
                y=trf_cols[col],
                hue="density",
                # palette={dens: _clr},
                palette=palette,
                levels=5,
                legend=False,
            )
        
        ax.set_title(titles[3 * row])
    
    plt.tight_layout()

    if save or save_kde:
        _fname = contour_fname + "." + fmt
        print("Writing to:", _fname)
        plt.savefig(_fname, dpi=dpi, format=fmt)


main(
    kde_densities = [2, 5],
    save_kde=True,
)
