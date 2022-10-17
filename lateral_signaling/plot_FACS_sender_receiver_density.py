import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import lateral_signaling as lsig

lsig.viz.default_rcParams()

FACS_dir = lsig.data_dir.joinpath("FACS")


def main(
    kde_densities=[2, 5],
    FACS_dir=FACS_dir,
    densities=["0.25x", "0.5x", "1x", "2x", "3x", "4x"],
    kde_colors=sns.color_palette("colorblind"),
    sender_cols=["FITC-A", "APC-A", "FSC-A"],
    receiver_cols=["FITC-A", "PE-Texas Red-A", "FSC-A"],
    save_dir=lsig.plot_dir,
    save=False,
    save_violin=False,
    save_kde=False,
    dpi=300,
    fmt="png",
):

    sender_colnames = ["GFP (AU)", "FRFP (AU)", "FSC (AU)"]
    sender_transformed_colnames = [r"Log_10(GFP)", r"Log_10(FRFP)", r"Log_10(FSC^3)"]
    receiver_colnames = ["GFP (AU)", "mCherry (AU)", "FSC (AU)"]
    receiver_transformed_colnames = [
        r"Log_10(GFP)",
        r"Log_10(mCherry)",
        r"Log_10(FSC^3)",
    ]

    # Some samples need a cutoff for fluorescence (extremelly high values)
    sender_fluor_cutoffs = np.array(
        [
            15000,
            np.inf,
            np.inf,
        ]
    )
    receiver_fluor_cutoffs = np.array(
        [
            1200,
            np.inf,
            np.inf,
        ]
    )

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
    sender_data = sorted(FACS_dir.joinpath("senders").glob("*.csv"))
    receiver_data = sorted(FACS_dir.joinpath("receivers").glob("*.csv"))

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
    sender_df = sender_df.loc[(sender_df[sender_colnames] > 0).all(axis=1)]
    receiver_df = receiver_df.loc[(receiver_df[receiver_colnames] > 0).all(axis=1)]

    # Filter spurious high values
    sender_df = sender_df.loc[
        (sender_df[sender_colnames] < sender_fluor_cutoffs).all(axis=1), :
    ]
    receiver_df = receiver_df.loc[
        (receiver_df[receiver_colnames] < receiver_fluor_cutoffs).all(axis=1), :
    ]

    ## Transformations for plotting
    # Use cubed forward scattering (FSC scales with diameter, so FSC^3 scales with volume)
    sender_df["FSC^3"] = sender_df["FSC (AU)"] ** 3
    receiver_df["FSC^3"] = receiver_df["FSC (AU)"] ** 3

    # Do other transformations
    def _tfunc(i):
        if i == (len(sender_transformed_colnames) - 1):
            return lambda x: np.log10(np.power(x, 3))
        else:
            return np.log10

    for i, (col, tcol) in enumerate(zip(sender_colnames, sender_transformed_colnames)):
        sender_df[tcol] = sender_df[col].apply(_tfunc(i))

    for i, (col, tcol) in enumerate(
        zip(receiver_colnames, receiver_transformed_colnames)
    ):
        receiver_df[tcol] = receiver_df[col].apply(_tfunc(i))

    # Medians
    sender_medians = {
        "Median"
        + col: [
            np.median(sender_df.loc[sender_df["density"] == d, col]) for d in densities
        ]
        for col in sender_colnames + sender_transformed_colnames
    }
    sender_medians["density"] = densities
    sender_dfm = pd.DataFrame(sender_medians)

    receiver_medians = {
        "Median"
        + col: [
            np.median(receiver_df.loc[receiver_df["density"] == d, col])
            for d in densities
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
        color=lsig.viz.black,
        s=35,
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

        sns.violinplot(
            ax=ax, x="density", y=_y, data=_data, palette=["w"], **violin_opts
        )
        ax.set(ylim=ylims[i], xlabel="")
        sns.scatterplot(
            ax=ax, x="density", y="Median" + _y, data=_Mdata, **scatter_opts
        )

        ax.set_ylabel(_y)
        ax.set_xlim(-0.5, len(densities) - 0.5)
        ax.set_title(titles[i])

    plt.tight_layout()

    if save or save_violin:
        _fname = save_dir.joinpath(f"sender_receiver_violins.{fmt}")
        print("Writing to:", _fname.resolve().absolute())
        plt.savefig(_fname, dpi=dpi)

    nrows = 2
    ncols = 2
    #    fig, axs = plt.subplots(nrows, ncols, figsize=(5, 5), sharex=False, sharey=False)
    fig = plt.figure(figsize=(8, 8))

    kde_xlim = (1e14, None)
    kde_ylims = 10 ** np.array(
        [
            (2.0, 4.2),
            (2.5, 3.5),
            (1.25, 3.2),
            (1.75, 4.8),
        ]
    )

    if save or save_kde:

        plot_dens = [densities[i] for i in kde_densities]
        palette = {d: c for d, c in zip(plot_dens, kde_colors[: len(kde_densities)])}

        for i in range(nrows * ncols):
            ax = fig.add_subplot(nrows, ncols, i + 1)

            row = i // ncols
            col = i % ncols
            _data = (sender_df, receiver_df)[row]
            samples_mask = np.isin(_data["density"], plot_dens)
            _data = _data.loc[samples_mask, :]

            trf_cols = (sender_transformed_colnames, receiver_transformed_colnames)[row]
            _cols = (sender_colnames, receiver_colnames)[row]

            sns.kdeplot(
                ax=ax,
                data=_data,
                x="FSC^3",
                y=_cols[col],
                hue="density",
                palette=palette,
                levels=10,
                linewidths=0.5,
                log_scale=True,
            )
            sns.kdeplot(
                ax=ax,
                data=_data,
                x="FSC^3",
                y=_cols[col],
                hue="density",
                palette=palette,
                levels=10,
                alpha=0.7,
                fill=True,
                log_scale=True,
            )

            ax.legend_.set_title("Density")
            sns.move_legend(ax, "lower left")

            ax.set_title(titles[3 * row])
            ax.set_xlim(kde_xlim)
            ax.set_ylim(kde_ylims[col + 2 * row])

        plt.tight_layout()

        # Save
        _fname = save_dir.joinpath(f"sender_receiver_contours.{fmt}")
        print("Writing to:", _fname.resolve().absolute())
        plt.savefig(_fname, dpi=dpi)


if __name__ == "__main__":
    main(
        save=True,
    )
