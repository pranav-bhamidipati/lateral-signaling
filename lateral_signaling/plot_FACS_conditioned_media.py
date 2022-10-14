import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

import lateral_signaling as lsig


FACS_dir = lsig.data_dir.joinpath("FACS", "conditioned_media")


def main(
    FACS_dir=FACS_dir,
    annotate=False,
    figsize=(4, 4),
    SMALL_SIZE=14,
    MEDIUM_SIZE=16,
    BIGGER_SIZE=20,
    save_dir=lsig.plot_dir,
    save=False,
    dpi=300,
    fmt="png",
):

    metadata = pd.read_csv(FACS_dir.joinpath("metadata.csv"))
    mdata_cols = metadata.columns
    dfs = []
    for (cols,) in zip(metadata.values):
        f = next(FACS_dir.glob(f"*{cols[0]}*.csv"))
        _dat = {mc: c for mc, c in zip(mdata_cols, cols)}
        _dat["GFP"] = pd.read_csv(f)["FITC-A"]
        _dat["mCherry"] = pd.read_csv(f)["PE-Texas Red-A"]
        dfs.append(pd.DataFrame(_dat))

    data = pd.concat(dfs)

    median_data = data.groupby("id")["mCherry"].agg(np.median).reset_index()
    median_data.columns = ["id", "MedianmCherry"]

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
        y="mCherry",
        order=metadata.id.values,
    )
    scatter_data = dict(
        data=median_data,
        x="id",
        y="MedianmCherry",
        edgecolor="k",
        facecolor="w",
        linewidth=1.5,
    )

    lsig.default_rcParams(
        SMALL_SIZE=SMALL_SIZE, MEDIUM_SIZE=MEDIUM_SIZE, BIGGER_SIZE=BIGGER_SIZE
    )

    fig, ax1 = plt.subplots(figsize=figsize)
    sns.violinplot(ax=ax1, palette=["w"], inner=None, **violin_data)
    for coll in ax1.collections:
        coll.set_edgecolor("k")

    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()

    sns.scatterplot(ax=ax1, **scatter_data)
    ax1.set(
        xlabel="",
        xlim=xlim,
        ylabel="mCherry (AU)",
        ylim=ylim,
    )

    if annotate:
        annotator = Annotator(ax=ax1, pairs=pairs, **violin_data)
        annotator.configure(
            test="Mann-Whitney",
            text_format="star",
        )
        annotator.apply_and_annotate()

    ax1.set_xticks([-1, 0, 1, 2, 3])
    ax1.set_xticklabels(["Cells:\nMedia:", "1x\n1x", "1x\n4x", "4x\n1x", "4x\n4x"])

    plt.title("Cells w/ conditioned media")

    plt.tight_layout()

    if save:
        _fname = save_dir.joinpath(f"conditioned_media_violins.{fmt}")
        print(f"Writing to: {_fname.resolve().absolute()}")
        plt.savefig(_fname, dpi=dpi)


if __name__ == "__main__":
    main(
        # save_dir=lsig.temp_plot_dir,
        # save=True,
        # annotate=True,
    )
