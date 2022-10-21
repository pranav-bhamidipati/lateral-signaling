from itertools import combinations
from typing import Literal

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

import lateral_signaling as lsig

lsig.viz.default_rcParams()


def main(
    data_csv=lsig.data_dir.joinpath("aggregation/ligand_aggregation_data.csv"),
    figsize=(3, 3),
    save_dir=lsig.plot_dir,
    save=False,
    dpi=300,
    fmt="png",
):

    order = ["1x", "2x", "4x"]

    df = pd.read_csv(data_csv)
    df["Density"] = pd.Categorical(df["Density"], ordered=True, categories=order)

    # df["Diameter_um"] = 2 * lsig.area_to_radius(df.Area_um2.values)
    # x = "Density"
    # y = "Diameter_um"

    x = "Density"
    y = "Area"

    fig = plt.figure(figsize=figsize)

    ax = sns.swarmplot(data=df, x=x, y=y, order=order, s=3)

    pairs = list(combinations(order, 2))
    annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order)

    annotator.configure(test="Mann-Whitney", text_format="star", loc="outside")
    annotator.apply_and_annotate()

    sns.despine()
    plt.tight_layout()

    if save:
        _fpath = save_dir.joinpath(f"ligand_aggregation_swarmplot.{fmt}")
        print("Writing to:", _fpath.resolve().absolute())
        plt.savefig(_fpath, dpi=dpi)


if __name__ == "__main__":
    main(
        # save_dir=lsig.temp_plot_dir,
        save=True,
    )
