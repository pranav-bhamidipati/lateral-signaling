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
    stats_method: Literal["bootstrap", "statsannotation"] = "bootstrap",
    bs_medians_csv=lsig.anaysis_dir.joinpath(
        "ligand_aggregation_bootstrap_medians.csv"
    ),
    save_dir=lsig.plot_dir,
    save=False,
    dpi=300,
    fmt="png",
):
    df = pd.read_csv(data_csv)
    df["Density"] = pd.Categorical(df["Density"], ordered=True)
    df["Diameter_um"] = 2 * lsig.area_to_radius(df.Area_um2.values)

    x = "Density"
    y = "Diameter_um"
    order = ["1x", "2x", "4x"]

    ax = sns.swarmplot(data=df, x=x, y=y, order=order)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pairs = list(combinations(order, 2))
    annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order)

    if stats_method == "statannotation":
        annotator.configure(test="Mann-Whitney", text_format="star", loc="outside")
        annotator.apply_and_annotate()

    elif stats_method == "bootstrap":
        bs_medians_df = pd.read_csv(bs_medians_csv)

        ## Get p-values for each pair

        ## Draw using Annotator

    plt.tight_layout()

    if save:
        _fpath = save_dir.joinpath(f"ligand_aggregation_swarmplot.{fmt}")
        print("Writing to:", _fpath.resolve().absolute())
        plt.savefig(_fpath, dpi=dpi)


if __name__ == "__main__":
    main(
        save_dir=lsig.temp_plot_dir,
        save=True,
    )
