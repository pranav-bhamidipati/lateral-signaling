from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

import lateral_signaling as lsig
from lateral_signaling.lateral_signaling import MultithreadedBootstrap
lsig.default_rcParams()


def main(
    data_csv=lsig.data_dir.joinpath("aggregation/ligand_aggregation_data.csv"),
    stats_method="bootstrap",
    n_bs_reps = int(1e6),
    save_dir=lsig.plot_dir,
    save=False,
    dpi=300,
    fmt="png",
):
    df = pd.read_csv(data_csv)
    df["Density"] = pd.Categorical(df["Density"], ordered=True)
    df["Diameter_um"] = 2 * lsig.area_to_radius(df.Area_um2.values)
    x = "Density"
    # y = "Area_um2"
    y = "Diameter_um"

    order = ["1x", "2x", "4x"]
    pairs = list(combinations(order, 2))

    ax = sns.swarmplot(data=df, x=x, y=y, order=order)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if stats_method == "statannotation":
        annotator = Annotator(ax, pairs, data=df, x=x, y=y, order=order)
        annotator.configure(test="Mann-Whitney", text_format="star", loc="outside")
        annotator.apply_and_annotate()
    
    elif stats_method == "bootstrap":
        
        categories = df.Density.categories
        bs_medians = np.zeros((len(categories), n_bs_reps), dtype=np.float64)
        for i, cat in enumerate(categories):
            data = df.loc[df["Density"] == cat, y].values
            mtb = MultithreadedBootstrap(data, n_bs_reps, seed=2021 + i)
            mtb.draw_bootstraps()
            bs_medians[i] = np.median(mtb.values, axis=1)
        
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
