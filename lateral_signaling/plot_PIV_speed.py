from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lateral_signaling import analysis_dir, plot_dir

# load_dir = analysis_dir.joinpath("piv/raw")
load_dir = analysis_dir.joinpath("piv")
save_dir = plot_dir.joinpath("tmp")


def main(
    load_dir: Path = load_dir,
    figsize: tuple[int] = (8, 6),
    save: bool = False,
    save_dir: Path = save_dir,
    fmt: str = "png",
    dpi: int = 300,
    **kwargs,
):

    # metadata = pd.read_csv(load_dir.joinpath("metadata.csv"))
    # metadata.columns = ["video_file", "video_index", "density_mm2", "replicate"]
    # data = pd.concat([pd.read_csv(f) for f in load_dir.glob("PIV*.csv")])
    # data = pd.merge(data, metadata, on=["video_index"], how="left")

    data = pd.read_csv(load_dir.joinpath("PIV_aggregated.csv"))
    data["time_hrs"] = data.frame.values
    densities = np.array(["1/40x", "1/16x", "1/8x", "1/4x", "1/2x", "1x", "2x", "4x"])
    data["density"] = pd.Categorical(
        densities[data.density_mm2.astype("category").cat.codes],
        categories=densities,
        ordered=True,
    )

    fig = plt.figure(figsize=figsize)

    # sns.lineplot(data=data, x="time_hrs", y="speed_um_hr", hue="density_mm2")
    sns.lineplot(
        data=data,
        x="time_hrs",
        y="speed_um_hr_mean",
        hue="density",
        # errorbar=("ci", 0.95),
        markers=True,
        palette="flare",
        **kwargs,
    )

    plt.xlabel("Time (hrs)")
    plt.ylabel("Mean speed (microns/hr)")

    if save:
        fpath = save_dir.joinpath(f"PIV_speed_vs_time.{fmt}").resolve().absolute()
        print(f"Writing to: {fpath}")
        plt.savefig(fpath, dpi=dpi, bbox_inches="tight")


if __name__ == "__main__":

    main(
        save=True,
    )
