import os
from pathlib import Path
import pandas as pd
from pyparsing import Iterable

from lateral_signaling import analysis_dir

load_dir = analysis_dir.joinpath("piv/raw")
save_dir = analysis_dir.joinpath("piv")


def main(
    confidence_intervals: Iterable[float] = (0.8, 0.9),
    load_dir: Path = load_dir,
    save: bool = False,
    save_dir: Path = save_dir,
):

    metadata = pd.read_csv(load_dir.joinpath("metadata.csv"))
    metadata.columns = ["video_file", "video_index", "density_mm2", "replicate"]
    data = pd.concat([pd.read_csv(f) for f in load_dir.glob("PIV*.csv")])

    data = pd.merge(data, metadata, on=["video_index"], how="left")

    quantiles = [0.5 - ci / 2 for ci in sorted(confidence_intervals, reverse=True)]
    quantiles = quantiles + [1 - q for q in reversed(quantiles)]

    means = data.groupby(["density_mm2", "replicate", "frame"])["speed_um_hr"].mean()
    medians = data.groupby(["density_mm2", "replicate", "frame"])["speed_um_hr"].median()
    stds = data.groupby(["density_mm2", "replicate", "frame"])["speed_um_hr"].std()
    covs = stds / means
    quants = [
        data.groupby(["density_mm2", "replicate", "frame"])["speed_um_hr"].quantile(q)
        for q in quantiles
    ]

    aggs = [means, medians, stds, covs, *quants]
    agg_suffixes = ["mean", "median", "std", "cov"] + [
        f"Q{str(round(100 * q)).zfill(2)}" for q in quantiles
    ]
    aggregated = pd.concat(aggs, axis=1)
    aggregated.columns = [
        f"{c}_{sfx}" for c, sfx in zip(aggregated.columns, agg_suffixes)
    ]

    aggregated = aggregated.reset_index()
    aggregated["time_hrs"] = aggregated.frame.values

    if save:
        fpath = save_dir.joinpath("PIV_aggregated.csv").resolve().absolute()
        print(f"Writing to: {fpath}")
        aggregated.to_csv(fpath)


if __name__ == "__main__":

    main(
        save=True,
    )
