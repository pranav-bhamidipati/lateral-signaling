import pandas as pd
from lateral_signaling import data_dir, analysis_dir, area_to_radius


def _draw_one_bootstrap_median_value(rg, values, labels, _=None):
    return pd.Series(rg.permutation(values)).groupby(labels).median()


def main(
    data_csv=data_dir.joinpath("aggregation", "ligand_aggregation_data.csv"),
    seed=2021,
    n_bs_reps=int(1e7),
    progress=True,
    save_dir=analysis_dir,
    save=False,
):
    from tqdm import tqdm
    from functools import partial
    from psutil import cpu_count

    # from tqdm import trange
    import numpy as np
    import multiprocessing as mp

    rg = np.random.default_rng(seed=seed)

    categories = ["1x", "2x", "4x"]

    df = pd.read_csv(data_csv)
    df["Density"] = pd.Categorical(df["Density"], ordered=True, categories=categories)
    df = df.sort_values("Density")
    labels = df.Density

    diameters = 2 * area_to_radius(df.Area_um2.values)

    prog = tqdm if progress else lambda a, *_: a
    results = []
    get_bs_rep = partial(_draw_one_bootstrap_median_value, rg, diameters, labels)

    with mp.Pool(cpu_count(logical=True)) as pool:
        for result in prog(
            pool.imap_unordered(get_bs_rep, range(n_bs_reps), chunksize=20),
            total=n_bs_reps,
        ):
            results.append(result)

    bs_medians = pd.DataFrame({i: bs for i, bs in enumerate(results)}).T
    bs_medians.columns = categories

    if save:
        save_csv = save_dir.joinpath("ligand_aggregation_bootstrap_medians.csv")
        print("Writing to:", save_csv.resolve().absolute())
        bs_medians.to_csv(save_csv, index=False)


if __name__ == "__main__":
    main(
        seed=2021,
        n_bs_reps=int(1e7),
        save=True,
    )
