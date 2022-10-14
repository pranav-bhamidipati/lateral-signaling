import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
import lateral_signaling as lsig
lsig.default_rcParams()


def main(
    data_csv=lsig.data_dir.joinpath("aggregation/ligand_aggregation_data.csv"),
    seed=2021,
    n_bs_reps = int(1e7),
    sizes = None,
    save_dir=lsig.data_dir.joinpath("analysis"),
    save_data=False,
):
    rg = np.random.default_rng(seed=seed)
    
    df = pd.read_csv(data_csv)
    
    df["Density"] = pd.Categorical(df["Density"], ordered=True)
    categories = df.Density.cat.categories
    n_cats = len(categories)
    df["Diameter_um"] = 2 * lsig.area_to_radius(df.Area_um2.values)

    
    def _draw_one_bootstrap_median(i, rep):
        bs_rep = rg.choice(data_list[i], sizes[i], replace=True)
        return np.median(bs_rep)
    
    data_list = [g.values for _, g in df.groupby("Density")["Diameter_um"]]
    if sizes is None:
        sizes = [d.size for d in data_list]
    
    bs_medians = np.zeros((n_cats, n_bs_reps), dtype=np.float64)
    pbar = tqdm(total=n_cats * n_bs_reps)
    for i in range(n_cats):
        for rep in range(n_bs_reps):
            bs_medians[i, rep] = _draw_one_bootstrap_median(i, rep)
            pbar.update(1)

    if save_data:
        save_file = save_dir.joinpath("ligand_aggregation_bootstrap_replicates.hdf5")
        with h5py.File(save_file, "w") as f:
            for cat, median_data in zip(categories, bs_medians):
                f.create_dataset(cat, data=median_data)
            

if __name__ == "__main__":
    main(
        save_data=True,
    )