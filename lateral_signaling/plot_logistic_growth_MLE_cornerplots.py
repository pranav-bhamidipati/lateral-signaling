import os
import h5py

import numpy as np
import pandas as pd
from tqdm import tqdm

import lateral_signaling as lsig

lsig.default_rcParams()

import matplotlib.pyplot as plt
import seaborn as sns


# To read
data_dir = os.path.abspath("../data")
MLE_dir = os.path.join(data_dir, "growth_curves_MLE")
data_fname = os.path.join(MLE_dir, "growth_curves.csv")
bs_reps_dump_fpath = os.path.join(
    data_dir, "analysis", "growth_curve_bootstrap_replicates.hdf5"
)

# To write
save_dir = os.path.abspath("../plots/tmp")
corner_fname = lambda t: os.path.join(save_dir, f"MLE_bootstrap_cornerplot_{t}")


def main(
    figsize=(5, 5),
    save=False,
    dpi=300,
    fmt="png",
):

    param_names = ["g", "rho_max", "sigma"]

    # Load bootstrap MLEs of parameters
    treatments = []
    bs_reps_list = []
    with h5py.File(bs_reps_dump_fpath, "r") as f:
        for s in f.keys():
            t = s.split("_")[-1]
            treatments.append(t)
            bs_reps_list.append(np.asarray(f[s]))

    _treatment_name_map = {
        "FG": "FGF2",
        "UN": "Untreated",
        "RI": "ROCK-i",
        "RO": "ROCK-i",
    }
    treatments = [_treatment_name_map[t[:2].upper()] for t in treatments]

    # Corner plots
    for i, (t, bs_reps) in enumerate(zip(treatments, bs_reps_list)):

        bs_reps_df = pd.DataFrame(dict(zip(param_names, bs_reps.T)))

        pg = sns.pairplot(bs_reps_df, kind="hist", corner=True)
        # pg = sns.PairGrid(bs_reps_df, diag_sharey=False)
        # pg.map_lower(sns.histplot)
        # pg.map_diag(sns.kdeplot)

        pg.fig.set_size_inches(*figsize)

        plt.suptitle(t)
        plt.tight_layout()

        if save:

            fname = corner_fname(t) + "." + fmt
            print("Writing to:", fname)
            plt.savefig(fname, dpi=dpi, facecolor="w")


if __name__ == "__main__":
    main(
        # save=True,
    )
