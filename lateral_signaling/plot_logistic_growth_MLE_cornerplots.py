import h5py

import numpy as np
import pandas as pd

import lateral_signaling as lsig

lsig.viz.default_rcParams()

import matplotlib.pyplot as plt
import seaborn as sns


def main(
    bootstrap_mle_replicates_hdf,
    treatment_names=None,
    param_names=["g", "rho_max", "sigma"],
    figsize=(5, 5),
    save_dir=lsig.plot_dir,
    save=False,
    dpi=300,
    fmt="png",
):

    # Load bootstrap MLEs of parameters
    treatments = []
    bs_reps_list = []

    def collect_hdf_contents(name_with_prefix, obj):
        if isinstance(obj, h5py.Dataset):
            name = name_with_prefix.removeprefix("bs_reps_")
            if treatment_names is None or name in treatment_names:
                treatments.append(name)
                bs_reps_list.append(np.array(obj))

    with h5py.File(bootstrap_mle_replicates_hdf, "r") as f:
        f.visititems(collect_hdf_contents)

    # _treatment_name_map = {
    #     "FG": "FGF2",
    #     "UN": "Untreated",
    #     "RI": "ROCK-i",
    #     "RO": "ROCK-i",
    # }
    # treatments = [_treatment_name_map[t[:2].upper()] for t in treatments]

    # Corner plots
    for i, (t, bs_reps) in enumerate(zip(treatments, bs_reps_list)):
        fig = plt.figure(figsize=figsize)
        bs_reps_df = pd.DataFrame(dict(zip(param_names, bs_reps.T)))
        pg = sns.pairplot(
            bs_reps_df,
            kind="hist",
            corner=True,
            plot_kws={"bins": "sqrt", "common_norm": False},
            diag_kws={"bins": "sqrt", "common_norm": False},
        )
        # pg = sns.PairGrid(bs_reps_df, diag_sharey=False)
        # pg.map_lower(sns.histplot)
        # pg.map_diag(sns.kdeplot)

        plt.suptitle(t)
        plt.tight_layout()

        if save:
            from datetime import datetime

            today = datetime.today().strftime("%y%m%d")
            t_name = t.replace(" ", "_").replace("/", "_")
            fname = save_dir.joinpath(
                f"{today}_MLE_bootstrap_cornerplot_{t_name}.{fmt}"
            )
            print("Writing to:", fname.resolve().absolute())
            plt.savefig(fname, dpi=dpi, facecolor="w")


if __name__ == "__main__":

    # # Untreated condition (10% FBS)
    # bs_reps_hdf = lsig.analysis_dir.joinpath(
    #     "240327_growth_curve_bootstrap_replicates.hdf5"
    # )
    # # save_dir = lsig.plot_dir.joinpath("cornerplots")
    # treatment_names = ["10% FBS"]
    # param_names = ["g", "rho_max", "sigma"]

    # ROCK-inhibitor and FGF2 conditions
    bs_reps_hdf = lsig.analysis_dir.joinpath(
        "240402_growth_curve_bootstrap_replicates_fixed_rhomax.hdf5"
    )
    # save_dir = lsig.plot_dir.joinpath("fixed_rhomax_cornerplots")
    treatment_names = None
    param_names = ["g", "sigma"]

    main(
        bootstrap_mle_replicates_hdf=bs_reps_hdf,
        treatment_names=treatment_names,
        param_names=param_names,
        save=True,
        # fmt="pdf",
        # save_dir=save_dir,
    )
