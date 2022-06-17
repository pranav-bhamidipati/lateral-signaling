import os
import h5py

import numpy as np
import pandas as pd
from tqdm import tqdm

import lateral_signaling as lsig

import matplotlib.pyplot as plt

# To read
data_dir           = os.path.abspath("../data/growth_curves_MLE")
data_fname         = os.path.join(data_dir, "growth_curves.csv")
bs_reps_dump_fpath = os.path.join(data_dir, "growth_curve_bs_reps.hdf5")
mle_df_fpath       = os.path.join(data_dir, "growth_parameters_MLE__.csv")

# To write
save_dir = os.path.abspath("../plots/tmp")
pred_reg_fname = lambda t, r0: os.path.join(
    save_dir, f"growth_regression_{t}_rho0_{r0/1250:.1f}"
)
overlay_reg_fname = lambda r0: os.path.join(
    save_dir, f"growth_regression_overlay_rho0_{r0/1250:.1f}"
)

def main(
    param_names=["untreated", "FGF2", "RI"],
    rho_0s=[1250., 2500., 5000.],  #cells / mm^2
    percentiles=[68, 90],
    overlay_ptile=80,
    overlay_colors=list(lsig.growthrate_colors),
    overlay_markers=["o", "^", "s"],
    figsize=(5, 3),
    seed=2021,
    save=False,
    dpi=300,
    fmt="png",
):

    # Set random state
    rg = np.random.default_rng(seed=seed)

    # Load experimental dataset
    data_df = pd.read_csv(data_fname)

    # Load best-fit MLE of parameters
    mle_df = pd.read_csv(mle_df_fpath, index_col=0)

    # Load bootstrap MLEs of parameters
    treatments   = []
    bs_reps_list = []
    with h5py.File(bs_reps_dump_fpath, "r") as f:
        for s in f.keys():
            t = s.split("_")[-1]

            treatments.append(t)
            bs_reps_list.append(np.asarray(f[s]))

    ## Set plotting options
    plot_kw = dict(
        xlabel="Days",
        ylabel=r"Cell density ($mm^{-2}$)",
        ylim=(0, 10000),
        yticks=np.linspace(0, 10000, 9)
    )

    ## Plot predictive regression for all samples
    regression_dfs = []
    for r0 in rho_0s:
        
        growth_curve_data_list = []
        bs_dens_t_list         = []
        t_data_list            = []

        for i, t in enumerate(treatments):

            # Experimental data
            growth_curve_data = data_df.loc[
                (data_df["treatment"] == t) \
                & (data_df["initial cell density (mm^-2)"] == r0), 
                ["days_integer", "cell density (mm^-2)"]
            ].values
            t_data = growth_curve_data[:, 0]
            growth_curve_data_list.append(growth_curve_data)
            t_data_list.append(t_data)

            # Generate predictions using bootstrap MLEs of growth parameters
            bs_mle = bs_reps_list[i]
            _gs, _rms, _sigma  = bs_mle.T
            bs_dens_t = np.array([
                lsig.logistic(_t, _gs, r0, _rms) + rg.normal(scale=_sigma)
                for _t in t_data
            ]).T
            bs_dens_t_list.append(bs_dens_t)

            # Get confidences for predictive regression
            df_pred = lsig.predictive_regression(
                bs_dens_t,
                t_data,
                percentiles=percentiles,
            )

            fig, ax = plt.subplots(figsize=figsize)
            ax.set(**plot_kw);
            lsig.plot_predictive_regression(
                df_pred=df_pred,
                data=growth_curve_data,
                ax=ax,
            )
            plt.title(f"{t}, rho_0={int(r0)}")
            plt.tight_layout()

            if save:
                fname = pred_reg_fname(t, r0) + "." + fmt
                print("Writing to:", fname)
                plt.savefig(fname, dpi=dpi)
            
            plt.close(fig)

        # Overlay all treatments
        fig, ax = plt.subplots(figsize=figsize)
        ax.set(**plot_kw)
        for i, (gc, bs, td) in enumerate(zip(
            growth_curve_data_list, bs_dens_t_list, t_data_list
        )):
            
            df_pred = lsig.predictive_regression(
                bs, td, percentiles=[overlay_ptile],
            )
            
            _color = np.array([lsig.hex2rgb(overlay_colors[i])]) / 255
            _color_lite = np.hstack([_color, [[0.2]]])
            lsig.plot_predictive_regression(
                df_pred=df_pred, 
                data=gc, 
                ax=ax,
                colors=[_color_lite, _color],
                median_lw=1,
            )
            
        for i, (gc, bs, td) in enumerate(zip(
            growth_curve_data_list, bs_dens_t_list, t_data_list
        )):
            
            _color = np.array([lsig.hex2rgb(overlay_colors[i])]) / 255
            df_data = pd.DataFrame(data=gc, columns=["__data_x", "__data_y"])
            df_data = df_data.sort_values(by="__data_x")
            plt.scatter(
                "__data_x", 
                "__data_y", 
                data=df_data, 
                c=_color,
                s=5,
                marker=overlay_markers[i],
            )
            
        plt.title(f"$\\rho_0={{{int(r0)}}}$")
        plt.tight_layout()

        if save:
            fname = overlay_reg_fname(r0) + "." + fmt
            print("Writing to:", fname)
            plt.savefig(fname, dpi=dpi)
        
        plt.close(fig)

main(
    save=True,
)


