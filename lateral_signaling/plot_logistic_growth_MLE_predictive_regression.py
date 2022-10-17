import h5py

import numpy as np
import pandas as pd

import lateral_signaling as lsig

import matplotlib.pyplot as plt
import seaborn as sns

data_csv = lsig.data_dir.joinpath("growth_curves_MLE", "growth_curves.csv")
bs_reps_dump_fpath = lsig.analysis_dir.joinpath(
    "growth_curve_bootstrap_replicates.hdf5"
)

colors = [lsig.viz.rgb2hex(rgb) for rgb in sns.color_palette()[:3]]


def main(
    rho_0s=[1250.0, 2500.0, 5000.0],  # cells / mm^2
    rho_0_labels=["1x", "2x", "4x"],
    percentiles=[68, 90],
    overlay_ptile=80,
    overlay_colors=colors,
    overlay_markers=["o", "o", "o"],
    figsize=(5, 3),
    seed=2021,
    save_dir=lsig.plot_dir,
    save=False,
    dpi=300,
    fmt="png",
):

    # Set random state
    rg = np.random.default_rng(seed=seed)

    # Load experimental dataset
    data_df = pd.read_csv(data_csv)

    # Load bootstrap MLEs of parameters
    treatments = []
    bs_reps_list = []
    with h5py.File(bs_reps_dump_fpath, "r") as f:
        for key, bs_reps in f.items():
            treatments.append(key.split("_")[-1])
            bs_reps_list.append(np.asarray(bs_reps))

    ## Set plotting options
    plot_kw = dict(
        xlabel="Days",
        ylabel=r"Cell density ($mm^{-2}$)",
        ylim=(0, 10000),
        yticks=np.linspace(0, 10000, 9),
    )

    ## Plot predictive regression for all samples
    for i, t in enumerate(treatments):

        growth_curve_data_list = []
        bs_dens_t_list = []
        t_data_list = []

        bs_mle = bs_reps_list[i]
        _gs, _rms, _sigma = bs_mle.T

        for r0 in rho_0s:

            # Experimental data
            growth_curve_data = data_df.loc[
                (data_df["treatment"] == t)
                & (data_df["initial cell density (mm^-2)"] == r0),
                ["days_integer", "cell density (mm^-2)"],
            ].values
            t_data = growth_curve_data[:, 0]
            growth_curve_data_list.append(growth_curve_data)
            t_data_list.append(t_data)

            # Generate predictions using bootstrap MLEs of growth parameters
            bs_dens_t = np.array(
                [
                    lsig.logistic(_t, _gs, r0, _rms) + rg.normal(scale=_sigma)
                    for _t in t_data
                ]
            ).T
            bs_dens_t_list.append(bs_dens_t)

            # Get confidences for predictive regression
            df_pred = lsig.viz.predictive_regression(
                bs_dens_t,
                t_data,
                percentiles=percentiles,
            )

            fig, ax = plt.subplots(figsize=figsize)
            ax.set(**plot_kw)
            lsig.viz.plot_predictive_regression(
                df_pred=df_pred,
                data=growth_curve_data,
                ax=ax,
            )
            plt.title(f"{t}, rho_0={int(r0)}")
            plt.tight_layout()

            if save:
                fname = save_dir.joinpath(
                    f"growth_regression_{t}_rho0_{r0/1250:.1f}.{fmt}"
                )
                print("Writing to:", fname.resolve().absolute())
                plt.savefig(fname, dpi=dpi)

            plt.close(fig)

        # Overlay all densities
        fig, ax = plt.subplots(figsize=figsize)
        ax.set(**plot_kw)
        for i, (gc, bs, td) in enumerate(
            zip(growth_curve_data_list, bs_dens_t_list, t_data_list)
        ):

            df_pred = lsig.viz.predictive_regression(
                bs,
                td,
                percentiles=[overlay_ptile],
            )

            _color = np.array([lsig.viz.hex2rgb(overlay_colors[i])]) / 255
            _color_lite = np.hstack([_color, [[0.2]]])
            lsig.viz.plot_predictive_regression(
                df_pred=df_pred,
                #                data=gc,
                ax=ax,
                colors=[_color_lite, _color],
                median_lw=1,
                median_kwargs=dict(label="_nolegend_"),
            )

        for i, (gc, bs, td) in enumerate(
            zip(growth_curve_data_list, bs_dens_t_list, t_data_list)
        ):

            _color = np.array([lsig.viz.hex2rgb(overlay_colors[i])]) / 255
            df_data = pd.DataFrame(data=gc, columns=["__data_x", "__data_y"])
            df_data = df_data.sort_values(by="__data_x")
            plt.scatter(
                "__data_x",
                "__data_y",
                data=df_data,
                c=_color,
                s=15,
                marker=overlay_markers[i],
                label=rho_0_labels[i],
            )

        plt.title(t)
        plt.legend(title="Plating density", bbox_to_anchor=(1, 0.5), loc="center left")
        #        plt.title(f"$\\rho_0={{{int(r0)}}}$")
        plt.tight_layout()

        if save:
            fname = save_dir.joinpath(f"growth_regression_overlay_{t}.{fmt}")
            print("Writing to:", fname.resolve().absolute())
            plt.savefig(fname, dpi=dpi)


if __name__ == "__main__":
    main(
        save=True,
    )
