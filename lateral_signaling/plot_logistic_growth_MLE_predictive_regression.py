import h5py

import numpy as np
import pandas as pd

import lateral_signaling as lsig

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

lsig.set_growth_params()


def fix_name_for_saving(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")


def main(
    growth_curve_data_csv,
    bootstrap_mle_replicates_hdf,
    treatment_names=None,
    rho_0s=[1250.0],  # cells / mm^2
    rho_0_labels=["1x", "2x", "4x"],
    percentiles=[68, 80, 90],
    overlay_ptile=80,
    overlay_colors=None,
    overlay_markers=["o", "o", "o"],
    figsize=(5, 3),
    seed=2021,
    save_dir=lsig.plot_dir,
    save=False,
    dpi=300,
    fmt="png",
    save_plotting_data=False,
    save_data_dir=lsig.analysis_dir,
    rho_max=None,
    return_data=False,
):

    if rho_max is None:
        rho_max = lsig.mle_params.rho_max_inv_mm2

    if overlay_colors is None:
        overlay_colors = [lsig.viz.rgb2hex(rgb) for rgb in sns.color_palette()]

    # Set random state
    rg = np.random.default_rng(seed=seed)

    # Load experimental dataset
    data_df = pd.read_csv(growth_curve_data_csv)
    t_pred = np.linspace(0, data_df["days_integer"].max(), 200)

    # Load bootstrap MLEs of parameters
    treatments = []
    bs_reps_list = []

    def collect_hdf_contents(name_with_prefix, obj):
        if isinstance(obj, h5py.Dataset):
            name = name_with_prefix.removeprefix("bs_reps_")
            if treatment_names is None or name in treatment_names:
                treatments.append(name)
                bs_mle = np.array(obj)
                if bs_mle.shape[1] == 2:
                    # Assume rho_max was fixed to the MLE value and make a column for it
                    bs_mle = np.insert(bs_mle, 1, rho_max, axis=1)
                bs_reps_list.append(bs_mle)

    with h5py.File(bootstrap_mle_replicates_hdf, "r") as f:
        f.visititems(collect_hdf_contents)

    ## Set plotting options
    plot_kw = dict(
        xlabel="Days",
        ylabel=r"Cell density ($mm^{-2}$)",
        ylim=(0, 10000),
        yticks=np.linspace(0, 10000, 9),
    )

    ## Plot predictive regression for all samples
    # Also save data for plotting in other software
    dfs = []
    n_total = len(treatments) * (len(rho_0s) + (len(rho_0s) > 1))
    pbar = tqdm(total=n_total, desc="Plotting predictive regression")
    for i, t in enumerate(treatments):

        growth_curve_data_list = []
        bs_dens_t_list = []

        bs_mle = bs_reps_list[i]
        _gs, _rho_maxs, _sigmas = bs_mle.T

        for r0 in rho_0s:

            # Experimental data
            experiment_mask = (data_df["treatment"] == t) & (
                np.isclose(data_df["initial cell density (mm^-2)"], r0)
            )
            if experiment_mask.sum() == 0:
                pbar.update(1)
                continue
            growth_curve_data = data_df.loc[
                experiment_mask,
                ["days_integer", "cell density (mm^-2)"],
            ].values
            t_data = np.unique(growth_curve_data[:, 0])
            growth_curve_data_list.append(growth_curve_data)

            # Generate predictions using bootstrap MLEs of growth parameters
            bs_dens_t = np.array(
                [
                    lsig.logistic(_t, _gs, r0, _rho_maxs) + rg.normal(scale=_sigmas)
                    for _t in t_pred
                ]
            ).T
            bs_dens_t_list.append(bs_dens_t)

            # Get confidences for predictive regression
            df_pred = lsig.viz.predictive_regression(
                bs_dens_t,
                t_pred,
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
                from datetime import datetime

                today = datetime.today().strftime("%y%m%d")
                tsave = fix_name_for_saving(t)
                fname = save_dir.joinpath(
                    f"{today}_growth_regression_{tsave}_rho0_{r0/1250:.1f}.{fmt}"
                )
                print("Writing to:", fname.resolve().absolute())
                plt.savefig(fname, dpi=dpi)

            plt.close(fig)

            df_pred["treatment"] = t
            df_pred["rho_0"] = r0
            dfs.append(df_pred)

            pbar.update(1)

        if len(rho_0s) > 1:
            # Overlay all densities
            fig, ax = plt.subplots(figsize=figsize)
            ax.set(**plot_kw)
            for i, (gc, bs) in enumerate(zip(growth_curve_data_list, bs_dens_t_list)):

                df_pred = lsig.viz.predictive_regression(
                    bs, t_pred, percentiles=[overlay_ptile]
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

            for i, (gc, bs) in enumerate(zip(growth_curve_data_list, bs_dens_t_list)):

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
            plt.legend(
                title="Plating density", bbox_to_anchor=(1, 0.5), loc="center left"
            )
            #        plt.title(f"$\\rho_0={{{int(r0)}}}$")
            plt.tight_layout()

            if save:
                from datetime import datetime

                today = datetime.today().strftime("%y%m%d")
                tsave = fix_name_for_saving(t)
                fname = save_dir.joinpath(
                    f"{today}_growth_regression_overlay_{tsave}.{fmt}"
                )
                print("Writing to:", fname.resolve().absolute())
                plt.savefig(fname, dpi=dpi)

        pbar.update(1)

    pbar.close()

    df = pd.concat(dfs, ignore_index=True)
    df = df.rename({"__x": "days"}, axis=1)
    df = df.rename(
        lambda x: f"{float(x):.1f}_percentile" if str(x)[0] in "0123456789" else x,
        axis=1,
    )

    if save_plotting_data:
        from datetime import datetime

        today = datetime.today().strftime("%y%m%d")

        # Save data with the correct date
        stem = bs_mle_reps_hdf.stem
        if set(stem[:6]).issubset(set("0123456789")):
            stem = stem[6:].lstrip("_")

        fpath = save_data_dir.joinpath(f"{today}_{stem}_regression_data.csv")
        print("Writing to:", fpath.resolve().absolute())
        df.to_csv(fpath, index=False)

    if return_data:
        return df


if __name__ == "__main__":

    # # Old growth curves
    # growth_curve_data_csv = lsig.data_dir.joinpath(
    #     "growth_curves_MLE/growth_curves.csv"
    # )
    # bs_mle_reps_hdf = lsig.analysis_dir.joinpath(
    #     "growth_curve_bootstrap_replicates.hdf5"
    # )

    # Results from parameter fitting for untreated condition (10% FBS)
    growth_curve_data_csv = lsig.data_dir.joinpath(
        "growth_curves_MLE/231219_growth_curves.csv"
    )
    bs_mle_reps_hdf = lsig.analysis_dir.joinpath(
        "240327_growth_curve_bootstrap_replicates.hdf5"
    )
    treatment_names = ["10% FBS"]

    data_untreated = main(
        growth_curve_data_csv=growth_curve_data_csv,
        bootstrap_mle_replicates_hdf=bs_mle_reps_hdf,
        treatment_names=treatment_names,
        save=True,
        save_plotting_data=True,
        return_data=True,
    )

    # Results from parameter fitting with fixed rho_max
    growth_curve_data_csv = lsig.data_dir.joinpath(
        "growth_curves_MLE/231219_growth_curves.csv"
    )
    bs_mle_reps_hdf = lsig.analysis_dir.joinpath(
        # "240401_growth_curve_bootstrap_replicates_fixed_rhomax.hdf5"
        "240402_growth_curve_bootstrap_replicates_fixed_rhomax.hdf5"
    )
    treatment_names = None

    data_fixed_rhomax = main(
        growth_curve_data_csv=growth_curve_data_csv,
        bootstrap_mle_replicates_hdf=bs_mle_reps_hdf,
        treatment_names=treatment_names,
        save=True,
        # save_plotting_data=True,
        return_data=True,
    )

    # Save combined data
    from datetime import datetime

    data_combined = pd.concat([data_untreated, data_fixed_rhomax], ignore_index=True)
    today = datetime.today().strftime("%y%m%d")
    fpath = lsig.analysis_dir.joinpath(f"{today}_growth_regression_data_combined.csv")
    print("Writing to:", fpath.resolve().absolute())
    data_combined.to_csv(fpath, index=False)

    # Plot overlay of multiple densities in untreated condition (includes Marco's data)

    # Results from parameter fitting for untreated condition (10% FBS)
    growth_curve_data_csv = lsig.data_dir.joinpath(
        "growth_curves_MLE/231219_growth_curves.csv"
    )
    bs_mle_reps_hdf = lsig.analysis_dir.joinpath(
        "240327_growth_curve_bootstrap_replicates.hdf5"
    )
    treatment_names = ["10% FBS"]
    initial_densities = [1250.0, 2500.0, 5000.0]

    data_untreated = main(
        growth_curve_data_csv=growth_curve_data_csv,
        bootstrap_mle_replicates_hdf=bs_mle_reps_hdf,
        treatment_names=treatment_names,
        rho_0s=initial_densities,
        save=True,
        # save_plotting_data=True,
        # return_data=True,
    )
