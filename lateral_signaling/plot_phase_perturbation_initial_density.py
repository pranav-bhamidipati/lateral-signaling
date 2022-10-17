import json
import h5py

import numpy as np
import pandas as pd

import holoviews as hv

hv.extension("matplotlib")

import lateral_signaling as lsig

sacred_dir = lsig.simulation_dir.joinpath("20220114_phase_perturbations/sacred")
data_fname = lsig.data_dir.joinpath("single_spots/singlespot_timeseries.csv")

save_dir = lsig.plot_dir


def main(
    sacred_dir=sacred_dir,
    data_fname=data_fname,
    pad=0.05,
    seed=2021,
    sample_every=1,
    save_dir=save_dir,
    save=False,
    fmt="png",
    dpi=300,
):

    # Set random seed
    rg = np.random.default_rng(seed=seed)

    ## Read invitro data from file
    data = pd.read_csv(str(data_fname))
    data = data.loc[data.Condition.str.contains("cell/mm2")]
    data = data.astype({"Condition": "category", "days": float})

    # Get last time point
    tmax_days = data.days.max() + 0.5

    # Get conditions
    conds = np.unique(data.Condition.values)

    # Convert area units and take sqrt
    data["Area_mm2"] = data["area (um2)"] / 1e6
    del data["area (um2)"]
    # data["sqrtA_mm"] = np.sqrt(data["Area_mm2"])
    data["r_prop_mm"] = lsig.area_to_radius(data["Area_mm2"].values)

    # Get means and standard deviations
    agg_data = (
        data.groupby(["Condition", "days"]).agg([np.mean, np.std, np.max]).reset_index()
    )
    agg_data.columns = ["_".join(col).strip("_") for col in agg_data.columns.values]
    # agg_data["sqrtA_mm_mean_norm"] = agg_data["sqrtA_mm_mean"] / data["sqrtA_mm"].max()
    # agg_data["sqrtA_mm_std_norm"] = agg_data["sqrtA_mm_std"] / data["sqrtA_mm"].max()
    agg_data["r_prop_mm_mean_norm"] = (
        agg_data["r_prop_mm_mean"] / data["r_prop_mm"].max()
    )
    agg_data["r_prop_mm_std_norm"] = agg_data["r_prop_mm_std"] / data["r_prop_mm"].max()

    # axis limits with padding
    xmin = 0.0
    xmax = tmax_days
    ymin = 0.0
    ymax = 0.25
    xlim = xmin - pad * (xmax - xmin), xmax + pad * (xmax - xmin)
    ylim = ymin - pad * (ymax - ymin), ymax + pad * (ymax - ymin)

    # axis ticks
    yticks = (0, 0.1, 0.2, 0.3, 0.4)

    # Set colors/linestyles/markers
    ccycle = lsig.viz.sample_cycle(lsig.viz.yob[1:], conds.size)

    # Other options
    opts = dict(
        xlabel="Days",
        xlim=xlim,
        xticks=(0, 2, 4, 6, 8),
        # ylabel=r"$\sqrt{Area}$ ($mm$)",
        ylabel=r"$r_\mathrm{prop}$ ($mm$)",
        ylim=ylim,
        yticks=yticks,
        # aspect=1.3,
        fontscale=1.6,
        #        show_legend=False,
        legend_position="top_right",
    )

    ## Plot in vitro data
    curves = [
        hv.Curve(
            agg_data.loc[agg_data.Condition == cond],
            kdims=["days"],
            # vdims=["sqrtA_mm_mean", "Condition"],
            vdims=["r_prop_mm_mean", "Condition"],
            label=("1x", "2x", "4x")[i],
        ).opts(
            c=ccycle.values[i],
            linewidth=3,
        )
        for i, cond in enumerate(conds)
    ]

    errors = [
        hv.ErrorBars(
            agg_data.loc[agg_data.Condition == cond],
            kdims=["days"],
            # vdims=["sqrtA_mm_mean", "sqrtA_mm_std"],
            vdims=["r_prop_mm_mean", "r_prop_mm_std"],
            label=("1x", "2x", "4x")[i],
        ).opts(
            edgecolor=ccycle.values[i],
            linewidth=1,
            capsize=2,
        )
        for i, cond in enumerate(conds)
    ]

    invitro_plot = hv.Overlay([c * e for c, e in zip(curves, errors)]).opts(**opts)

    if save:

        _fpath = save_dir.joinpath(f"density_perturbation_prop_radius_invitro.{fmt}")
        print(f"Writing to: {_fpath.resolve().absolute()}")
        hv.save(invitro_plot, _fpath, dpi=dpi)

    ## Plot summary of max propagation size

    # Get the maximum size achieved by each spot
    max_data = data.groupby(["Condition", "SpotNum"]).agg(np.max).reset_index()
    max_data["xval"] = 1 + max_data.Condition.cat.codes.values

    # Get mean of size for each condition
    mean_max = max_data.groupby("Condition").agg(np.mean).reset_index()

    # Apply jitter for swarm plotting
    max_data["jitter_xval"] = max_data.xval.values + rg.normal(
        0, 0.1, size=max_data.shape[0]
    )

    # Construct line segments through the means for plotting
    width = 0.5
    xvals = mean_max["xval"].values
    # means = mean_max["sqrtA_mm"].values
    means = mean_max["r_prop_mm"].values
    mean_max_arr = np.array([xvals - width / 2, means, xvals + width / 2, means]).T

    # Plot points and means
    # max_points = hv.Scatter(max_data, kdims=["jitter_xval"], vdims=["sqrtA_mm"],).opts(
    #     c="k",
    #     s=45,
    #     alpha=0.6,
    # )
    max_points = hv.Scatter(max_data, kdims=["jitter_xval"], vdims=["r_prop_mm"],).opts(
        c="k",
        s=45,
        alpha=0.6,
    )

    mean_max_segs = hv.Segments(mean_max_arr,).opts(
        color=lsig.viz.cols_blue[-2],
        linewidth=4,
    )

    invitro_max_plot = hv.Overlay([max_points, mean_max_segs]).opts(
        xlabel="Init. density",
        xlim=(0.5, 3.5),
        xticks=[(1, "1x"), (2, "2x"), (3, "4x")],
        ylabel=r"$\mathrm{max.}\; r_\mathrm{prop}\,(\mathrm{mm})$",
        ylim=opts["ylim"],
        yticks=opts["yticks"],
        aspect=0.9,
        fontsize=dict(labels=20, ticks=16),
    )

    if save:

        _fpath = save_dir.joinpath(
            f"density_perturbation_prop_radius_invitro_max.{fmt}"
        )
        print(f"Writing to: {_fpath.resolve().absolute()}")
        hv.save(invitro_max_plot, _fpath, dpi=dpi)

    ## Read simulated data
    run_dirs = list(sacred_dir.glob("[0-9]*"))

    # Initialize lists to store data
    rho_0s = []
    rho_ts = []
    S_ts = []

    for i, run_dir in enumerate(run_dirs):
        config_file = run_dir.joinpath("config.json")
        results_file = run_dir.joinpath("results.hdf5")

        with config_file.open("r") as c:
            config = json.load(c)

            # Check if run matches conditions
            if config["drug_condition"] != "untreated":
                continue

            # Expression threshold
            k = config["k"]

            # Initial density
            rho_0 = config["rho_0"]

        with h5py.File(str(results_file), "r") as f:

            # Restrict time-range to match in vitro
            t = np.asarray(f["t"])
            t_days = lsig.t_to_units(t)
            tmask = t_days <= tmax_days
            t = t[tmask]
            t_days = t_days[tmask]

            # Density vs. time
            rho_t = np.asarray(f["rho_t"])[tmask]

            # Signal and reporter expression vs. time
            S_t = np.asarray(f["S_t"])[tmask]

            # Store data
            rho_0s.append(rho_0)
            rho_ts.append(rho_t)
            S_ts.append(S_t)

    # Convert to arrays
    rho_ts = np.asarray(rho_ts)
    S_ts = np.asarray(S_ts)

    # Get initial densities as strings
    n_runs = len(rho_0s)
    rho_0_strs = [f"{int(r)}x" for r in rho_0s]

    # Calculate the number of activated transceivers
    n_act_ts = (S_ts > k).sum(axis=-1) - 1

    # Size of activation
    A_ts = np.array([lsig.ncells_to_area(nc, rt) for nc, rt in zip(n_act_ts, rho_ts)])
    # sqrtA_ts = np.sqrt(A_ts)
    r_prop_ts = lsig.area_to_radius(A_ts)

    ## Make plot
    # Make dataframe for plotting
    insilico_data = {
        "Init. density": np.repeat(rho_0_strs, t_days[::sample_every].size),
        "Days": np.tile(t_days[::sample_every], n_runs),
        "Area_mm": A_ts[:, ::sample_every].flatten(),
        # "sqrtA_mm" : sqrtA_ts[:, ::sample_every].flatten(),
        "r_prop_mm": r_prop_ts[:, ::sample_every].flatten(),
    }

    # Plot curves
    insilico_plot = (
        hv.Curve(
            insilico_data,
            kdims=["Days"],
            # vdims=["sqrtA_mm", "Init. density"],
            vdims=["r_prop_mm", "Init. density"],
        )
        .groupby(
            "Init. density",
        )
        .opts(
            linewidth=3,
            color=ccycle,
        )
        .overlay("Init. density")
        .opts(**opts)
    )

    if save:

        _fpath = save_dir.joinpath(f"density_perturbation_prop_radius_insilico.{fmt}")
        print(f"Writing to: {_fpath.resolve().absolute()}")
        hv.save(insilico_plot, _fpath, dpi=dpi)


if __name__ == "__main__":
    main(
        save=True,
        save_dir=lsig.temp_plot_dir,
    )
