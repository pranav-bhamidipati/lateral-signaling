import json
import h5py

import numpy as np
import pandas as pd

import colorcet as cc
import holoviews as hv

hv.extension("matplotlib")

import lateral_signaling as lsig

lsig.set_growth_params()

# sim_dir = lsig.simulation_dir.joinpath("20220113_increasingdensity/sacred")
sim_dir = lsig.simulation_dir.joinpath("20240401_increasingdensity/sacred")
invitro_data_csv = lsig.data_dir.joinpath("single_spots/singlespot_timeseries.csv")


def main(
    sim_dir=sim_dir,
    invitro_data_csv=invitro_data_csv,
    pad=0.05,
    sample_every=20,
    save_dir=lsig.plot_dir,
    save=False,
    fmt="png",
    dpi=300,
    save_curve_data=False,
    curve_data_dir=lsig.analysis_dir,
):

    ## Read invitro data from file
    data = pd.read_csv(invitro_data_csv)
    data = data.loc[data.Condition.str.contains("1250 cell/mm2")]
    data.days = data.days.astype(float)
    tmax_days = data.days.max()

    # Convert area units
    data["Area_mm2"] = data["area (um2)"] / 1e6
    del data["area (um2)"]

    # data["sqrtA_mm"] = np.sqrt(data["Area_mm2"])
    data["r_prop_mm"] = lsig.area_to_radius(data["Area_mm2"].values)

    # Get means and standard deviations
    agg_data = data.groupby(["Condition", "days"]).agg([np.mean, np.std]).reset_index()
    agg_data.columns = ["_".join(col).strip("_") for col in agg_data.columns.values]
    # agg_data["sqrtA_mm_mean_norm"] = agg_data["sqrtA_mm_mean"] / data["sqrtA_mm"].max()
    # agg_data["sqrtA_mm_std_norm"]  = agg_data["sqrtA_mm_std"]  / data["sqrtA_mm"].max()
    agg_data["r_prop_mm_mean_norm"] = (
        agg_data["r_prop_mm_mean"] / data["r_prop_mm"].max()
    )
    agg_data["r_prop_mm_std_norm"] = agg_data["r_prop_mm_std"] / data["r_prop_mm"].max()

    ## Read simulated data
    run_dir = next(d for d in sim_dir.glob("*") if d.joinpath("config.json").exists())

    # Expression threshold
    k = json.load(run_dir.joinpath("config.json").open("r"))["k"]

    with h5py.File(run_dir.joinpath("results.hdf5"), "r") as f:

        # Time-points
        t = np.asarray(f["t"])
        t_days = lsig.t_to_units(t)

        # Index of sender cell
        sender_idx = np.asarray(f["sender_idx"])

        # Density vs. time
        rho_t = np.asarray(f["rho_t"])

        # Signal and reporter expression vs. time
        S_t = np.asarray(f["S_t"])

    # Restrict time-range to match invitro
    # tmask  = t_days <= tmax_days
    # t      = t[tmask]
    # t_days = t_days[tmask]
    # rho_t  = rho_t[tmask]
    # S_t    = S_t[tmask]

    # Calculate the number of activated transceivers
    n_act_t = (S_t > k).sum(axis=-1) - 1

    # Area and sqrt(Area) of activation
    A_t = lsig.ncells_to_area(n_act_t, rho_t)
    # sqrtA_t = np.sqrt(A_t)
    r_prop_t = lsig.area_to_radius(A_t)

    ## Make plot
    # Axis limits with padding
    xmin = 0.0
    xmax = tmax_days + 1.0
    ymin = 0.0
    ymax = 0.45
    xlim = xmin - pad * (xmax - xmin), xmax + pad * (xmax - xmin)
    ylim = ymin - pad * (ymax - ymin), ymax + pad * (ymax - ymin)

    # Make data
    curve_data = {
        "days": t_days[::sample_every],
        "Area_mm": A_t[::sample_every],
        # "sqrtA_mm" : sqrtA_t[::sample_every],
        "r_prop_mm": r_prop_t[::sample_every],
    }

    if save_curve_data:
        from datetime import datetime

        today = datetime.today().strftime("%y%m%d")
        csv = curve_data_dir.joinpath(
            f"{today}_long_timecourse_insilico_curve_data.csv"
        )
        print(f"Writing to: {csv.resolve().absolute()}")
        pd.DataFrame(curve_data).to_csv(csv, index=False)

    # Plot
    points = hv.Scatter(
        agg_data,
        kdims=["days"],
        # vdims=["sqrtA_mm_mean"],
        vdims=["r_prop_mm_mean"],
        label=r"$in$ $vitro$",
    ).opts(
        c="k",
        s=20,
        marker="o",
    )

    curve = hv.Curve(
        curve_data,
        kdims=["days"],
        # vdims=["sqrtA_mm"],
        vdims=["r_prop_mm"],
        label=r"$in$ $silico$",
    ).opts(
        c=cc.gray[100],
        linewidth=2,
        linestyle="dashed",
    )

    errors = hv.ErrorBars(
        agg_data,
        kdims=["days"],
        # vdims=["sqrtA_mm_mean", "sqrtA_mm_std"],
        vdims=["r_prop_mm_mean", "r_prop_mm_std"],
    ).opts(
        edgecolor="k",
        linewidth=1,
        capsize=1,
    )

    overlay = hv.Overlay([curve, points, errors]).opts(
        xlabel="Days",
        xlim=xlim,
        xticks=(0, 2, 4, 6, 8),
        # ylabel=r"$\sqrt{Area}$ ($mm$)",
        #        ylabel=r"$\sqrt{Area}$ (norm.)",
        ylabel=r"$r_\mathrm{prop}$ ($mm$)",
        ylim=ylim,
        yticks=(0, 0.1, 0.2, 0.3, 0.4),
        aspect=1.3,
        fontscale=1.3,
        # legend_position="right",
    )

    if save:
        _fpath = save_dir.joinpath(f"longtimescale_invitro_insilico_overlay.{fmt}")
        print(f"Writing to: {_fpath.resolve().absolute()}")
        hv.save(overlay, _fpath, fmt=fmt, dpi=dpi)


if __name__ == "__main__":

    save_dir = lsig.plot_dir.joinpath("long_time_course")
    save_dir.mkdir(exist_ok=True)

    main(
        # save=True,
        # save_dir=save_dir,
        save_curve_data=True,
    )
