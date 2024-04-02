import json
import h5py
import numpy as np
import pandas as pd
import holoviews as hv

hv.extension("matplotlib")

import lateral_signaling as lsig

lsig.set_growth_params()

sim_dir = lsig.simulation_dir.joinpath(
    # "20220204_phase_perturbations_growthrate/sacred"
    "20240401_wholewell_phase_perturbations/sacred"
)
data_csv = lsig.data_dir.joinpath(
    "whole_wells/drug_conditions_propagation_and_cellcount.csv"
)


def main(
    sim_dir=sim_dir,
    data_csv=data_csv,
    colors=lsig.viz.growthrate_colors,
    pad=0.05,
    sample_every=5,
    save_dir=lsig.plot_dir,
    save=False,
    fmt="png",
    dpi=300,
    save_curve_data=False,
    curve_save_dir=lsig.analysis_dir,
):

    ## Read invitro data from file
    data = pd.read_csv(data_csv).dropna()

    # Get last time point
    tmax_days = data.Day.max()

    # Get conditions
    conds = data.Condition.unique()

    # Get normalized area
    data["Area_norm"] = data["GFP_um2"] / data["GFP_um2"].max()

    #    # Convert area units and take sqrt
    #    data["Area_mm2"]  = data["GFP_um2"] / 1e6
    #    del data["GFP_um2"]
    #    data["sqrtA_mm"] = np.sqrt(data["Area_mm2"])

    # axis limits with padding
    xmin = 1.0
    xmax = tmax_days
    ymin = 0.0
    ymax = 1.0

    # axis ticks
    yticks = (0, 0.2, 0.4, 0.6, 0.8, 1.0)
    xticks = (1, 2, 3, 4, 5, 6, 7, 8)

    # Set colors/linestyles/markers
    ccycle = hv.Cycle(colors)

    # Other options
    opts = dict(
        xlabel="Days",
        xlim=(xmin - pad * (xmax - xmin), xmax + pad * (xmax - xmin)),
        xticks=xticks,
        ylabel="% of area",
        ylim=(ymin - pad * (ymax - ymin), ymax + pad * (ymax - ymin)),
        yticks=yticks,
        aspect=1.3,
        fontscale=1.3,
        #        show_legend=False,
        legend_position="right",
    )

    ## Plot in vitro data
    invitro_curves = (
        hv.Curve(
            data,
            kdims=["Day"],
            vdims=["Area_norm", "Condition"],
        )
        .groupby("Condition")
        .opts(
            c=ccycle,
            linewidth=2,
        )
        .overlay()
    )

    invitro_points = (
        hv.Scatter(
            data,
            kdims=["Day"],
            vdims=["Area_norm", "Condition"],
        )
        .groupby("Condition")
        .opts(
            c=ccycle,
            s=40,
        )
        .overlay()
    )

    invitro_plot = hv.Overlay([invitro_curves, invitro_points]).opts(**opts)

    if save:
        from datetime import datetime

        today = datetime.today().strftime("%y%m%d")
        _fpath = save_dir.joinpath(f"{today}_growthrate_perturbation_invitro.{fmt}")
        print(f"Writing to: {_fpath.resolve().absolute()}")
        hv.save(invitro_plot, _fpath, fmt=fmt, dpi=dpi)

    ## Read simulated data
    run_dirs = [d for d in sim_dir.glob("*") if d.joinpath("config.json").exists()]

    # Initialize lists to store data
    gs = []
    dconds = []
    rho_ts = []
    S_t_reps = []
    sender_idx_reps = []

    for i, rd in enumerate(run_dirs):

        with rd.joinpath("config.json").open("r") as c:
            config = json.load(c)

            # Drug condition
            dcond = config["drug_condition"]

            # Expression threshold
            k = config["k"]

            # Intrinsic prolif rate
            g = config["g"]

        with h5py.File(rd.joinpath("results.hdf5"), "r") as f:

            # Restrict time-range to match in vitro
            t = np.asarray(f["t"])
            t_days = lsig.t_to_units(t)
            tmask = t_days <= tmax_days
            t = t[tmask]
            t_days = t_days[tmask]
            nt = t.size

            # Re-scale time axis
            opts["xlim"] = t_days[0], opts["xlim"][1]
            opts["xticks"] = tuple([0] + list(opts["xticks"]))

            # Index of sender cell
            n_senders = int(np.asarray(f["n_senders"]))

            # Density vs. time
            rho_t = np.asarray(f["rho_t"])[tmask]

            # Signal and reporter expression vs. time
            S_t_rep = np.asarray(f["S_t_rep"])[:, tmask]
            n = S_t_rep.shape[-1]

            # Get senders
            sender_idx_rep = np.asarray(f["sender_idx_rep"])

            # Store data
            gs.append(g)
            dconds.append(dcond)
            rho_ts.append(rho_t)
            S_t_reps.append(S_t_rep)
            sender_idx_reps.append(sender_idx_rep)

    # Convert to arrays
    rho_ts = np.asarray(rho_ts)
    S_t_reps = np.asarray(S_t_reps)
    sender_idx_reps = np.asarray(sender_idx_reps)

    # Get number of conditions
    n_runs = len(dconds)

    # Calculate the avg number of activated transceivers across replicates
    n_act_t_reps = (S_t_reps > k).sum(axis=-1) - n_senders
    n_act_ts = n_act_t_reps.mean(axis=1)

    # Percent activated transceivers
    pctArea_ts = n_act_ts / (n - n_senders)

    # (Normalized) Area and sqrt(Area) of activation
    A_ts = np.array([lsig.ncells_to_area(nc, rt) for nc, rt in zip(n_act_ts, rho_ts)])
    normA_ts = A_ts / A_ts.max()
    sqrtA_ts = np.sqrt(A_ts)

    ## Make plot
    # Make dataframe for plotting
    insilico_data = {
        "condition": np.repeat(dconds, t_days[::sample_every].size),
        "days": np.tile(t_days[::sample_every], n_runs),
        "pctArea": pctArea_ts[:, ::sample_every].flatten(),
        "Area_norm": normA_ts[:, ::sample_every].flatten(),
        "Area_mm": A_ts[:, ::sample_every].flatten(),
        "sqrtA_mm": sqrtA_ts[:, ::sample_every].flatten(),
    }

    if save_curve_data:
        from datetime import datetime

        today = datetime.today().strftime("%y%m%d")
        csv = curve_save_dir.joinpath(
            f"{today}_growthrate_perturbation_insilico_curve_data.csv"
        )
        print(f"Writing to: {csv.resolve().absolute()}")
        pd.DataFrame(insilico_data).to_csv(csv, index=False)

    # Plot curves
    insilico_plot = (
        hv.Curve(
            insilico_data,
            kdims=["days"],
            vdims=["pctArea", "condition"],
        )
        .groupby(
            "condition",
        )
        .opts(
            linewidth=3,
            color=ccycle,
        )
        .overlay("condition")
        .opts(**opts)
    )

    if save:
        from datetime import datetime

        today = datetime.today().strftime("%y%m%d")
        _fpath = save_dir.joinpath(
            f"{today}_growthrate_perturbation_pctarea_insilico.{fmt}"
        )
        print(f"Writing to: {_fpath}")
        hv.save(insilico_plot, _fpath, fmt=fmt, dpi=dpi)


if __name__ == "__main__":

    save_dir = lsig.plot_dir.joinpath("growth_perturbations")
    save_dir.mkdir(exist_ok=True)

    main(
        save=True,
        save_dir=save_dir,
        save_curve_data=True,
    )
