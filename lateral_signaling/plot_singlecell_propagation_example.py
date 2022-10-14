import os
from glob import glob
import json
import h5py

import numpy as np
import holoviews as hv

hv.extension("matplotlib")

import lateral_signaling as lsig

# Reading
sim_dir = lsig.simulation_dir.joinpath("20220124_singlecell_example/sacred")


def main(
    sim_dir=sim_dir,
    save_dir=lsig.plot_dir,
    save=False,
    fmt="png",
    dpi=300,
):

    # Read simulated data
    run_dir = next(sim_dir.glob("[0-9]*"))

    with run_dir.joinpath("config.json").open("r") as c:
        config = json.load(c)
        delay = config["delay"]

    with h5py.File(run_dir.joinpath("results.hdf5"), "r") as f:

        # Time parameters
        t = np.asarray(f["t"])
        nt = t.size
        dt = t[1] - t[0]

        # Get delay in number of time-points
        nt_t = int(1 / dt)
        step_delay = int(delay * nt_t)

        # Indices of cell types
        sender_idx = np.atleast_1d(f["sender_idx"]).astype(int)[0]
        receiver_idx = np.atleast_1d(f["receiver_idx"]).astype(int)[0]
        transceiver_idx = np.atleast_1d(f["transceiver_idx"]).astype(int)[0]
        SRT_idx = np.array([sender_idx, receiver_idx, transceiver_idx])

        # Expression in each cell type
        E_t = np.asarray(f["E_t"])
        S_sender = E_t[:, sender_idx]
        S_transceiver = E_t[:, transceiver_idx]
        R_receiver = E_t[:, receiver_idx]

    ## Transform data for plotting
    # Get time-course with past included (negative time)
    t_past = -t[int(1.5 * step_delay) : 0 : -1]
    npast = t_past.size
    t_wpast = np.concatenate([t_past, t])

    # Prepend past expression
    S_sender = np.concatenate([np.zeros(npast), S_sender])
    R_receiver = np.concatenate([np.zeros(npast), R_receiver])
    S_transceiver = np.concatenate([np.zeros(npast), S_transceiver])

    # Normalize
    S_sender_norm = S_sender / S_sender.max()
    R_receiver_norm = R_receiver / R_receiver.max()
    S_transceiver_norm = S_transceiver / S_transceiver.max()
    E_t_norm = np.array([S_sender_norm, R_receiver_norm, S_transceiver_norm])[SRT_idx].T

    ## Plotting options
    # Ticks
    xticks = [
        (-2 * delay, "-2τ"),
        (-1 * delay, "-τ"),
        (0 * delay, "0"),
        (1 * delay, "τ"),
        (2 * delay, "2τ"),
        (3 * delay, "3τ"),
        (4 * delay, "4τ"),
        (5 * delay, "5τ"),
        (6 * delay, "6τ"),
        (7 * delay, "7τ"),
    ]

    # Curve labels
    labels = tuple(
        np.array(["Sender (GFP)", "Receiver (mCherry)", "Transceiver (GFP)"])[SRT_idx]
    )

    # Curve colors
    colors = tuple(
        np.array(
            [
                lsig.hexa2hex(lsig._gfp_green, 0.5),
                lsig._receiver_red,
                lsig._gfp_green,
            ]
        )[SRT_idx]
    )

    ## Make plot
    tau_lines = [
        hv.VLine(v).opts(
            c=lsig.gray,
            linewidth=1,
            linestyle="dashed",
        )
        for v in np.arange(t_wpast[npast - step_delay], t_wpast.max(), delay)
    ]
    t0_line = hv.VLine(0).opts(
        c=lsig.black,
        linewidth=1,
    )
    curves = [
        hv.Curve(
            (t_wpast, e),
            label=l,
        ).opts(c=c)
        for e, c, l in zip(E_t_norm.T, colors, labels)
    ]

    overlay = hv.Overlay([*tau_lines, t0_line, *curves]).opts(
        hv.opts.Curve(
            linewidth=3,
        ),
        hv.opts.Overlay(
            xlabel="Time",
            # xticks=(0, 1, 2, 3),
            xticks=xticks,
            ylabel="Norm. expression",
            yticks=(0, 0.5, 1),
            aspect=1.8,
            legend_position="right",
        ),
    )

    if save:
        _fpath = save_dir.joinpath(f"singlecell_example_plot.{fmt}")
        print(f"Writing to: {_fpath.resolve().absolute()}")
        hv.save(overlay, _fpath, fmt=fmt, dpi=dpi)


if __name__ == "__main__":
    main(
        save=True,
    )
