import os
from glob import glob
import json
from copy import deepcopy
import h5py

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation

import lateral_signaling as lsig

sacred_dir = lsig.simulation_dir.joinpath("20220111_constantdensity/sacred")


def main(
    sacred_dir=sacred_dir,
    tmax_delay=7,
    save_dir=lsig.plot_dir,
    save=False,
    writer="ffmpeg",
    n_frames=100,
    fps=20,
    dpi=300,
):

    # Read in data from experiments
    run_dirs = [d for d in sacred_dir.glob("*") if d.joinpath("config.json").exists()]
    rd0 = run_dirs[0]

    with rd0.joinpath("config.json").open("r") as c:
        config = json.load(c)

        # Dimensions of cell sheet
        rows = config["rows"]
        cols = config["cols"]

        # Delay parameter
        delay = config["delay"]

    with h5py.File(rd0.joinpath("results.hdf5"), "r") as f:

        # Time-points
        t = np.asarray(f["t"])
        dt = t[1] - t[0]

        # Index of sender cell
        sender_idx = np.asarray(f["sender_idx"])

    Xs = []
    rhos = []
    S_ts = []
    R_ts = []
    for i, rd in enumerate(run_dirs):
        with h5py.File(rd.joinpath("results.hdf5"), "r") as f:

            # Density (constant)
            rho = np.asarray(f["rho_t"])[0]

            # Signal and reporter expression vs. time
            S_t = np.asarray(f["S_t"])
            R_t = np.asarray(f["R_t"])

        # Store data
        rhos.append(rho)
        S_ts.append(S_t)
        R_ts.append(R_t)

    tmax = delay * tmax_delay

    sort_rhos = np.argsort(rhos)
    rhos = np.asarray(rhos)[sort_rhos]
    S_ts = np.asarray(S_ts)[sort_rhos][:, t <= tmax]
    R_ts = np.asarray(R_ts)[sort_rhos][:, t <= tmax]
    t = t[t <= tmax]
    nt = t.size

    # Make a lattice centered on the sender
    X = lsig.hex_grid(rows, cols)
    X = X - X[sender_idx]

    # Get mask of non-sender cells (transceivers)
    n = X.shape[0]
    ns_mask = np.ones(n, dtype=bool)
    ns_mask[sender_idx] = False

    # Get cell positions based on density
    Xs = np.multiply.outer(1 / np.sqrt(rhos), X)

    ## Some manual plotting options
    # Font sizes
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    # Zoom in to a factor of `zoom` (to emphasize ROI)
    zoom = 0.50

    if save:

        # Get which frames to animate
        nt = t.size
        frames = lsig.vround(np.linspace(0, nt - 1, n_frames))

        # Set font sizes
        plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
        plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
        plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

        # Make figure
        fig, axs = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(6.0, 2.0),
            gridspec_kw=dict(width_ratios=[1, 1, 1.3]),
        )

        # Get default kwargs for plotting
        kw = deepcopy(lsig.viz.plot_kwargs)
        kw["sender_idx"] = sender_idx

        # Turn on scalebar
        kw["scalebar"] = True

        # axis limits
        _xmax = np.abs(Xs[np.argmax(rhos), :, 0]).max()
        _ymax = np.abs(Xs[np.argmax(rhos), :, 1]).max()
        kw["xlim"] = -_xmax * zoom, _xmax * zoom
        kw["ylim"] = -_ymax * zoom, _ymax * zoom

        # colorscale limits
        kw["vmin"] = 0
        kw["vmax"] = S_ts[:, :, ns_mask].max()

        # some args for colorbar
        kw["cmap"] = lsig.viz.kgy
        kw["cbar_aspect"] = 8
        kw["cbar_kwargs"] = dict(shrink=0.9, label="GFP (AU)", format="%.2f")

        # Make colorbars
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(
                norm=mpl.colors.Normalize(kw["vmin"], kw["vmax"]), cmap=kw["cmap"]
            ),
            ax=axs[-1],
            aspect=kw["cbar_aspect"],
            extend=kw["extend"],
            ticks=[kw["vmin"], kw["vmax"]],
            **kw["cbar_kwargs"],
        )

        # Turn off further colorbar plotting during animation
        kw["colorbar"] = False

        # Initialize params that change over time
        var_kw = dict(
            X=Xs[0],
            var=S_ts[0],
            rho=rhos[0],
            title=fr"$\rho$ = {rhos[0]}",
        )

        # Update which data is used for each run, in each frame
        def update_var_kw(col, f):
            var_kw.update(
                X=Xs[col],
                var=S_ts[col, frames[f]],
                rho=rhos[col],
                title=fr"$\rho$ = {rhos[col]}",
            )

        # Get params that don't change over time
        hex_static_kw = {k: kw[k] for k in kw.keys() - var_kw.keys()}

        # Animate title with time
        suptitle_fun = lambda f: fr"Time: {t[frames[f]] / delay:.2f} $\tau$"

        # Plot one frame of animation
        def make_frame(f):

            # Set title at top of figure
            plt.suptitle(suptitle_fun(f))

            for col, ax in enumerate(axs.flat):

                # Update plotting params
                update_var_kw(col, f)

                # Clear axis
                ax.clear()

                # Plot cell sheet
                lsig.viz.plot_hex_sheet(
                    ax=ax,
                    **var_kw,
                    **hex_static_kw,
                )

        # Initialize
        make_frame(0)
        plt.tight_layout()

        try:
            _writer = animation.writers[writer](fps=fps, bitrate=1800)
        except RuntimeError:
            print(
                "  The `ffmpeg` writer must be installed inside the runtime environment.      \n"
                "  Writer availability can be checked in the current enviornment by executing \n"
                "  `matplotlib.animation.writers.list()` in Python. Install location can be   \n"
                "  checked by running `which ffmpeg` on a command line/terminal."
            )
            raise

        _anim_FA = animation.FuncAnimation(
            fig, make_frame, frames=n_frames, interval=200
        )

        # Get path and print to output
        _fpath = save_dir.joinpath(f"constantdensity_simulation.mp4")
        print("Writing to:", _fpath.resolve().absolute())
        _anim_FA.save(
            _fpath,
            writer=_writer,
            dpi=dpi,
            progress_callback=lambda i, n: print(f"Frame {i+1} / {n}"),
        )


if __name__ == "__main__":
    main(
        # save=True,
        # n_frames=100,
    )
