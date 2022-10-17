import os
from glob import glob
import json
from copy import deepcopy
import h5py

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation

import lateral_signaling as lsig

sacred_dir = lsig.simulation_dir.joinpath("20220113_increasingdensity/sacred")


def main(
    save_dir=lsig.plot_dir,
    save=False,
    writer="ffmpeg",
    n_frames=100,
    fps=20,
    dpi=300,
):

    # Read in data from experiments
    run_dir = next(
        d for d in sacred_dir.glob("*") if d.joinpath("config.json").exists()
    )

    with run_dir.joinpath("config.json").open("r") as c:
        config = json.load(c)

        # Dimensions of cell sheet
        rows = config["rows"]
        cols = config["cols"]

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
        R_t = np.asarray(f["R_t"])

    # Make a lattice centered on the sender
    X = lsig.hex_grid(rows, cols)
    X = X - X[sender_idx]

    # Get mask of non-sender cells (transceivers)
    n = X.shape[0]
    ns_mask = np.ones(n, dtype=bool)
    ns_mask[sender_idx] = False

    # Compute cell positions vs. time
    X_t = np.multiply.outer(1 / np.sqrt(rho_t), X)

    ## Some manual plotting options
    # Font sizes
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    # Zoom in to a factor of `zoom` (to emphasize ROI)
    zoom = 0.45

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
            ncols=2,
            figsize=(6.0, 2.5),
        )

        # Get default kwargs for plotting
        plot_kwargs = deepcopy(lsig.viz.plot_kwargs)
        plot_kwargs["sender_idx"] = sender_idx

        # Turn on scalebar
        plot_kwargs["scalebar"] = True

        # Axis title
        plot_kwargs["title"] = ""

        # axis limits
        _xmax = np.abs(X_t[-1, :, 0]).max()
        _ymax = np.abs(X_t[-1, :, 1]).max()
        plot_kwargs["xlim"] = -_xmax * zoom, _xmax * zoom
        plot_kwargs["ylim"] = -_ymax * zoom, _ymax * zoom

        # colorscale limits
        plot_kwargs["vmin"] = 0
        vmax_S = S_t[:, ns_mask].max()
        vmax_R = R_t[:, ns_mask].max()

        # some args for colorbar
        cmap_S = lsig.viz.kgy
        cmap_R = plt.get_cmap("cet_kr")
        plot_kwargs["cbar_aspect"] = 8
        plot_kwargs["cbar_kwargs"] = dict(shrink=0.7, label="", format="%.2f")

        # Make colorbars
        cbar_S = plt.colorbar(
            plt.cm.ScalarMappable(
                norm=mpl.colors.Normalize(plot_kwargs["vmin"], vmax_S), cmap=cmap_S
            ),
            ax=axs[0],
            aspect=plot_kwargs["cbar_aspect"],
            extend=plot_kwargs["extend"],
            ticks=[plot_kwargs["vmin"], vmax_S],
            **plot_kwargs["cbar_kwargs"],
        )
        cbar_R = plt.colorbar(
            plt.cm.ScalarMappable(
                norm=mpl.colors.Normalize(plot_kwargs["vmin"], vmax_R), cmap=cmap_R
            ),
            ax=axs[1],
            aspect=plot_kwargs["cbar_aspect"],
            extend=plot_kwargs["extend"],
            ticks=[plot_kwargs["vmin"], vmax_R],
            **plot_kwargs["cbar_kwargs"],
        )

        # Turn off further colorbar plotting during animation
        plot_kwargs["colorbar"] = False

        # Initialize params that change over time
        var_kw = dict(
            X=X_t[0], var=S_t[0], rho=rho_t[0], vmax=vmax_S, cmap=cmap_S, title="Signal"
        )

        # Update which data is used for each run, in each frame
        def update_var_kw(E, f):
            var_kw.update(
                X=X_t[frames[f]],
                var=(S_t, R_t)[E][frames[f]],
                rho=rho_t[frames[f]],
                vmax=(vmax_S, vmax_R)[E],
                cmap=(cmap_S, cmap_R)[E],
                title=("Signal (AU)", "Reporter (AU)")[E],
            )

        # Get params that don't change over time
        hex_static_kw = {k: plot_kwargs[k] for k in plot_kwargs.keys() - var_kw.keys()}

        # Plot one frame of animation
        def make_frame(f):

            # Set title at top of figure
            plt.suptitle(f"Time: {t_days[frames[f]]:.2f} days")

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

        try:
            _writer = animation.writers[writer](fps=fps, bitrate=1800)
        except RuntimeError:
            print(
                "The `ffmpeg` writer must be installed inside the runtime environment. \n"
                "Writer availability can be checked in the current enviornment by executing  \n"
                "`matplotlib.animation.writers.list()` in Python. Install location can be \n"
                "checked by running `which ffmpeg` on a command line/terminal."
            )
            raise

        _anim_FA = animation.FuncAnimation(
            fig, make_frame, frames=n_frames, interval=200
        )

        # Get path and print to output
        _fpath = save_dir.joinpath("increasingdensity_simulation.mp4")
        print("Writing to:", _fpath.resolve().absolute())
        _anim_FA.save(
            _fpath,
            writer=_writer,
            dpi=dpi,
            progress_callback=lambda i, n: print(f"Frame {i+1} / {n}"),
        )


if __name__ == "__main__":
    main(
        save_dir=lsig.temp_plot_dir,
        save=True,
        # n_frames=5,
        # fps=1,
    )
