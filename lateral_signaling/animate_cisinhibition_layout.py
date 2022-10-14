import json
from copy import deepcopy
import h5py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation

import lateral_signaling as lsig


sacred_dir = lsig.simulation_dir.joinpath("20211119_sweep_cisparam/sacred")


def main(
    sacred_runs=(115, 117, 123),
    vmins=None,
    vmaxs=None,
    save_dir=lsig.plot_dir,
    save=False,
    writer="ffmpeg",
    n_frames=100,
    fps=20,
    dpi=300,
):

    # Read in data from experiments
    run_dirs = [sacred_dir.joinpath(str(r)) for r in sacred_runs]
    rd0 = run_dirs[0]

    with rd0.joinpath("config.json").open("r") as f:
        _config = json.load(f)
        g = _config["g"]
        rho_0 = _config["rho_0"]
        rho_max = _config["rho_max"]
        rows = _config["rows"]
        cols = _config["cols"]

    with h5py.File(rd0.joinpath("results.hdf5"), "r") as f:
        t = np.asarray(f["t"])

    nt = t.size

    rho_t = lsig.logistic(t, g, rho_0, rho_max)
    X = lsig.hex_grid(rows, cols)
    sender_idx = lsig.get_center_cells(X)
    X = X - X[sender_idx]
    n = X.shape[0]
    X_t = np.multiply.outer(1 / np.sqrt(rho_t), X)

    deltas = []
    t = []
    S_t_res = []
    R_t_res = []
    for i, rd in enumerate(run_dirs):
        with h5py.File(rd.joinpath("results.hdf5"), "r") as f:
            S_t_res.append(np.asarray(f["S_t"]))
            R_t_res.append(np.asarray(f["R_t"]))

        with rd.joinpath("config.json").open("r") as f:
            _config = json.load(f)
            if isinstance(_config["delta"], dict):
                _delta = _config["delta"]["value"]  # If serialized as Numpy value
            else:
                _delta = _config["delta"]  # If serialized as a float

        deltas.append(f"{_delta:.1f}")

    if save:

        # Make figure
        fig, axs = plt.subplots(
            nrows=1,
            ncols=3,
            figsize=(8.5, 2.5),
            #            gridspec_kw=dict(width_ratios=[1, 1, 1.2]),
        )

        X_func = lambda i: X_t[i]
        var_func = lambda r, i: S_t_res[r][i]
        rho_func = lambda i: rho_t[i]
        suptitle_fun = lambda i: f"Time: {t[i]:.2f} days"
        titles = [
            ("No inhibition\n", "Weak inhibition\n", "Strong inhibition\n")[r]
            + fr"$\delta={deltas[r]}$"
            for r in range(3)
        ]

        # Get default kwargs for plotting
        plot_kwargs = deepcopy(lsig.plot_kwargs)
        plot_kwargs["sender_idx"] = sender_idx

        # Get axis limits
        plot_kwargs["xlim"] = X_t[-1, :, 0].min(), X_t[-1, :, 0].max()
        plot_kwargs["ylim"] = X_t[-1, :, 1].min(), X_t[-1, :, 1].max()

        # Set some args for colorbar
        plot_kwargs["cmap"] = lsig.kgy
        plot_kwargs["cbar_aspect"] = 8
        plot_kwargs["cbar_kwargs"]["shrink"] = 0.7
        plot_kwargs["cbar_kwargs"]["format"] = "%.2f"
        plot_kwargs["extend"] = "neither"

        # Get colorscale limits
        ns_mask = np.ones(n, dtype=bool)
        ns_mask[sender_idx] = False
        if vmins is None:
            vmins = [v[:, ns_mask].min() for v in S_t_res]
        if vmaxs is None:
            vmaxs = [v[:, ns_mask].max() for v in S_t_res]

        # Make list to hold colorbars for easy reference
        colorbars = []

        # Turn off automatic colorbar plotting
        plot_kwargs["colorbar"] = False

        # Get frames to animate
        frames = lsig.vround(np.linspace(0, nt - 1, n_frames))

        # Get params that don't change over time
        static_kw = {
            k: plot_kwargs[k]
            for k in plot_kwargs.keys()
            - set(["X", "var", "rho", "title", "vmin", "vmax", "cbar_kwargs"])
        }

        def _anim(i):
            """Creates one frame of animation."""

            # Make each plot in layout
            for r, ax in enumerate(axs):

                # Get changing arguments
                var_kw = dict(
                    X=X_func(frames[i]),
                    var=var_func(r, frames[i]),
                    rho=rho_func(frames[i]),
                    title=titles[r],
                    vmin=vmins[r],
                    vmax=vmaxs[r],
                )

                # Plot frame of animation
                ax.clear()
                lsig.plot_hex_sheet(ax=ax, **var_kw, **static_kw)

            # Make colorbars if they don't exist
            if not colorbars:
                for r, ax in enumerate(axs):

                    _vmin = vmins[r]
                    _vmax = vmaxs[r]

                    _cbar_kwargs = dict(
                        ticks=[_vmin, _vmax],
                        format="%.1f",
                        label=("", "GFP (AU)")[r == (len(axs) - 1)],
                    )

                    cbar = plt.colorbar(
                        plt.cm.ScalarMappable(
                            norm=mpl.colors.Normalize(_vmin, _vmax),
                            cmap=plot_kwargs["cmap"],
                        ),
                        ax=ax,
                        aspect=plot_kwargs["cbar_aspect"],
                        extend=plot_kwargs["extend"],
                        **_cbar_kwargs,
                    )
                    colorbars.append(cbar)

            # Change suptitle
            suptitle = suptitle_fun(frames[i])
            plt.suptitle(suptitle)

            if i == 0:
                plt.tight_layout()

        try:
            _writer = animation.writers[writer](fps=fps, bitrate=1800)
        except RuntimeError:
            print(
                "The `ffmpeg` writer must be installed inside the runtime environment. \n"
                "Writer availability can be checked in the current enviornment by executing  \n"
                "`matplotlib.animation.writers.list()` in Python. Install location can be \n"
                "checked by running `which ffmpeg` on a command line/terminal."
            )

        _anim_FA = animation.FuncAnimation(fig, _anim, frames=n_frames, interval=200)

        # Clean up path
        _fpath = save_dir.joinpath("cis_inhibition_examples_animation.mp4")
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
