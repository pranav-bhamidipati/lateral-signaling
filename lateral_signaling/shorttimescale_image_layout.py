import json
from copy import deepcopy
import h5py

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import holoviews as hv

hv.extension("matplotlib")

import lateral_signaling as lsig

sacred_dir = lsig.simulation_dir.joinpath("20220111_constantdensity/sacred")
save_dir = lsig.plot_dir


def main(
    sacred_dir=sacred_dir,
    save_dir=save_dir,
    delays_to_plot=[2, 4, 6],
    save=False,
    fmt="png",
    dpi=300,
):

    # Read in data from experiments
    run_dirs = list(sacred_dir.glob("[0-9]*"))
    run_dirs = [rd for rd in run_dirs if rd.joinpath("config.json").exists()]
    rd0 = run_dirs[0]

    with rd0.joinpath("config.json").open() as j:
        config = json.load(j)

        rows = config["rows"]
        cols = config["cols"]
        delay = config["delay"]

    with h5py.File(rd0.joinpath("results.hdf5"), "r") as f:

        t = np.asarray(f["t"])
        sender_idx = np.asarray(f["sender_idx"])

    # Define data to read
    rhos = []
    S_ts = []
    R_ts = []

    for rd in run_dirs:

        with h5py.File(rd.joinpath("results.hdf5"), "r") as f:
            rho = np.asarray(f["rho_t"])[0]  # Constant density
            S_t = np.asarray(f["S_t"])
            R_t = np.asarray(f["R_t"])

        rhos.append(rho)
        S_ts.append(S_t)
        R_ts.append(R_t)

    sort_rhos = np.argsort(rhos)
    rhos = np.asarray(rhos)[sort_rhos]
    S_ts = np.asarray(S_ts)[sort_rhos]
    R_ts = np.asarray(R_ts)[sort_rhos]

    # Convert selected time-points from units of delay
    plot_times = np.array([delay * p for p in delays_to_plot])

    # Get indices of the closest time-points
    plot_frames = np.argmin(np.subtract.outer(t, plot_times) ** 2, axis=0)

    # Make a lattice centered on the sender
    X = lsig.hex_grid(rows, cols)
    X = X - X[sender_idx]

    # Get mask of non-sender cells (transceivers)
    n = X.shape[0]
    ns_mask = np.ones(n, dtype=bool)
    ns_mask[sender_idx] = False

    # Get cell positions based on density
    Xs = np.multiply.outer(1 / np.sqrt(rhos), X)

    # Zoom in to a factor of `zoom` (to emphasize ROI)
    zoom = 0.7

    lsig.default_rcParams()

    # Get default kwargs for plotting
    plot_kwargs = deepcopy(lsig.plot_kwargs)
    plot_kwargs["sender_idx"] = sender_idx

    # Turn on scalebar
    plot_kwargs["scalebar"] = True

    # Axis title
    plot_kwargs["title"] = ""

    # axis limits
    densest_lattice = np.argmax(rhos)
    _xmax = np.abs(Xs[densest_lattice, :, 0]).max()
    _ymax = np.abs(Xs[densest_lattice, :, 1]).max()
    plot_kwargs["xlim"] = -_xmax * zoom, _xmax * zoom
    plot_kwargs["ylim"] = -_ymax * zoom, _ymax * zoom

    # colorscale limits
    plot_kwargs["vmin"] = 0
    plot_kwargs["vmax"] = S_ts[:, : (plot_frames.max() + 1), ns_mask].max()

    # some args for colorscale
    plot_kwargs["cmap"] = lsig.kgy
    plot_kwargs["cbar_aspect"] = 8
    plot_kwargs["colorbar"] = False

    # Make figure
    prows = 3
    pcols = 3
    fig, axs = plt.subplots(
        nrows=prows,
        ncols=pcols,
        figsize=(6.2, 5.0),
        gridspec_kw=dict(width_ratios=[1] * (pcols - 1) + [1.2]),
    )

    # Plot sheets
    for i, ax in enumerate(axs.flat):

        row = i // pcols
        col = i % pcols

        # Hide scalebar text except first image
        font_size = (i == 0) * 10
        plot_kwargs["sbar_kwargs"]["font_properties"] = dict(
            weight=1000,
            size=font_size,
        )

        # Plot cell sheet
        lsig.plot_hex_sheet(
            ax=ax,
            X=Xs[row],
            var=S_ts[row, plot_frames[col]],
            rho=rhos[row],
            **plot_kwargs,
        )

        # Make colorbars (empty except in first row)
        if col == pcols - 1:
            cbar = plt.colorbar(
                plt.cm.ScalarMappable(
                    norm=mpl.colors.Normalize(plot_kwargs["vmin"], plot_kwargs["vmax"]),
                    cmap=plot_kwargs["cmap"],
                ),
                ax=ax,
                aspect=plot_kwargs["cbar_aspect"],
                extend=plot_kwargs["extend"],
                shrink=(1e-5, 1.0)[row == 0],
                label="",
                ticks=[],
                #                format="%.2f",
            )

    plt.tight_layout()

    if save:

        _fpath = save_dir.joinpath(f"constant_density_img_layout.{fmt}")
        print("Writing to:", _fpath.resolve().absolute())
        plt.savefig(_fpath, dpi=dpi)


if __name__ == "__main__":
    main(
        save=True,
    )
