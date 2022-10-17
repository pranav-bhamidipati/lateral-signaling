import json
from copy import deepcopy
import h5py

import numpy as np
import colorcet as cc
import matplotlib.pyplot as plt
import matplotlib as mpl
import holoviews as hv

hv.extension("matplotlib")

import lateral_signaling as lsig

sacred_dir = lsig.simulation_dir.joinpath("20220113_increasingdensity", "sacred")


def main(
    plot_days=[],
    vmax_R_scale=1.0,
    print_updates=False,
    save_dir=lsig.plot_dir,
    save=False,
    fmt="png",
    dpi=300,
    transparent=True,
):

    # Read in data from experiments
    run_dir = next(sacred_dir.glob("[0-9]*"))

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

        # Density (constant)
        rho_t = np.asarray(f["rho_t"])

        # Signal and reporter expression vs. time
        S_t = np.asarray(f["S_t"])
        R_t = np.asarray(f["R_t"])

    # Get indices of the closest time-points
    plot_times = np.asarray(plot_days)
    plot_frames = np.argmin(np.subtract.outer(t_days, plot_times) ** 2, axis=0)

    # Make a lattice centered on the sender
    X = lsig.hex_grid(rows, cols)
    X = X - X[sender_idx]

    # Get mask of non-sender cells (transceivers)
    n = X.shape[0]
    ns_mask = np.ones(n, dtype=bool)
    ns_mask[sender_idx] = False

    # Get cell positions based on density
    X_t = np.multiply.outer(1 / np.sqrt(rho_t[plot_frames]), X)

    ## Some manual plotting options
    # Font sizes
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16

    # Zoom in to a factor of `zoom` (to emphasize ROI)
    zoom = 0.45

    # Set font sizes
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

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
    vmax_S = S_t[plot_frames][:, ns_mask].max()
    vmax_R = vmax_R_scale * R_t[plot_frames][:, ns_mask].max()

    # some args for colorscale
    cmap_S = lsig.viz.kgy
    cmap_R = cc.cm["kr"]
    plot_kwargs["cbar_aspect"] = 8
    plot_kwargs["colorbar"] = False

    # Make update rules for keywords that change between images
    var_kw = dict(
        X=X_t[0],
        var=S_t[0],
        rho=rho_t[0],
        cmap=cmap_S,
        vmax=vmax_S,
    )

    def update_var_kw(row, col):
        var_kw.update(
            X=X_t[col],
            var=(S_t, R_t)[row][plot_frames[col]],
            rho=rho_t[plot_frames[col]],
            cmap=(cmap_S, cmap_R)[row],
            vmax=(vmax_S, vmax_R)[row],
        )

    # Extract keywords that don't change
    kw = {k: plot_kwargs[k] for k in plot_kwargs.keys() - var_kw.keys()}

    # Make figure
    prows = 2
    pcols = plot_times.size
    fig, axs = plt.subplots(
        nrows=prows,
        ncols=pcols,
        figsize=(1.7 * pcols, 1.5 * prows),
        gridspec_kw=dict(width_ratios=[1] * (pcols - 1) + [1.25]),
    )

    if print_updates:
        print("Plotting images")

    # Plot sheets
    for i, ax in enumerate(axs.flat):

        row = i // pcols
        col = i % pcols

        # Hide scalebar text except first image
        kw["sbar_kwargs"]["font_properties"] = dict(
            weight=(i == 0) * 1000,
            size=(i == 0) * 10,
        )

        # Update kwargs
        update_var_kw(row, col)

        # Plot cell sheet
        lsig.viz.plot_hex_sheet(
            ax=ax,
            **var_kw,
            **kw,
        )

        if print_updates:
            print(f"\t {i+1} / {prows * pcols}")

        # Make colorbars in last column
        if col == pcols - 1:
            cmap_label = ("GFP", "mCherry")[row]
            cmap_extend = ("neither", "max")[row * (vmax_R_scale <= 1.0)]
            cbar = plt.colorbar(
                plt.cm.ScalarMappable(
                    norm=mpl.colors.Normalize(kw["vmin"], var_kw["vmax"]),
                    cmap=var_kw["cmap"],
                ),
                ax=ax,
                aspect=kw["cbar_aspect"],
                extend=cmap_extend,
                shrink=0.95,
                label=cmap_label,
                ticks=[],
                #                format="%.2f",
            )

    plt.tight_layout()

    if save:
        _fpath = save_dir.joinpath(f"increasing_density_imlayout.{fmt}")
        print("Writing to:", _fpath.resolve().absolute())
        plt.savefig(_fpath, dpi=dpi, transparent=transparent)


if __name__ == "__main__":
    main(
        print_updates=True,
        save=True,
        plot_days=np.arange(1, 8),
        vmax_R_scale=0.4,
    )
