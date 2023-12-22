import json
from pathlib import Path
from typing import OrderedDict
import h5py

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import lateral_signaling as lsig

lsig.set_simulation_params()
lsig.set_steady_state_data()

# Reading simulated data
data_dir = Path("../data/simulations/")
log_sacred_dir = Path("sacred")
lin_sacred_dir = data_dir.joinpath("20211209_phase_2D/sacred")

# Writing
save_dir = Path("../plots/tmp")


def get_phase(actnum_t, v_init, v_init_thresh, rho_0):

    # If activation doesn't happen immediately, signaling is attenuated
    if (v_init < v_init_thresh) and (rho_0 > 1.0):
        return 0

    # Find time-point where activation first happens
    activate_idx = lsig.first_nonzero(actnum_t)
    if activate_idx != -1:

        # If there's deactivation, signaling was limited
        deactivate_idx = lsig.first_zero(actnum_t[activate_idx:])
        if deactivate_idx != -1:
            return 1

        # If no deactivation, unlimited
        else:
            return 2

    # If no activation, signaling is attenuated
    else:
        return 0


def make_plots_linear_rho_0(
    grad_t,
    grad_lo,
    grad_hi,
    grad_g=1.0,
    figsize=(3, 3),
    well_figsize=(2, 2),
    ny=201,
    sacred_dir=lin_sacred_dir,
    save_dir=save_dir,
    prefix="phase_diagram_2D_lin",
    atol=1e-8,
    save=False,
    fmt="png",
    dpi=300,
    **kwargs,
):

    v_init_thresh = lsig.simulation_params.v_init_thresh

    # Read in phase metric data
    data_dirs = list(sacred_dir.glob("[0-9]*"))
    data_dirs = [d for d in data_dirs if d.joinpath("config.json").exists()]

    # Extract metadata
    d0 = data_dirs[0]
    with d0.joinpath("config.json").open("r") as f:
        config = json.load(f)
        rho_max = config["rho_max"]
        g_space = config["g_space"]

    with h5py.File(str(d0.joinpath("results.hdf5")), "r") as f:
        t = np.asarray(f["t"])

    grad_t_idx = np.minimum(np.searchsorted(t, grad_t), t.size)
    t_days = lsig.t_to_units(t)

    # Extract data for each run
    dfs = []
    for d in data_dirs:

        _config_file = d.joinpath("config.json")
        _results_file = d.joinpath("results.hdf5")

        with _config_file.open("r") as c:
            config = json.load(c)
            rho_0 = config["rho_0"]

        if not (1.0 - atol <= rho_0 <= rho_max + atol):
            continue

        with h5py.File(str(_results_file), "r") as f:

            # Number of activated cells and density vs. time
            actnum_t_g = np.asarray(f["S_t_g_actnum"])

            # Initial velocity of activation
            v_init = np.asarray(f["v_init_g"])

        phase_g = [
            get_phase(actnum_t[:grad_t_idx], v_init_, v_init_thresh, rho_0)
            for actnum_t, v_init_ in zip(actnum_t_g, v_init)
        ]

        # Assemble dataframe
        _df = pd.DataFrame(
            dict(
                g=g_space,
                rho_0=rho_0,
                rho_max=rho_max,
                v_init=v_init,
                phase=phase_g,
            )
        )

        dfs.append(_df)

    # Concatenate into one dataset
    df = pd.concat(dfs).reset_index(drop=True)
    df["g_inv_days"] = lsig.g_to_units(df["g"].values)
    df = df.sort_values("phase")

    # Extract data ranges
    g_space = np.unique(df["g"])
    g_range = g_space[-1] - g_space[0]
    rho_0_space = np.unique(df["rho_0"])
    rho_0_range = rho_0_space[-1] - rho_0_space[0]

    # Colors for phase regions
    phase_colors = lsig.viz.cols_blue[::-1]
    phase_cmap = mpl.colors.ListedColormap(phase_colors)

    # Plot phase diagram
    data = df.pivot(columns="g_inv_days", index="rho_0", values="phase")
    data_vals = data.values

    g_space_inv_days = lsig.g_to_units(g_space)
    grad_g_inv_days = lsig.g_to_units(grad_g)

    dr = rho_0_space[1] - rho_0_space[0]
    dg = g_space_inv_days[1] - g_space_inv_days[0]
    extent_r = (rho_0_space[0] - dr / 2, rho_max + dr / 2)
    extent_g = (g_space_inv_days[0] - dg / 2, g_space_inv_days[-1] + dg / 2)
    img_aspect = (extent_g[1] - extent_g[0]) / (extent_r[1] - extent_r[0])

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    plt.imshow(
        data_vals,
        origin="lower",
        cmap=phase_cmap,
        aspect=img_aspect,
        extent=(*extent_g, *extent_r),
    )

    plt.hlines(
        rho_max,
        *extent_g,
        linestyles="dashed",
        colors="k",
        lw=2,
    )
    plt.text(
        -0.02, rho_max, r"$\rho_\mathrm{max}$", ha="right", va="center", fontsize=12
    )

    grad_t_days = lsig.t_to_units(grad_t)
    plt.title(fr"$t={{{grad_t_days:.1f}}}$ days")
    plt.xlabel(r"$g$ ($\mathrm{days}^{-1}$)")
    # plt.xlim(_xlim)
    # plt.xticks([])
    plt.ylabel(r"$\rho_0$")
    # plt.ylim(_ylim)
    # plt.yticks([])

    plt.text(
        0.1,
        4.0,
        "No signaling (NS)",
        c="w",
        fontsize=8,
        weight="bold",
        ha="left",
        va="center",
    )
    plt.text(
        0.1,
        2.6,
        "Signaling (S)",
        c="w",
        fontsize=8,
        weight="bold",
        ha="left",
        va="center",
    )

    # # Labeling a possible gradient on phase diagram
    # plt.vlines(grad_g_inv_days, grad_lo, grad_hi, colors="k", lw=2)
    # plt.hlines(
    #     (grad_lo, grad_hi),
    #     grad_g_inv_days - 0.1,
    #     grad_g_inv_days + 0.1,
    #     colors="k",
    #     lw=2,
    # )

    plt.tight_layout()

    if save:

        fpath = save_dir.joinpath(f"{prefix}_gradient.{fmt}")
        print(f"Writing to: {str(fpath)}")
        plt.savefig(fpath, dpi=dpi)

    rho_0_y = np.linspace(grad_lo, grad_hi, ny)
    rho_y = lsig.logistic(grad_t, 1.0, rho_0_y, rho_max)
    SS_y = lsig.get_steady_state_mean(rho_y)

    fig, ax = make_well_with_GFP_SS(
        well_figsize,
        SS_y,
        nx=ny,
    )

    if save:

        fpath = save_dir.joinpath(f"{prefix}_GFP.{fmt}")
        print(f"Writing to: {str(fpath)}")
        plt.savefig(fpath, dpi=dpi)


def make_well_with_GFP_SS(
    figsize,
    SS_y,
    rho_ON_y=None,
    rho_OFF_y=None,
    ylim=(0, 1),
    nx=201,
    bg_clr="k",
    plot_arrow=False,
):

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(bg_clr)

    # Make a circle to clip the gradient into a circle
    ymin, ymax = ylim
    ymid = (ymax + ymin) / 2
    rad = (ymax - ymin) / 2
    xx = np.linspace(-rad, rad, nx)
    yy = np.sqrt(rad ** 2 - xx ** 2)
    circle = plt.fill_between(xx, ymid + yy, ymid - yy, lw=0, color="none")

    ax = plt.gca()
    ax.set_facecolor(bg_clr)
    ax.axis("off")

    gradient = plt.imshow(
        np.ones((nx, nx)) * SS_y[:, np.newaxis],
        cmap=lsig.viz.kgy,
        extent=(-rad, rad, ymin, ymax),
        origin="lower",
    )
    gradient.set_clip_path(circle.get_paths()[0], transform=ax.transData)
    ax.set_aspect(1)

    _xlim = plt.xlim()
    _ylim = plt.ylim()
    _yrange = _ylim[1] - _ylim[0]

    if (rho_ON_y is not None) or (rho_OFF_y is not None):

        text_x = 0.9 * (_xlim[1] - _xlim[0]) + _xlim[0]

        if rho_ON_y is not None:

            plt.hlines(rho_ON_y, *_xlim, colors="w", linestyle="dotted")
            plt.text(
                text_x,
                rho_ON_y - 0.05 * _yrange,
                r"$\rho_\mathrm{ON}$",
                c="w",
                ha="center",
                va="top",
            )

        if rho_OFF_y is not None:

            plt.hlines(rho_OFF_y, *_xlim, colors="w", linestyle="dashed")
            plt.text(
                text_x,
                rho_OFF_y + 0.05 * _yrange,
                r"$\rho_\mathrm{OFF}$",
                c="w",
                ha="center",
                va="bottom",
            )

        if plot_arrow:

            arrow_min = rho_ON_y
            arrow_max = rho_OFF_y
            arrow_len = arrow_max - arrow_min
            arrow_min += 0.05 * arrow_len
            arrow_max -= 0.05 * arrow_len
            ax.annotate(
                "",
                xy=(text_x, arrow_min),
                xytext=(text_x, arrow_max),
                arrowprops=dict(arrowstyle="<->", color="w"),
            )

        plt.xlim(_xlim)
        plt.ylim(_ylim)

    return fig, ax


def make_plots_logarithmic_rho_0(
    grad_ts,
    grad_lo,
    grad_hi,
    rho_min=0.0,
    yticks_log10=(-2, -1, 0),
    figsize=(3, 3),
    well_figsize=(2, 2),
    grad_g=1.0,
    nscan=101,
    ny=201,
    well_ylim=(0, 1),
    sacred_dir=log_sacred_dir,
    save_dir=save_dir,
    prefix="phase_diagram_2D_log",
    atol=1e-8,
    save=False,
    fmt="png",
    dpi=300,
    **kwargs,
):

    v_init_thresh = lsig.simulation_params.v_init_thresh

    # Read in phase metric data
    data_dirs = list(sacred_dir.glob("[0-9]*"))
    data_dirs = [d for d in data_dirs if d.joinpath("config.json").exists()]

    # Make a dictionary assigning time-points to DataFrame columns
    time_dict = OrderedDict([(f"phase_t{i + 1}", gt) for i, gt in enumerate(grad_ts)])

    # Extract metadata
    d0 = data_dirs[0]
    with d0.joinpath("config.json").open("r") as f:
        config = json.load(f)
        rho_max = config["rho_max"]
        g_space = np.asarray(config["g_space"])
        delay = config["delay"]

    with h5py.File(str(d0.joinpath("results.hdf5")), "r") as f:
        t = np.asarray(f["t"])

    t_days = lsig.t_to_units(t)
    delay_days = lsig.t_to_units(delay)
    step_delay = (t_days <= delay_days).sum()

    g_idx = np.isclose(g_space, grad_g).nonzero()[0][0]
    g_space_inv_days = lsig.g_to_units(g_space)
    grad_g_inv_days = lsig.g_to_units(grad_g)

    # Extract data for each run
    dfs = []
    for d in data_dirs:

        _config_file = d.joinpath("config.json")
        _results_file = d.joinpath("results.hdf5")

        with _config_file.open("r") as c:
            config = json.load(c)
            rho_0 = config["rho_0"]

        if not (rho_min - atol <= rho_0 <= rho_max + atol):
            continue

        with h5py.File(str(_results_file), "r") as f:

            # Number of activated cells and density vs. time
            actnum_t_g = np.asarray(f["S_t_g_actnum"])

            # Initial velocity of activation
            v_init = np.asarray(f["v_init_g"])

        phase_dict = dict()
        for col, grad_t in time_dict.items():
            grad_t_idx = np.minimum(np.searchsorted(t, grad_t), t.size)
            phase_g = [
                get_phase(actnum_t[:grad_t_idx], v_init_, v_init_thresh, rho_0)
                for actnum_t, v_init_ in zip(actnum_t_g, v_init)
            ]
            phase_dict[col] = phase_g

        # Assemble dataframe
        _df = pd.DataFrame(
            dict(
                g=g_space,
                rho_0=rho_0,
                rho_max=rho_max,
                v_init=v_init,
                **phase_dict,
            )
        )

        dfs.append(_df)

    # Concatenate into one dataset
    df = pd.concat(dfs).reset_index(drop=True)
    df["g_inv_days"] = lsig.g_to_units(df["g"].values)
    df = df.sort_values([col for col in time_dict.keys()])

    # Extract data ranges
    g_space = np.unique(df["g"])
    g_range = g_space[-1] - g_space[0]
    rho_0_space = np.unique(df["rho_0"])
    rho_0_range = rho_0_space[-1] - rho_0_space[0]

    # Colors for phase regions
    phase_colors = lsig.viz.cols_blue[::-1]
    phase_cmap = mpl.colors.ListedColormap(phase_colors)

    # Calculate level sets for critical densities

    # Plot phase diagrams for different time-points
    dr = rho_0_space[1] / rho_0_space[0]
    extent_r = (
        np.log10(rho_0_space[0] / np.sqrt(dr)),
        np.log10(rho_0_space[-1] * np.sqrt(dr)),
    )

    dg = g_space_inv_days[1] - g_space_inv_days[0]
    extent_g = (g_space_inv_days[0] - dg / 2, g_space_inv_days[-1] + dg / 2)
    img_aspect = (extent_g[1] - extent_g[0]) / (extent_r[1] - extent_r[0])

    g_scan = np.linspace(g_space[0], g_space[-1], nscan)
    g_scan_inv_days = lsig.g_to_units(g_scan)
    rho_ON = lsig.rho_crit_low
    rho_OFF = lsig.rho_crit_high

    for i, (phase_col, grad_t) in enumerate(time_dict.items()):

        data = df.pivot(columns="g_inv_days", index="rho_0", values=phase_col)
        data_vals = data.values

        # critial density level sets
        rho_ON_levelset = lsig.logistic_solve_rho_0(rho_ON, grad_t, g_scan, rho_max)
        rho_OFF_levelset = lsig.logistic_solve_rho_0(rho_OFF, grad_t, g_scan, rho_max)
        rho_ON_levelset_log = np.log10(rho_ON_levelset)
        rho_OFF_levelset_log = np.log10(rho_OFF_levelset)

        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        plt.imshow(
            data_vals,
            origin="lower",
            cmap=phase_cmap,
            aspect=img_aspect,
            extent=(*extent_g, *extent_r),
        )

        plt.hlines(
            np.log10(rho_max),
            *extent_g,
            linestyles="dashed",
            colors="k",
            lw=2,
        )
        plt.text(
            -0.02,
            np.log10(rho_max),
            r"$\rho_\mathrm{max}$",
            ha="right",
            va="center",
            fontsize=12,
        )

        grad_t_days = lsig.t_to_units(grad_t)
        plt.title(fr"$t={{{grad_t_days:.1f}}}$ days")

        plt.xlabel(r"$g$ ($\mathrm{days}^{-1}$)")
        plt.ylabel(r"$\rho_0$")
        yticks = [k for k in yticks_log10 if np.isclose(k, int(k))]
        yticklabels = [fr"$10^{{{int(k)}}}$" for k in yticks]
        plt.yticks(yticks, yticklabels)

        # plt.vlines(
        #     grad_g_inv_days, np.log10(grad_lo), np.log10(grad_hi), colors="k", lw=2
        # )
        # plt.hlines(
        #     (np.log10(grad_lo), np.log10(grad_hi)),
        #     grad_g_inv_days - 0.1,
        #     grad_g_inv_days + 0.1,
        #     colors="k",
        #     lw=2,
        # )

        _xlim = plt.xlim()
        _ylim = plt.ylim()
        plt.plot(g_scan_inv_days, rho_ON_levelset_log, c="w", linestyle="dotted")
        plt.plot(g_scan_inv_days, rho_OFF_levelset_log, c="w", linestyle="dashed")
        plt.xlim(_xlim)
        plt.ylim(_ylim)

        if i == 0:
            # Labels for crtical density range
            label_idx = int(0.03 * nscan)
            plt.text(
                g_scan_inv_days[label_idx],
                rho_ON_levelset_log[label_idx] + np.log10(dr),
                r"$\rho_\mathrm{ON}$",
                ha="left",
                va="bottom",
                c="w",
            )

            plt.text(
                g_scan_inv_days[label_idx],
                rho_OFF_levelset_log[label_idx] - np.log10(dr),
                r"$\rho_\mathrm{OFF}$",
                ha="left",
                va="top",
                c="w",
            )

            arrow_idx = int(0.65 * nscan)
            arrow_g = g_scan_inv_days[arrow_idx]
            arrow_min = rho_ON_levelset_log[arrow_idx]
            arrow_max = rho_OFF_levelset_log[arrow_idx]
            arrow_len = arrow_max - arrow_min
            arrow_min += 0.05 * arrow_len
            arrow_max -= 0.05 * arrow_len
            # plt.arrow(
            #     arrow_g,
            #     arrow_min,
            #     0,
            #     arrow_max - arrow_min,
            #     ec="w",
            #     fc="w",
            #     head_width=0.05,
            # )
            ax.annotate(
                "",
                xy=(arrow_g, arrow_min),
                xytext=(arrow_g, arrow_max),
                arrowprops=dict(arrowstyle="<->", color="w"),
            )

        plt.tight_layout()

        if save:

            fpath = save_dir.joinpath(f"{prefix}_t{phase_col[-1]}_gradient.{fmt}")
            print(f"Writing to: {str(fpath)}")
            plt.savefig(fpath, dpi=dpi)

        grad_lo_after_growth, grad_hi_after_growth = lsig.logistic(
            grad_t, 1.0, np.array([grad_lo, grad_hi]), rho_max
        )

        rho_0_y = np.geomspace(grad_lo, grad_hi, ny)
        rho_y = lsig.logistic(grad_t, 1.0, rho_0_y, rho_max)
        SS_y = lsig.get_steady_state_mean(rho_y)

        y = np.linspace(*well_ylim, ny)
        if rho_y.min() < rho_ON < rho_y.max():
            rho_ON_y = y[np.minimum(np.searchsorted(rho_y, rho_ON), ny - 1)]
        else:
            rho_ON_y = None
        if rho_y.min() < rho_OFF < rho_y.max():
            rho_OFF_y = y[np.minimum(np.searchsorted(rho_y, rho_OFF), ny - 1)]
        else:
            rho_OFF_y = None

        plot_arrow = i == 1
        fig, ax = make_well_with_GFP_SS(
            well_figsize,
            SS_y,
            rho_ON_y,
            rho_OFF_y,
            nx=ny,
            ylim=well_ylim,
            plot_arrow=plot_arrow,
        )

        if save:

            fpath = save_dir.joinpath(f"{prefix}_t{phase_col[-1]}_GFP.{fmt}")
            print(f"Writing to: {str(fpath)}")
            plt.savefig(fpath, dpi=dpi)


def main(
    linear_grad_t=1.66,
    linear_grad_lo=2.0,
    linear_grad_hi=5.0,
    logarithmic_grad_ts=(1, 3, 5),
    logarithmic_grad_lo=0.02,
    logarithmic_grad_hi=2.0,
    **kwargs,
):

    make_plots_linear_rho_0(
        grad_t=linear_grad_t, grad_lo=linear_grad_lo, grad_hi=linear_grad_hi, **kwargs
    )

    make_plots_logarithmic_rho_0(
        grad_ts=logarithmic_grad_ts,
        grad_lo=logarithmic_grad_lo,
        grad_hi=logarithmic_grad_hi,
        **kwargs,
    )


if __name__ == "__main__":

    log_grad_ts = np.array([1, 3, 5]) / lsig.t_to_units(1)

    main(
        save=True,
        figsize=(2.3, 2.3),
        logarithmic_grad_ts=log_grad_ts,
        # logarithmic_grad_ts=(0.92403328, 1.84806656, 2.77209984),
    )
