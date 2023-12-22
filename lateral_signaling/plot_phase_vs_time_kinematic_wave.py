import json
import h5py
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import lateral_signaling as lsig

lsig.set_simulation_params()
lsig.set_growth_params()
lsig.set_steady_state_data()
lsig.viz.default_rcParams()

sacred_dir = lsig.simulation_dir.joinpath("20220819_phase_logrho/sacred")
# sacred_dir = Path("./sacred")


def main(
    grad_ts=(2, 6, 10),
    grad_lo=0.05,
    grad_hi=3.0,
    rho_min=0.0,
    figsize=(3, 3),
    sacred_dir=sacred_dir,
    save_dir=lsig.plot_dir,
    save=False,
    dpi=300,
    fmt="png",
    bg_color="w",
):
    data_dirs = list(sacred_dir.glob("[0-9]*"))
    data_dirs = [d for d in data_dirs if d.joinpath("config.json").exists()]
    d0 = data_dirs[0]
    with open(d0.joinpath("config.json"), "r") as f:
        j = json.load(f)
        delay = j["delay"]
        rho_max = j["rho_max"]
        g_space = np.asarray(j["g_space"])
        tmax_days = j["tmax_days"]

    with h5py.File(d0.joinpath("results.hdf5"), "r") as h:
        t = np.asarray(h["t"])
        # print(*list(h.keys()), sep="\n")

    nt = t.size
    t_days = lsig.t_to_units(t)
    delay_days = lsig.t_to_units(delay)
    step_delay = (t_days <= delay_days).sum()
    g_idx = np.isclose(g_space, 1.0).nonzero()[0][0]

    rho_0s = []
    actnum_ts = []
    v_inits = []
    for d in data_dirs:
        with open(d.joinpath("config.json"), "r") as f:
            j = json.load(f)
            rho_0 = j["rho_0"]

        if rho_0 < rho_min:
            continue

        with h5py.File(d.joinpath("results.hdf5"), "r") as h:
            actnum_t = np.asarray(h["S_t_g_actnum"])[g_idx]
            v_init = np.asarray(h["v_init_g"])[g_idx]

        rho_0s.append(rho_0)
        actnum_ts.append(actnum_t)
        v_inits.append(v_init)

    sort_idx = np.argsort(rho_0s)
    rho_space = np.asarray(rho_0s)[sort_idx]
    actnum_t_rho_0 = np.asarray(actnum_ts)[sort_idx]

    rho_t = lsig.logistic(t[np.newaxis, :], 1.0, rho_space[:, np.newaxis], rho_max)
    SS_t = lsig.get_steady_state_mean(rho_t)
    area_t = lsig.ncells_to_area(actnum_t_rho_0, rho_t)

    phase_t = np.zeros(actnum_t_rho_0.shape, dtype=int)
    for i, actnum_t in enumerate(actnum_t_rho_0):
        activate_idx = lsig.first_nonzero(actnum_t)
        if activate_idx != -1:
            phase_t[i, activate_idx:] += 2
            deactivate_idx = lsig.first_zero(actnum_t[activate_idx:])
            if deactivate_idx != -1:
                phase_t[i, (activate_idx + deactivate_idx) :] -= 1

        # if init_rho_0[i]:
        #     deactivate_idx = np.where(np.diff(activated_t[i]) < 0)[0]
        #     if deactivate_idx.size > 0:
        #         deactivated_t[i, deactivate_idx[0] :] = 1

    # phase_t = deactivated_t + 2 * activated_t  # * init_rho_0[:, np.newaxis]

    phase_cmap = mpl.colors.ListedColormap(lsig.viz.cols_blue[::-1])

    dt = t_days[1] - t_days[0]
    extent_t = (t_days[0] - dt / 2, t_days[-1] + dt / 2)

    dr = rho_space[1] / rho_space[0]
    extent_r = (
        np.log10(rho_space[0] / np.sqrt(dr)),
        np.log10(rho_space[-1] * np.sqrt(dr)),
    )

    fig1 = plt.figure(1, figsize=figsize)
    ax = plt.gca()

    plt.imshow(
        phase_t,
        cmap=phase_cmap,
        aspect="auto",
        origin="lower",
        extent=(*extent_t, *extent_r),
        interpolation="nearest",
    )

    grad_rho_lims = (np.log10(grad_lo), np.log10(grad_hi))
    grad_ts_array = np.repeat(np.array(grad_ts), 2)
    plt.vlines(grad_ts, *grad_rho_lims, colors="k", lw=2)
    plt.hlines(
        grad_rho_lims * len(grad_ts),
        grad_ts_array - 0.5,
        grad_ts_array + 0.5,
        colors="k",
        lw=2,
    )

    plt.hlines(
        np.log10(rho_max),
        *extent_t,
        linestyles="dashed",
        colors="k",
        lw=2,
    )
    plt.text(
        -dr,
        np.log10(rho_max),
        r"$\rho_\mathrm{max}$",
        ha="right",
        va="center",
        fontsize=12,
    )

    # plt.xlabel("Days")
    plt.xlabel("Time", loc="right")
    plt.xticks([])
    plt.ylabel(r"$\rho_0$")
    yticks = np.arange(-2, 1)
    # yticklabels = 2 ** yticks
    yticklabels = [rf"$10^{{{i}}}$" for i in yticks]
    plt.yticks(ticks=yticks, labels=yticklabels)

    # t_OFF_days = lsig.get_t_OFF(lsig.mle_params.g_inv_days, rho_space)
    # t_ON_days = lsig.get_t_ON(lsig.mle_params.g_inv_days, rho_space)
    # _xylims = ax.axis()
    # plt.plot(t_OFF_days, np.log10(rho_space), linestyle="dashed", color="w", lw=1)
    # plt.plot(t_ON_days, np.log10(rho_space), linestyle="dashed", color="w", lw=1)
    # plt.xlim(_xylims[:2])
    # plt.ylim(_xylims[2:])

    plt.tight_layout()

    if save:
        fname = save_dir.joinpath(f"phase_vs_t_kinematicwave").with_suffix(f".{fmt}")
        print(f"Writing to: {fname.resolve().absolute()}")
        plt.savefig(fname, dpi=dpi, facecolor=bg_color)

    #####
    ## Run the same expreiment with low density also, scanning through rho_0
    ##  on a log-scale.
    ## Make a comparable plot with "error bars" at multiple time-points

    fig2 = plt.figure(2, figsize=figsize)
    ax = plt.gca()

    plt.imshow(
        SS_t,
        cmap=lsig.viz.kgy,
        aspect="auto",
        origin="lower",
        extent=(*extent_t, *extent_r),
        interpolation="nearest",
    )

    t_OFF_days = lsig.get_t_OFF(lsig.mle_params.g_inv_days, rho_space)
    t_ON_days = lsig.get_t_ON(lsig.mle_params.g_inv_days, rho_space)
    _xylims = ax.axis()
    plt.plot(t_OFF_days, np.log10(rho_space), linestyle="dashed", color="w", lw=1)
    plt.plot(t_ON_days, np.log10(rho_space), linestyle="dashed", color="w", lw=1)
    plt.xlim(_xylims[:2])
    plt.ylim(_xylims[2:])

    grad_rho_lims = (np.log10(grad_lo), np.log10(grad_hi))
    grad_ts_array = np.repeat(np.array(grad_ts), 2)
    plt.vlines(grad_ts, *grad_rho_lims, colors="k", lw=2)
    plt.hlines(
        grad_rho_lims * len(grad_ts),
        grad_ts_array - 0.5,
        grad_ts_array + 0.5,
        colors="w",
        lw=2,
    )

    plt.hlines(
        np.log10(rho_max),
        *extent_t,
        linestyles="dashed",
        colors="w",
        lw=2,
    )
    plt.text(
        -dr,
        np.log10(rho_max),
        r"$\rho_\mathrm{max}$",
        ha="right",
        va="center",
        fontsize=12,
    )

    plt.xlabel("Days")
    plt.ylabel(r"$\rho_0$")
    yticks = np.arange(-2, 1)
    # yticklabels = 2 ** yticks
    yticklabels = [rf"$10^{{{i}}}$" for i in yticks]
    plt.yticks(ticks=yticks, labels=yticklabels)

    plt.tight_layout()

    if save:
        fname = save_dir.joinpath(f"GFP_vs_t_kinematicwave").with_suffix(f".{fmt}")
        print(f"Writing to: {fname.resolve().absolute()}")
        plt.savefig(fname, dpi=dpi, facecolor=bg_color)


if __name__ == "__main__":
    main(
        save=True,
    )
