from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import lateral_signaling as lsig

lsig.default_rcParams()

save_dir = lsig.plot_dir


save_prefix = save_dir.joinpath("spatiotemporal_signaling")


def main(
    figsize=(5, 3),
    save_dir=save_dir,
    save=False,
    dpi=300,
    fmt="png",
):
    steady_state_v_density(
        figsize=figsize,
        save_dir=save_dir,
        save=save,
        dpi=dpi,
        fmt=fmt,
    )


def steady_state_v_density(
    figsize=(5, 3),
    bias=0.03,
    save_dir=save_dir,
    save=False,
    dpi=300,
    fmt="png",
):

    rho_crit_lo = lsig.phase_params.rho_ON
    rho_crit_hi = lsig.phase_params.rho_OFF

    bg_alpha = 0.35
    bg_hex = hex(int(bg_alpha * 256)).split("x")[-1]
    light_bg_clr = lsig.rgb2hex(lsig.kgy(0.7)[:3]) + bg_hex

    rho_space = np.logspace(-1, 1, 501)
    mean_ss = lsig.get_steady_state_mean(rho_space)
    std_ss = lsig.get_steady_state_std(rho_space)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    plt.semilogx()
    plt.xlabel("Density")
    plt.ylabel(r"$[\mathrm{GFP}]_\mathrm{SS}$")
    plt.plot(rho_space, mean_ss, lw=2, color="k")

    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    xbias = 10 ** (bias * (np.log10(xmax) - np.log10(xmin)))
    ybias = bias * (ymax - ymin)
    plt.vlines(
        (rho_crit_hi, rho_crit_lo),
        ymin,
        ymax,
        colors="k",
        linestyles="dashed",
        lw=1,
    )
    plt.text(
        rho_crit_lo / xbias,
        ymax,
        r"$\rho_\mathrm{OFF}$",
        ha="right",
        va="top",
        color="k",
        size="large",
    )
    plt.text(
        rho_crit_hi * xbias,
        ymax,
        r"$\rho_\mathrm{ON}$",
        ha="left",
        va="top",
        color="k",
        size="large",
    )

    x_fill = [xmin, xmax, xmax, xmin]
    y_fill = [ymin, ymin, ymax, ymax]
    x_fill_optimal_density = [rho_crit_lo, rho_crit_hi, rho_crit_hi, rho_crit_lo]

    plt.fill(x_fill, y_fill, light_bg_clr, zorder=-2)
    plt.fill(x_fill_optimal_density, y_fill, light_bg_clr, zorder=-2)

    plt.tight_layout()

    if save:
        _fpath = save_dir.joinpath(f"steady_state_vs_density.{fmt}")
        print(f"Writing to: {_fpath.resolve().absolute()}")
        plt.savefig(_fpath, dpi=dpi)

    fig, ax2 = plt.subplots(figsize=figsize)
    plt.semilogx()
    plt.xlabel("Density")
    plt.ylabel(r"$\mathrm{COV}\left([\mathrm{GFP}]_\mathrm{SS}\right)$")
    plt.ylim(-0.01, 0.2)
    plt.plot(rho_space, std_ss / mean_ss, lw=1, color="k")

    plt.tight_layout()


def steady_state_v_density2(
    figsize=(5, 3),
    bias=0.03,
    save_dir=save_dir,
    save=False,
    dpi=300,
    fmt="png",
):

    rho_crit_lo = lsig.phase_params.rho_ON
    rho_crit_hi = lsig.phase_params.rho_OFF

    bg_alpha = 0.35
    bg_hex = hex(int(bg_alpha * 256)).split("x")[-1]
    light_bg_clr = lsig.rgb2hex(lsig.kgy(0.7)[:3]) + bg_hex

    rho_space = np.logspace(-1, 1, 501)
    mean_ss = lsig.get_steady_state_mean(rho_space)
    # std_ss = lsig.get_steady_state_std(rho_space)
    ci_ss   = lsig.get_steady_state_ci(rho_space)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    plt.semilogx()
    plt.xlabel("Density")
    plt.ylabel(r"$[\mathrm{GFP}]_\mathrm{SS}$")
    plt.plot(rho_space, mean_ss, lw=2, color="k")

    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    xbias = 10 ** (bias * (np.log10(xmax) - np.log10(xmin)))
    ybias = bias * (ymax - ymin)
    
    #################################################
    # plt.fill_between(ci_ss)
    #################################################
    
    plt.vlines(
        (rho_crit_hi, rho_crit_lo),
        ymin,
        ymax,
        colors="k",
        linestyles="dashed",
        lw=1,
    )
    plt.text(
        rho_crit_lo / xbias,
        ymax,
        r"$\rho_\mathrm{OFF}$",
        ha="right",
        va="top",
        color="k",
        size="large",
    )
    plt.text(
        rho_crit_hi * xbias,
        ymax,
        r"$\rho_\mathrm{ON}$",
        ha="left",
        va="top",
        color="k",
        size="large",
    )

    x_fill = [xmin, xmax, xmax, xmin]
    y_fill = [ymin, ymin, ymax, ymax]
    x_fill_optimal_density = [rho_crit_lo, rho_crit_hi, rho_crit_hi, rho_crit_lo]

    plt.fill(x_fill, y_fill, light_bg_clr, zorder=-2)
    plt.fill(x_fill_optimal_density, y_fill, light_bg_clr, zorder=-2)

    plt.tight_layout()

    if save:
        _fpath = save_dir.joinpath(f"steady_state_vs_density.{fmt}")
        print(f"Writing to: {_fpath.resolve().absolute()}")
        plt.savefig(_fpath, dpi=dpi)
