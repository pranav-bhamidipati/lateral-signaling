from pathlib import Path
import numba

import numpy as np
import pandas as pd

import colorcet as cc
import matplotlib.pyplot as plt

import lateral_signaling as lsig

lsig.default_rcParams()

mle_dir = Path("../data/growth_curves_MLE/")
mle_csv = mle_dir.joinpath("growth_parameters_MLE.csv")

save_dir = Path("../plots/tmp")
save_prefix = save_dir.joinpath("spatiotemporal_signaling")


@numba.njit
def get_rho_x_0(x, psi, rho_bar):
    """Number density of cells at time zero."""
    return np.log(psi) / (psi - 1) * rho_bar * psi ** x


@numba.njit
def get_rho_x_t(x, t, psi, rho_bar, rho_max):
    """
    Number density of cells over time.
    Initial condition is an exponential gradient, and growth
    follows the logistic equation.
    Diffusion and advection are assumed to be negligible.
    """
    rho_x_0 = get_rho_x_0(x, psi, rho_bar)
    return rho_max * rho_x_0 * np.exp(t) / (rho_max + rho_x_0 * (np.exp(t) - 1))


def main(
    # grad_mins=(0.01, 0.1),
    # grad_maxs=(2.0, 3.0),
    rho_bars=(0.5, 2.0),
    psis=(1 / 20, 1 / 10),
    rho_crit_lo=0.5,
    rho_crit_hi=3.0,
    nx=201,
    nt=6,
    tmax_days=6.0,
    figsize=(3, 2.25),
    save_prefix=save_prefix,
    save=False,
    fmt="png",
    dpi=300,
):
    """"""
    tmax = tmax_days / lsig.t_to_units(1)
    x = np.linspace(0, 1, nx)
    t = np.linspace(0, tmax, nt)
    t_x = np.repeat(t, nx).reshape(nt, nx)

    rho_max = pd.read_csv(mle_csv, index_col="treatment").loc[
        "untreated", "rho_max_ratio"
    ]
    rho_max = float(rho_max)

    # Where to put "time" label in density profile plots
    time_xys = [
        [(0.2, 1.0), (0.5, 5.5)],
        [(0.4, 1.5), (0.9, 6.0)],
    ]

    density_colors = plt.get_cmap("Blues")(np.linspace(0.2, 1, nt))
    ligand_colors = plt.get_cmap("Greens")(np.linspace(0.2, 1, nt))

    def make_steady_state_supplemental_plots():

        rho_space = np.logspace(-1, 1, 501)
        mean_ss, std_ss = lsig.get_steady_state_vector(rho_space)

        fig, ax1 = plt.subplots(figsize=figsize)
        plt.semilogx()
        plt.xlabel("Density")
        plt.ylabel(r"$[\mathrm{GFP}]_\mathrm{steady\, state}$")
        plt.plot(rho_space, mean_ss, lw=1)

        plt.tight_layout()

        if save:
            _fpath = save_prefix.with_stem(
                save_prefix.stem + "_steady_state_mean"
            ).with_suffix(f".{fmt}")
            _fpath = str(_fpath.resolve().absolute())
            print(f"Writing to: {_fpath}")
            plt.savefig(_fpath, dpi=dpi)

        fig, ax2 = plt.subplots(figsize=figsize)
        plt.semilogx()
        plt.xlabel("Density")
        plt.ylabel("Coeff. of variation")
        plt.ylim(-0.05, 0.2)
        plt.plot(rho_space, std_ss / mean_ss, lw=1)

        plt.tight_layout()

        if save:
            _fpath = save_prefix.with_stem(
                save_prefix.stem + "_steady_state_cov"
            ).with_suffix(f".{fmt}")
            _fpath = str(_fpath.resolve().absolute())
            print(f"Writing to: {_fpath}")
            plt.savefig(_fpath, dpi=dpi)

    def make_density_plots(
        fignum,
        psi,
        rho_bar,
        *args,
        _nx=201,
        _nt=201,
        _nx_sample=4,
        **kwargs,
    ):
        _x = np.linspace(0, 1, _nx)
        _t = np.linspace(t.min(), t.max(), _nt)
        _t_days = lsig.t_to_units(_t)
        _x_sample = np.linspace(0.2, 0.8, _nx_sample)
        _t_x = np.repeat(_t, _nx_sample).reshape(_nt, _nx_sample)

        bg_alpha = 0.35
        bg_hex = hex(int(bg_alpha * 256)).split("x")[-1]
        light_bg_clr = lsig.rgb2hex(lsig.kgy(0.7)[:3]) + bg_hex
        colors = plt.get_cmap("gray")(np.linspace(0.1, 0.9, _nx_sample))
        ylim = (-0.25, rho_max + 0.25)

        x_fill = [-0.1, 1.1, 1.1, -0.1]
        y_fill = [ylim[0], ylim[0], ylim[1], ylim[1]]
        y_fill_optimal_density = [rho_crit_lo, rho_crit_lo, rho_crit_hi, rho_crit_hi]

        # Solve logistic eqn to get density vs space at initial time and over time-course
        rho_x_0 = get_rho_x_0(_x, psi, rho_bar)
        rho_t_x = get_rho_x_t(_x_sample, _t_x, psi, rho_bar, rho_max).T

        fig, ax1 = plt.subplots(figsize=figsize)

        # ax1.spines["right"].set_visible(False)
        # ax1.spines["top"].set_visible(False)
        plt.xlabel("Position")
        plt.xlim(0, 1)
        plt.xticks([])
        plt.ylabel("Initial Density")
        plt.ylim(ylim)

        # plt.hlines(
        #     (rho_crit_lo, rho_crit_hi),
        #     0,
        #     1,
        #     linestyles="dashed",
        #     lw=0.5,
        #     color="gray",
        #     zorder=-1,
        # )

        plt.fill(x_fill, y_fill, light_bg_clr, zorder=-2)
        plt.fill(x_fill, y_fill_optimal_density, light_bg_clr, zorder=-2)

        # plt.fill(
        #     [xlim[0], xlim[1], xlim[1], xlim[0]],
        #     [rho_crit_lo, rho_crit_lo, rho_crit_hi, rho_crit_hi],
        #     light_bg_clr,
        # )

        # ax1.text(
        #     0.95,
        #     rho_crit_lo + 0.05,
        #     color="gray",
        #     ha="right",
        #     va="bottom",
        # )
        # ax1.text(
        #     0.95,
        #     rho_crit_lo + 0.05,
        #     "",
        #     color="gray",
        #     ha="right",
        #     va="bottom",
        # )

        plt.plot(_x, rho_x_0, color="k", lw=2)

        plt.tight_layout()

        if save:
            _fpath = save_prefix.with_stem(
                save_prefix.stem + f"_{fignum}_init_density"
            ).with_suffix(f".{fmt}")
            _fpath = str(_fpath.resolve().absolute())
            print(f"Writing to: {_fpath}")
            plt.savefig(_fpath, dpi=dpi)

        fig, ax2 = plt.subplots(figsize=figsize)

        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        plt.xlabel("Days")
        plt.ylabel("Density")
        plt.ylim(-0.25, rho_max + 0.25)

        plt.hlines(
            (rho_crit_lo, rho_crit_hi),
            0,
            _t_days.max(),
            linestyles="dashed",
            lw=1,
            color="gray",
            zorder=900,
        )

        for j, (rho_t) in enumerate(rho_t_x):
            plt.plot(
                _t_days,
                rho_t,
                color="k",
                # color=colors[j],
                lw=1,
            )

        plt.tight_layout()

        if save:
            _fpath = save_prefix.with_stem(
                save_prefix.stem + f"_{fignum}_density_dynamics"
            ).with_suffix(f".{fmt}")
            _fpath = str(_fpath.resolve().absolute())
            print(f"Writing to: {_fpath}")
            plt.savefig(_fpath, dpi=dpi)

    def make_profile_overlay_plots(
        fignum,
        psi,
        rho_bar,
        time_xy,
    ):

        # Solve logistic eqn to get density vs space, over time
        rho_x_t = get_rho_x_t(x, t_x, psi, rho_bar, rho_max)

        # Use data from simulations to get steady-state conc. of ligand (Sss)
        Sss_x_t_mean_std = np.array(list(map(lsig.get_steady_state_vector, rho_x_t)))
        Sss_x_t = Sss_x_t_mean_std[:, 0]
        Sss_x_t_std = Sss_x_t_mean_std[:, 1]

        # fig = plt.figure(figsize=figsize)
        # ax1 = fig.add_subplot(2, 1, 1)
        fig, ax1 = plt.subplots(figsize=figsize)

        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        plt.xlabel("Position")
        plt.xticks([])
        plt.ylabel("Density")

        plt.hlines(
            (rho_crit_lo, rho_crit_hi),
            0,
            1,
            linestyles="dashed",
            lw=1,
            color="gray",
            zorder=900,
        )

        for j, (rho_x) in enumerate(rho_x_t):
            plt.plot(x, rho_x, color=density_colors[j], lw=3)

        ax1.annotate(
            "Time",
            xy=time_xy[0],
            xytext=time_xy[1],
            color="gray",
            arrowprops=dict(
                facecolor="gray",
                edgecolor="gray",
                arrowstyle="<|-",
            ),
        )

        plt.tight_layout()

        if save:
            _fpath = save_prefix.with_stem(
                save_prefix.stem + f"_{fignum}_density_profiles"
            ).with_suffix(f".{fmt}")
            _fpath = str(_fpath.resolve().absolute())
            print(f"Writing to: {_fpath}")
            plt.savefig(_fpath, dpi=dpi)

        # ax2 = fig.add_subplot(2, 1, 2)
        fig, ax2 = plt.subplots(figsize=figsize)

        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        plt.xlabel("Position")
        plt.xticks([])
        plt.ylabel(r"$[\mathrm{GFP}]_\mathrm{steady\, state}$")
        for j, (ss_x) in enumerate(Sss_x_t):
            plt.plot(x, ss_x, color=ligand_colors[j], lw=3)

        plt.ylim(None, 1.7)
        ax2.annotate(
            "Time",
            xy=(0.75, 1.7),
            xytext=(0.25, 1.7),
            color="gray",
            va="center",
            ha="right",
            arrowprops=dict(
                facecolor="gray",
                edgecolor="gray",
                arrowstyle="-|>",
            ),
        )

        plt.tight_layout()

        if save:
            _fpath = save_prefix.with_stem(
                save_prefix.stem + f"_{fignum}_ligand_profiles"
            ).with_suffix(f".{fmt}")
            _fpath = str(_fpath.resolve().absolute())
            print(f"Writing to: {_fpath}")
            plt.savefig(_fpath, dpi=dpi)

    for i, args in enumerate(zip(psis, rho_bars, time_xys)):
        fignum = i + 1
        make_density_plots(fignum, *args)
        make_profile_overlay_plots(fignum, *args)

    make_steady_state_supplemental_plots()


if __name__ == "__main__":
    main(save=False)
