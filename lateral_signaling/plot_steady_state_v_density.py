import numpy as np
import numba
import matplotlib.pyplot as plt

import lateral_signaling as lsig

lsig.default_rcParams()

save_dir = lsig.temp_plot_dir
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
    rho_crit_lo=0.366,
    rho_crit_hi=3.0,
    nx=201,
    nt=6,
    tmax_days=6.0,
    figsize=(3, 2.25),
    bias=0.03,
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

    rho_max = float(lsig.mle_params.rho_max_ratio)

    # Where to put "time" label in density profile plots
    time_xys = [
        [(0.2, 1.0), (0.5, 5.5)],
        [(0.4, 1.5), (0.9, 6.0)],
    ]

    # density_colors = plt.get_cmap("Blues")(np.linspace(0.2, 1, nt))
    # ligand_colors = plt.get_cmap("Greens")(np.linspace(0.2, 1, nt))
    # ligand_colors = density_colors = plt.get_cmap("turbo")(np.linspace(0.05, 0.85, nt))
    ligand_colors = density_colors = plt.get_cmap("gray")(np.linspace(0.0, 0.75, nt))

    def make_steady_state_v_density_plots():

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
            _fpath = save_prefix.with_stem(
                save_prefix.stem + "_steady_state_mean"
            ).with_suffix(f".{fmt}")
            _fpath = str(_fpath.resolve().absolute())
            print(f"Writing to: {_fpath}")
            plt.savefig(_fpath, dpi=dpi)

        fig, ax2 = plt.subplots(figsize=figsize)
        plt.semilogx()
        plt.xlabel("Density")
        plt.ylabel(r"$\mathrm{COV}\left([\mathrm{GFP}]_\mathrm{SS}\right)$")
        plt.ylim(-0.01, 0.2)
        plt.plot(rho_space, std_ss / mean_ss, lw=1, color="k")

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
        xlim = (x.min(), x.max())
        ylim = (0, rho_max + 0.5)

        x_fill = [-0.1, 1.1, 1.1, -0.1]
        y_fill = [0, 0, ylim[1], ylim[1]]
        y_fill_optimal_density = [rho_crit_lo, rho_crit_lo, rho_crit_hi, rho_crit_hi]

        # Solve logistic eqn to get density vs space at initial time and over time-course
        rho_x_0 = get_rho_x_0(_x, psi, rho_bar)
        rho_t_x = get_rho_x_t(_x_sample, _t_x, psi, rho_bar, rho_max).T

        fig, ax1 = plt.subplots(figsize=figsize)

        # ax1.spines["right"].set_visible(False)
        # ax1.spines["top"].set_visible(False)
        plt.xlabel("Space")
        plt.xlim(0, 1)
        plt.xticks([])
        plt.ylabel("Initial Density")

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

        plt.xlim(xlim)
        plt.ylim(ylim)
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
        # Sss_x_t_mean_std = np.array(list(map(lsig.get_steady_state_vector, rho_x_t)))
        # Sss_x_t = Sss_x_t_mean_std[:, 0]
        # Sss_x_t_std = Sss_x_t_mean_std[:, 1]

        Sss_x_t = lsig.get_steady_state_mean(rho_x_t)
        Sss_x_t_std = lsig.get_steady_state_std(rho_x_t)

        # fig = plt.figure(figsize=figsize)
        # ax1 = fig.add_subplot(2, 1, 1)
        fig, ax1 = plt.subplots(figsize=figsize)

        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        plt.xlabel("Space")
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
        plt.xlabel("Space")
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

    def make_profile_overlay_plots2(
        psi,
        rho_bar,
        time_xy,
        bias=0.05,
    ):

        bg_alpha = 0.35
        bg_hex = hex(int(bg_alpha * 256)).split("x")[-1]
        light_bg_clr = lsig.rgb2hex(lsig.kgy(0.7)[:3]) + bg_hex
        xmin = x.min()
        xmax = x.max()
        ymin = 0
        ymax = rho_max + 1.0

        xbias = bias * (xmax - xmin)
        xmin -= xbias
        xmax += xbias

        x_fill = [xmin, xmax, xmax, xmin]
        y_fill = [0, 0, ymax, ymax]
        y_fill_optimal_density = [rho_crit_lo, rho_crit_lo, rho_crit_hi, rho_crit_hi]

        # Solve logistic eqn to get density vs space, over time
        rho_x_t = get_rho_x_t(x, t_x, psi, rho_bar, rho_max)

        # Use data from simulations to get steady-state conc. of ligand (Sss)
        # Sss_x_t_mean_std = np.array(list(map(lsig.get_steady_state_vector, rho_x_t)))
        # Sss_x_t = Sss_x_t_mean_std[:, 0]
        # Sss_x_t_std = Sss_x_t_mean_std[:, 1]

        Sss_x_t = lsig.get_steady_state_mean(rho_x_t)
        Sss_x_t_std = lsig.get_steady_state_std(rho_x_t)

        fig = plt.figure(figsize=(figsize[0], figsize[1] * 2))
        ax1 = fig.add_subplot(2, 1, 1)
        # fig, ax1 = plt.subplots(figsize=figsize)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)

        plt.xlabel("Space")
        plt.xticks([0, 1])
        plt.ylabel("Density")

        plt.fill(x_fill, y_fill, light_bg_clr, zorder=-2)
        plt.fill(x_fill, y_fill_optimal_density, light_bg_clr, zorder=-2)

        plt.hlines(
            rho_max,
            xmin,
            xmax,
            linestyles="dashed",
            lw=1,
            color="k",
            zorder=900,
        )
        plt.text(
            0.3,
            rho_max + 2 * xbias,
            r"$\rho_\mathrm{max}$",
            color="k",
            ha="center",
            va="bottom",
        )

        for j, (rho_x) in enumerate(rho_x_t):
            plt.plot(x, rho_x, color=density_colors[j], lw=1)
            # plt.plot(x, rho_x, color="k", lw=1)

        plt.text(0.9, 4.0, "Time", ha="center", va="bottom")
        hw = 0.03
        hl = 8 * hw
        plt.arrow(0.9, 0.5, 0.0, 3.0, head_width=hw, head_length=hl, ec="k", fc="k")
        # ax1.annotate(
        #     "Time",
        #     xy=(0.9, 0.0),
        #     ,
        #     color="k",
        #     arrowprops=dict(
        #         facecolor="k",
        #         edgecolor="k",
        #         arrowstyle="<|-",
        #     ),
        #     annotation_clip=False,
        # )

        # ax1.annotate(
        #     "Time",
        #     xy=(0.15, 0.5),
        #     xytext=(0.7, 5.0),
        #     color="k",
        #     arrowprops=dict(
        #         facecolor="k",
        #         edgecolor="k",
        #         arrowstyle="<|-",
        #     ),
        # )

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.tight_layout()

        ax2 = fig.add_subplot(2, 1, 2)
        # fig, ax2 = plt.subplots(figsize=figsize)

        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        plt.xlabel("Space")
        plt.xticks([0, 1])
        plt.ylabel(r"$[\mathrm{GFP}]_\mathrm{SS}$")
        for j, (ss_x) in enumerate(Sss_x_t):
            plt.plot(x, ss_x, color=ligand_colors[j], lw=1.5)

        plt.ylim(None, 1.7)
        ann = ax2.annotate(
            "Time",
            xy=(0.75, 1.7),
            xytext=(0.35, 1.7),
            color="k",
            va="center",
            ha="right",
            arrowprops=dict(
                facecolor="k",
                edgecolor="k",
                arrowstyle="-|>",
            ),
        )

        plt.tight_layout()

        if save:
            _fpath = save_prefix.with_stem(
                save_prefix.stem + "_density_and_ligand_profiles"
            ).with_suffix(f".{fmt}")
            _fpath = str(_fpath.resolve().absolute())
            print(f"Writing to: {_fpath}")
            plt.savefig(_fpath, dpi=dpi)

    for i, args in enumerate(zip(psis, rho_bars, time_xys)):
        # fignum = i + 1
        # make_density_plots(fignum, *args)
        # make_profile_overlay_plots(fignum, *args)

        if i == 0:
            make_profile_overlay_plots2(*args)

    make_steady_state_v_density_plots()


if __name__ == "__main__":
    main(
        save=True,
    )
