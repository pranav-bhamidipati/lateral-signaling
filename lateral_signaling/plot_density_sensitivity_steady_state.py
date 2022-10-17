from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import lateral_signaling as lsig

lsig.viz.default_rcParams()

save_dir = lsig.plot_dir


def main(
    figsize=(4, 5),
    bias=0.03,
    save_dir=save_dir,
    save=False,
    dpi=300,
    fmt="png",
):

    rho_crit_lo = lsig.rho_crit_low
    rho_crit_hi = lsig.rho_crit_high

    bg_alpha = 0.35
    bg_hex = hex(int(bg_alpha * 256)).split("x")[-1]
    light_bg_clr = lsig.viz.rgb2hex(lsig.viz.kgy(0.7)[:3]) + bg_hex

    rho_space = np.logspace(-1, 1, 501)
    beta_func = lsig.get_beta_func(lsig.simulation_params.beta_function)
    beta_rho = beta_func(rho_space, *lsig.simulation_params.beta_args)
    mean_ss = lsig.get_steady_state_mean(rho_space)
    # ss_ci_lo, ss_ci_hi = lsig.get_steady_state_ci(rho_space)

    fig = plt.figure(figsize=figsize)

    plot_data = [
        (beta_rho, "Density sensitivity factor", r"$\beta$", False),
        (mean_ss, "Steady state GFP expression", r"$[\mathrm{GFP}]_\mathrm{SS}$", True),
    ]

    for i, (yvals, title, ylabel, plot_crits) in enumerate(plot_data):

        ax = fig.add_subplot(2, 1, i + 1)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.semilogx()
        plt.xlabel("Density")

        plt.ylabel(ylabel)
        plt.plot(rho_space, yvals, lw=2, color="k")
        plt.title(title)

        if plot_crits:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            xbias = 10 ** (bias * (np.log10(xmax) - np.log10(xmin)))
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
                r"$\rho_\mathrm{c}^\mathrm{low}$",
                ha="right",
                va="top",
                color="k",
                size="large",
            )
            plt.text(
                rho_crit_hi * xbias,
                ymax,
                r"$\rho_\mathrm{c}^\mathrm{high}$",
                ha="left",
                va="top",
                color="k",
                size="large",
            )

    plt.tight_layout()

    if save:
        _fpath = save_dir.joinpath(f"steady_state_vs_density.{fmt}")
        print(f"Writing to: {_fpath.resolve().absolute()}")
        plt.savefig(_fpath, dpi=dpi)


if __name__ == "__main__":
    main(
        save_dir=lsig.temp_plot_dir,
        save=True,
    )
