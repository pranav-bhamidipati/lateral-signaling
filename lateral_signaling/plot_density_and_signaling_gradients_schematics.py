from pathlib import Path

import numpy as np
import pandas as pd
import numba

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as ptch
import matplotlib.collections as coll

import lateral_signaling as lsig

lsig.set_simulation_params()
lsig.set_steady_state_data()
lsig.viz.default_rcParams()

mle_csv = lsig.analysis_dir.joinpath("growth_parameters_MLE.csv")
im_png = Path("./culture_well.png")

save_dir = lsig.temp_plot_dir
save_prefix = save_dir.joinpath("signaling_gradient_schematic")


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


def make_schematic_1_no_callouts(
    im_png=im_png,
    figsize=(5, 3),
    scale=4.0,
):

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.axis("off")

    # Density gradient wedge
    wedge_coords = np.array(
        [
            [-1.2 * scale, -0.9 * scale],
            [-1.1 * scale, 0.9 * scale],
            [-1.3 * scale, 0.9 * scale],
        ]
    ).T.tolist()
    plt.fill(*wedge_coords, ec="None", fc="k")
    plt.text(
        -2.2 * scale,
        0,
        "Initial\ndensity\n$\\rho_0$",
        ha="center",
        va="center",
        fontdict=dict(size=18),
    )

    # Culture well
    im_extent = [-scale, scale, -scale, scale]
    img = mpimg.imread(im_png)
    plt.imshow(img, extent=im_extent)

    return fig, ax


def main(
    mle_csv=mle_csv,
    im_png=im_png,
    figsize1_nc=(4, 2),
    figsize1=(5, 3),
    scale=4.0,
    rho_0s=(4.0, 2.0, 1.0),
    tmax_days=4,
    nt=201,
    g=1.0,
    figsize2=(6, 2.5),
    cmap=lsig.viz.kgy,
    nt_sample=4,
    nx=201,
    dt_days=0.6,
    bg_clr="k",
    label_bias=0.1,
    save_prefix=save_prefix,
    save=True,
    dpi=300,
    fmt="png",
):
    rho_max = pd.read_csv(mle_csv, index_col="treatment").loc[
        "untreated", "rho_max_ratio"
    ]

    # bias_scaled = label_bias * scale

    fig, ax = make_schematic_1_no_callouts(
        im_png=im_png, figsize=figsize1_nc, scale=scale
    )
    plt.tight_layout()

    if save:
        _fpath = save_prefix.with_stem(save_prefix.stem + "_1_no_callouts").with_suffix(
            f".{fmt}"
        )
        _fpath = str(_fpath.resolve().absolute())
        print(f"Writing to: {_fpath}")
        plt.savefig(_fpath, dpi=dpi)

    # Callout boxes
    callout_ys = scale * np.array([0.5, -0.125, -0.5])
    callout_w = 0.6
    callout_h = 0.4
    callouts = [
        ptch.Rectangle((0, y), callout_w, callout_h, fc="None", ec="k", lw=2)
        for y in callout_ys
    ]
    p = coll.PatchCollection(callouts, match_original=True)

    fig, ax = make_schematic_1_no_callouts(im_png=im_png, figsize=figsize1, scale=scale)
    ax.add_collection(p)

    # inset axes with rho vs t examples
    inset_x = scale * 1.8
    inset_w = scale
    inset_h = 0.8 * inset_w
    inset_ys = -1.75 * scale + (1.75 * inset_h) * np.arange(3)
    inset_ys = inset_ys[::-1]

    tmax = tmax_days / lsig.t_to_units(1)
    t = np.linspace(0, tmax, nt)
    rho_ts = [lsig.logistic(t, g, r0, rho_max) for r0 in rho_0s]

    # for callout_y, inset_y, rho_t in zip(callout_ys, inset_ys, rho_ts):
    #     axins = ax.inset_axes(
    #         [inset_x, inset_y, inset_w, inset_h],
    #         transform=ax.transData,
    #     )
    #     axins.plot(t, rho_t, lw=2, c="k")
    #     axins.annotate("",  (callout_w, callout_h + callout_y), textcoords=(inset_x, inset_y + inset_h/2), arrowprops=dict(shrink=0.8))

    # callout_y = callout_ys[0]
    # inset_y = inset_ys[0]
    # rho_t = rho_ts[0]

    shrink = 0.8
    for callout_y, inset_y, rho_t in zip(callout_ys, inset_ys, rho_ts):

        src = np.array([callout_w, callout_y + callout_h / 2])
        dst = np.array([inset_x - inset_w / 5, inset_y + inset_h / 2])
        diff = dst - src

        src = src + (1 - shrink) / 2 * diff
        dst = dst - (1 - shrink) / 2 * diff

        ax.annotate(
            "",
            src,
            dst,
            xycoords="data",
            textcoords="data",
            arrowprops=dict(arrowstyle="-"),
        )

        axins = ax.inset_axes(
            [inset_x, inset_y, inset_w, inset_h],
            transform=ax.transData,
        )
        axins.plot(t, rho_t, lw=2, c="k")
        axins.plot(t, rho_max * np.ones_like(t), lw=1, c="k", linestyle="dashed")
        axins.set_xlabel(r"$t$", loc="right")
        axins.set_xticks([])
        axins.set_ylabel(r"$\rho$", y=0.7, ha="right", rotation="horizontal")
        axins.set_yticks([])
        axins.set_ylim(0, rho_max * 1.3)
        axins.spines.right.set_visible(False)
        axins.spines.top.set_visible(False)

    plt.xlim(-3 * scale, 3 * scale)
    plt.ylim(-1.5 * scale, 1.5 * scale)

    if save:
        _fpath = save_prefix.with_stem(save_prefix.stem + "_1").with_suffix(f".{fmt}")
        _fpath = str(_fpath.resolve().absolute())
        print(f"Writing to: {_fpath}")
        plt.savefig(_fpath, dpi=dpi)

    fig2 = plt.figure(2, figsize=figsize2)
    fig2.patch.set_facecolor(bg_clr)

    # ax1 = plt.subplot2grid((1, 5), (0, 0))
    # ax1.set_facecolor(bg_clr)
    # ax1.axis("off")
    # plt.xlim(-scale, scale)
    # plt.ylim(-scale, scale)
    # plt.text(
    #     -0.75 * scale,
    #     bias_scaled,
    #     "$[GFP]$",
    #     ha="center",
    #     va="bottom",
    #     fontdict=dict(color=cmap(1.0), size=20),
    # )
    # plt.text(
    #     -0.75 * scale,
    #     0,
    #     "at\nsteady-state",
    #     ha="center",
    #     va="top",
    #     fontdict=dict(color=cmap(1.0), size=14),
    # )

    #     ax1 = plt.subplot2grid((2, 5), (0, 0))
    #     ax1.set_facecolor(bg_clr)
    #     plt.xlim(-scale, scale)
    #     plt.ylim(-scale, scale)
    #     plt.text(-0.25 * scale, -0.75 * scale, "$[GFP]$ at\nsteady-state", ha="center", va="center", fontdict=dict(color="w", size=18))

    #     ax2 = plt.subplot2grid((2, 5), (1, 0))
    #     ax2.set_facecolor(bg_clr)
    #     fig2.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap), ax=ax2, location="top", aspect=4)

    psi = 1 / rho_max
    rho_bar = (rho_max - 1) / np.log(rho_max)

    x = np.linspace(0, 1, nx)
    time_step = dt_days / lsig.t_to_units(1)

    x_t = np.zeros((nt_sample, nx))
    SS_xs = np.zeros((nt_sample, nx))
    for i in range(nt_sample):
        t = (i + 1) * time_step
        rho_x = get_rho_x_t(x, t, psi, rho_bar, rho_max)
        SS_x = lsig.get_steady_state_mean(rho_x)

        x_t[i] = rho_x
        SS_xs[i] = SS_x

    SS_xs = lsig.normalize(SS_xs, SS_xs.min(), SS_xs.max())
    norm = mpl.colors.Normalize(0, 1)

    # Make a circle to clip the gradient into a circle
    rad = (nx - 1) / 2
    xx = np.linspace(0, nx - 1, nx * 2)
    yy = np.sqrt(rad ** 2 - (xx - rad) ** 2)
    circle = plt.fill_between(xx, rad + yy, rad - yy, lw=0, color="none")

    # axs = [ax1]
    axs = []
    for i, ss_x in enumerate(SS_xs):
        # ax = plt.subplot2grid((1, 5), (0, 1 + i))
        ax = plt.subplot2grid((1, 4), (0, i))
        ax.set_facecolor(bg_clr)
        ax.axis("off")

        # plt.plot(x, SS_x)
        gradient = plt.imshow(
            np.ones((nx, nx)) * ss_x[:, np.newaxis], cmap=cmap, norm=norm
        )
        gradient.set_clip_path(circle.get_paths()[0], transform=ax.transData)
        ax.set_aspect(1)

        plt.text(
            rad,
            2.5 * rad,
            fr"$t_{i + 1}$",
            ha="center",
            va="center",
            fontdict=dict(color="w", size=18),
        )

        axs.append(ax)

    # axs[-1].text(
    #     0.75 * scale,
    #     0.75 * scale,
    #     r"$[\mathrm{GFP}]_\mathrm{SS}$",
    #     ha="center",
    #     va="bottom",
    #     fontdict=dict(color=cmap(1.0), size=20),
    # )

    # ax1 = axs[0]
    # x0, x1, y0, y1 = ax1.axis()
    # width = (x1 - x0) * 2.4
    # # x0 += 2 * scale + 8 * label_bias
    # x0 += 8 * label_bias
    # x1 = x0 + width
    # y0 += 4 * label_bias
    # y1 -= 4 * label_bias
    # height = y1 - y0

    # border1 = ax1.vlines(
    #     x0, y0 + bias_scaled, y0 + height - bias_scaled, ec="w", linestyle="dotted"
    # )
    # border1.set_clip_on(False)

    # border2 = ax1.vlines(
    #     x0 + width,
    #     y0 + bias_scaled,
    #     y0 + height - bias_scaled,
    #     ec="w",
    #     linestyle="dotted",
    # )
    # border2.set_clip_on(False)

    # lbl1 = ax1.text(
    #     # x0 + 2 * bias_scaled,
    #     # y1 - 2 * bias_scaled,
    #     x0 + width / 2,
    #     y1,
    #     "Spatial gradient",
    #     ha="center",
    #     va="bottom",
    #     fontdict=dict(color="w", size=16),
    # )
    # lbl1.set_clip_on(False)

    # lbl2 = ax1.text(
    #     x0 - 6 * bias_scaled,
    #     y0 - 2 * bias_scaled,
    #     "Time",
    #     va="center",
    #     color="w",
    #     fontsize=18,
    # )
    # # text(x0 + 2 * bias_scaled, y1 - 2 * bias_scaled, "Spatial gradient", ha="left", va="top", fontdict=dict(color="w", size=14))
    # lbl2.set_clip_on(False)

    # arrow = ax1.arrow(
    #     x0 + 0.25 * width,
    #     y0 - 2 * bias_scaled,
    #     0.5 * width,
    #     0,
    #     ec="w",
    #     fc="w",
    #     head_width=0.1 * scale,
    # )
    # arrow.set_clip_on(False)

    if save:
        _fpath = save_prefix.with_stem(save_prefix.stem + "_2").with_suffix(f".{fmt}")
        _fpath = str(_fpath.resolve().absolute())
        print(f"Writing to: {_fpath}")
        plt.savefig(_fpath, dpi=dpi)


if __name__ == "__main__":
    main(
        save=True,
    )
