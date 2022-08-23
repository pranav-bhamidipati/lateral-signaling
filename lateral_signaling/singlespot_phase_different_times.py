import json
from pathlib import Path
import h5py

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import lateral_signaling as lsig


# Reading simulated data
data_dir = Path("../data/simulations/")
# sacred_dir = data_dir.joinpath("sacred")
sacred_dir = data_dir.joinpath("20211209_phase_2D/sacred")
thresh_fpath = data_dir.joinpath("phase_threshold.json")

# Reading growth parameter estimation data
mle_dir = Path("../data/growth_curves_MLE")
mle_fpath = mle_dir.joinpath("growth_parameters_MLE.csv")
pert_clr_json = mle_dir.joinpath("perturbation_colors.json")

# Writing
save_dir = Path("../plots/tmp")


def get_phase(actnum_t, v_init, v_init_thresh):

    # If activation doesn't happen immediately, signaling is attenuated
    if v_init < v_init_thresh:
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


def main(
    grad_t=2.7,
    grad_lo=2.0,
    grad_hi=5.0,
    grad_g=1.0,
    figsize=(3, 3),
    thresh_fpath=thresh_fpath,
    sacred_dir=sacred_dir,
    save_dir=save_dir,
    prefix="phase_diagram_2D_mpl",
    save=False,
    fmt="png",
    dpi=300,
):

    # Get threshold for v_init
    with thresh_fpath.open("r") as f:
        threshs = json.load(f)
        v_init_thresh = float(threshs["v_init_thresh"])

    # Read in phase metric data
    run_dirs = list(sacred_dir.glob("[0-9]*"))
    run_dirs = [rd for rd in run_dirs if rd.joinpath("config.json").exists()]

    # Extract metadata
    rd0 = run_dirs[0]
    with rd0.joinpath("config.json").open("r") as f:
        config = json.load(f)
        rho_max = config["rho_max"]
        g_space = config["g_space"]

    with h5py.File(str(rd0.joinpath("results.hdf5")), "r") as f:
        t = np.asarray(f["t"])

    grad_t_idx = np.minimum(np.searchsorted(t, grad_t), t.size)
    t_days = lsig.t_to_units(t)

    # Extract data for each run
    dfs = []
    for rd in run_dirs:

        _config_file = rd.joinpath("config.json")
        _results_file = rd.joinpath("results.hdf5")

        with _config_file.open("r") as c:
            config = json.load(c)
            rho_0 = config["rho_0"]

        if not (1.0 <= rho_0 <= rho_max):
            continue

        with h5py.File(str(_results_file), "r") as f:

            # Number of activated cells and density vs. time
            actnum_t_g = np.asarray(f["S_t_g_actnum"])

            # Initial velocity of activation
            v_init = np.asarray(f["v_init_g"])

        phase_g = [
            get_phase(actnum_t[:grad_t_idx], v_init_, v_init_thresh)
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

    # # Assign phases and sort by phase
    # sort_idx = np.argsort(rho_0s)
    # rho_0_space = np.asarray(rho_0s)[sort_idx]
    # actnum_t_rho_0 = np.asarray(actnum_ts)[sort_idx]
    # actnum = df.pivot(columns="g_inv_days", index="rho_0", values="actnum")
    # df["phase"] = (df.v_init > v_init_thresh).astype(int) * (
    #     1 + (df.n_act_fin > 0).astype(int)
    # )
    df = df.sort_values("phase")

    # Extract data ranges
    g_space = np.unique(df["g"])
    g_range = g_space[-1] - g_space[0]
    rho_0_space = np.unique(df["rho_0"])
    rho_0_range = rho_0_space[-1] - rho_0_space[0]

    # Colors for phase regions
    phase_colors = lsig.cols_blue[::-1]

    # # Options for different plot types
    # plot_kw = dict(
    #     xlim=xlim,
    #     ylim=ylim,
    #     xlabel=r"proliferation rate ($days^{-1}$)",
    #     xticks=(0.5, 1.0, 1.5),
    #     ylabel=r"init. density (x 100% confl.)",
    #     yticks=(0, 1, 2, 3, 4, 5, 6),
    #     hooks=[lsig.remove_RT_spines],
    #     fontscale=1.0,
    #     show_legend=False,
    #     #        aspect=1.,
    # )
    # bare_kw = dict(
    #     marker="s",
    #     edgecolor=None,
    #     s=60,
    #     color=hv.Cycle(phase_colors),
    # )
    # example_kw = dict(
    #     marker="s",
    #     s=60,
    #     edgecolor="k",
    #     linewidth=1.5,
    #     color=hv.Cycle(phase_colors),
    # )
    # text_init_kw = dict(
    #     fontsize=11,
    #     halign="left",
    # )
    # text_kw = dict(
    #     c="w",
    #     weight="normal",
    # )

    # Plot phase diagram
    data = df.pivot(columns="g_inv_days", index="rho_0", values="phase")
    data_vals = data.values

    g_space_inv_days = lsig.g_to_units(g_space)
    grad_g_inv_days = lsig.g_to_units(grad_g)

    phase_cmap = mpl.colors.ListedColormap(lsig.cols_blue[::-1])

    dr = rho_0_space[1] - rho_0_space[0]
    dg = g_space_inv_days[1] - g_space_inv_days[0]
    extent_r = (rho_0_space[0] - dr / 2, rho_max + dr / 2)
    extent_g = (g_space_inv_days[0] - dg / 2, g_space_inv_days[-1] + dg / 2)
    img_aspect = (extent_g[1] - extent_g[0]) / (extent_r[1] - extent_r[0])

    dg = (g_space_inv_days[1] - g_space_inv_days[0]) / 2
    dr = (rho_0_space[1] - rho_0_space[0]) / 2

    fig1 = plt.figure(1, figsize=figsize)
    ax = plt.gca()
    plt.imshow(
        data_vals,
        origin="lower",
        cmap=mpl.colors.ListedColormap(phase_colors, name="phase"),
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
    plt.text(-dr, rho_max, r"$\rho_\mathrm{max}$", ha="right", va="center", fontsize=12)

    plt.title(r"$t=2.7$ days")
    plt.xlabel(r"$g$ ($\mathrm{days}^{-1}$)")
    # plt.xlim(_xlim)
    # plt.xticks([])
    plt.ylabel(r"$\rho_0$")
    # plt.ylim(_ylim)
    # plt.yticks([])

    plt.tight_layout()

    if save:

        fpath = save_dir.joinpath(f"{prefix}_t{grad_t:.1f}_basic.{fmt}")
        print(f"Writing to: {str(fpath)}")
        plt.savefig(fpath, dpi=dpi)

    plt.vlines(grad_g_inv_days, grad_lo, grad_hi, colors="k", lw=3)
    plt.hlines(
        (grad_lo, grad_hi),
        grad_g_inv_days - dr / 2,
        grad_g_inv_days + dr / 2,
        colors="k",
        lw=2,
    )

    if save:

        fpath = save_dir.joinpath(f"{prefix}_t{grad_t:.1f}_gradient.{fmt}")
        print(f"Writing to: {str(fpath)}")
        plt.savefig(fpath, dpi=dpi)

    # Plot phase diagram
    g_wt = df.loc[np.isclose(df["g"].values, 1.0), "g_inv_days"].values[0]
    cols = data.columns.to_list()
    g_col_idx = cols.index(g_wt)
    g_col = cols[g_col_idx]
    g_col_phase = data[g_col].values
    data = data + 3
    data[g_col] = g_col_phase
    data = data.values
    # data.to_csv(Path(save_dir).joinpath("test.txt"), sep="\t")
    # 0 / 0
    g_space_inv_days = lsig.g_to_units(g_space)
    dg = (g_space_inv_days[1] - g_space_inv_days[0]) / 2
    dr = (rho_0_space[1] - rho_0_space[0]) / 2
    _xlim = g_space_inv_days[0] - dg, g_space_inv_days[-1] + dg
    _ylim = rho_0_space[0] - dr, rho_0_space[-1] + dr
    _yrange = _ylim[1] - _ylim[0]
    _aspect = (_xlim[1] - _xlim[0]) / (_ylim[1] - _ylim[0])

    fig2 = plt.figure(2, figsize=(2, 2))
    ax = plt.gca()
    plt.imshow(
        data,
        origin="lower",
        cmap=mpl.colors.ListedColormap(
            phase_colors + [lsig.blend_hex(c, lsig.white, 0.5) for c in phase_colors],
            name="phase",
        ),
        aspect=_aspect,
        extent=(*_xlim, *_ylim),
    )

    g_wt_xloc = lsig.normalize(g_wt, *_xlim)
    ax.annotate(
        "",
        xy=(g_wt_xloc, 1.0),
        xycoords="axes fraction",
        xytext=(g_wt_xloc, 1.15),
        arrowprops=dict(
            arrowstyle="->",
            color="k",
            linewidth=1.25,
            capstyle="projecting",
        ),
    )
    plt.xlabel(r"$g$")
    plt.xlim(_xlim)
    plt.xticks([])
    plt.ylabel(r"$\rho_0$")
    plt.ylim(_ylim)
    plt.yticks([])

    plt.tight_layout()

    if save:

        fpath = save_dir.joinpath(f"{prefix}_t{grad_t:.1f}_highlighted.{fmt}")
        print(f"Writing to: {str(fpath)}")
        plt.savefig(fpath, dpi=dpi)

    print()


if __name__ == "__main__":

    main(
        save=True,
    )
