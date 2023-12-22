import os
from glob import glob
import json
import h5py

import numpy as np
import pandas as pd
from tqdm import tqdm

import holoviews as hv

hv.extension("matplotlib")

import matplotlib.pyplot as plt
import matplotlib as mpl

import lateral_signaling as lsig

lsig.set_simulation_params()


# Reading simulated data
sacred_dir = lsig.simulation_dir.joinpath("20211209_phase_2D/sacred")
examples_dir = lsig.simulation_dir.joinpath("20211209_phase_examples/sacred")
examples_json = lsig.simulation_dir.joinpath("phase_examples.json")

# Reading growth parameter estimation data
mle_fpath = lsig.analysis_dir.joinpath("growth_parameters_MLE.csv")
pert_clr_json = lsig.data_dir.joinpath("growth_curves_MLE", "perturbation_colors.json")


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


def main(
    pad=0.05,
    area_ceiling=1,
    bg_alpha=0.8,
    marker_dim="max_sqrtarea_mm",
    marker_scale=130,
    legend_bgcol="#bebebe",
    legend_width=0.8,
    legend_pt_ypos=0.5,
    save_dir=lsig.plot_dir,
    save=False,
    prefix="phase_diagram_2D",
    fmt="png",
    dpi=300,
):
    ## Read in and assemble data
    # Get phase example values
    with open(examples_json, "r") as f:
        ex_dict = json.load(f)
        ex_name = ex_dict["name"]
        ex_xval = ex_dict["g"]
        ex_yval = ex_dict["rho_0"]
        ex_xlabel = ex_dict["label_x"]
        ex_ylabel = ex_dict["label_y"]
        ex_params = np.array([ex_xval, ex_yval]).T

    v_init_thresh = lsig.simulation_params.v_init_thresh

    # Read in phase metric data
    run_dirs = [d for d in sacred_dir.glob("*") if d.joinpath("config.json").exists()]

    # Store each run's data and in a DataFrame
    dfs = []
    rho_0_max = 0.0
    rho_0_min = 1.0
    for rd_idx, rd in enumerate(tqdm(run_dirs)):
        # Get some info from the run configuration
        with rd.joinpath("config.json").open("r") as c:
            config = json.load(c)

            # Intrinsic growth rate, initial density, carrying capacity
            rho_0 = config["rho_0"]
            g = config["g"]
            rho_max = config["rho_max"]
            g_space = config["g_space"]

        if rho_0 < rho_0_min:
            rho_0_min = rho_0

        if rho_0 > rho_0_max:
            rho_0_max = rho_0

        if not (1.0 <= rho_0 <= rho_max):
            continue

        # Get remaining info from run's data dump
        with h5py.File(rd.joinpath("results.hdf5"), "r") as f:
            t = np.asarray(f["t"])

            # Number of activated cells and density vs. time
            n_act_t_g = np.asarray(f["S_t_g_actnum"])
            rho_t_g = np.asarray(f["rho_t_g"])

            # Phase metrics
            v_init_g = np.asarray(f["v_init_g"])
            n_act_fin = np.asarray(f["n_act_fin_g"])

        # Calculate maximum area for each param set in batch
        maxArea_g = lsig.ncells_to_area(n_act_t_g, rho_t_g).max(axis=1)

        # Calculate phase
        phase_g = [
            get_phase(n_act_t, v_init, v_init_thresh, rho_0)
            for n_act_t, v_init in zip(n_act_t_g, v_init_g)
        ]

        # Assemble dataframe
        _df = pd.DataFrame(
            dict(
                g=g_space,
                rho_0=rho_0,
                rho_max=rho_max,
                v_init=v_init_g,
                n_act_fin=n_act_fin,
                max_area_mm2=maxArea_g,
                phase=phase_g,
            )
        )
        dfs.append(_df)

    # Concatenate into one dataset
    df = pd.concat(dfs).reset_index(drop=True)
    df["g_inv_days"] = lsig.g_to_units(df["g"].values)

    # Assign phases and sort by phase
    # df["phase"] = (df.v_init > v_init_thresh).astype(int) * (
    #     1 + (df.n_act_fin > 0).astype(int)
    # )
    df = df.sort_values("phase")

    # Extract data ranges
    g_space = np.unique(df["g"])
    g_range = g_space[-1] - g_space[0]
    rho_0_space = np.unique(df["rho_0"])
    rho_0_range = rho_0_space[-1] - rho_0_space[0]

    # Find phase examples in data
    df["example"] = ""
    for x, y, l in zip(ex_xval, ex_yval, ex_name):
        df.loc[np.isclose(df["g"], x) & np.isclose(df["rho_0"], y), "example"] = l

    ## Plot basic phase diagram
    # Set axis limits
    xlim = tuple(
        [
            lsig.g_to_units(g_space[0] - pad * g_range),
            lsig.g_to_units(g_space[-1] + pad * g_range),
        ]
    )
    x_range = xlim[1] - xlim[0]
    ylim = tuple(
        [
            rho_0_space[0] - pad * rho_0_range,
            rho_0_max + pad * rho_0_range,
        ]
    )
    y_range = ylim[1] - ylim[0]

    # Colors for phase regions
    phase_colors = lsig.viz.cols_blue[::-1]

    data = df.pivot(columns="g_inv_days", index="rho_0", values="phase")
    g_wt = df.loc[np.isclose(df["g"].values, 1.0), "g_inv_days"].values[0]
    cols = data.columns.to_list()
    g_col_idx = cols.index(g_wt)
    g_col = cols[g_col_idx]
    g_col_phase = data[g_col].values
    data = data + 3
    data[g_col] = g_col_phase
    data = data.values

    g_space_inv_days = lsig.g_to_units(g_space)
    dg = g_space_inv_days[1] - g_space_inv_days[0]
    dr = rho_0_space[1] - rho_0_space[0]
    _xlim = g_space_inv_days[0] - dg / 2, g_space_inv_days[-1] + dg / 2
    _ylim = rho_0_space[0] - dr / 2, rho_0_space[-1] + dr / 2
    _yrange = _ylim[1] - _ylim[0]
    _aspect = (_xlim[1] - _xlim[0]) / (_ylim[1] - _ylim[0])

    ## [Matplotlib] A phase diagram "highlighted" at one value of `g`
    fig1 = plt.figure(1, figsize=(2, 2))
    ax = plt.gca()
    plt.imshow(
        data,
        origin="lower",
        cmap=mpl.colors.ListedColormap(
            phase_colors
            + [lsig.viz.blend_hex(c, lsig.viz.white, 0.5) for c in phase_colors],
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

    if save:
        _fname = save_dir.joinpath(f"{prefix}_highlighted.{fmt}")
        print(f"Writing to: {_fname.resolve().absolute()}")
        plt.savefig(_fname, dpi=dpi)

    ## [Holoviews] Various versions of same phase diagram
    # Options for different plot types
    plot_kw = dict(
        xlim=xlim,
        ylim=ylim,
        xlabel=r"Proliferation rate ($\mathrm{days}^{-1}$)",
        xticks=(0.5, 1.0, 1.5),
        ylabel=r"Init. density (x $1250\,\mathrm{mm}^{-2}$)",
        yticks=(0, 1, 2, 3, 4, 5, 6),
        hooks=[lsig.viz.remove_RT_spines],
        fontscale=1.0,
        show_legend=False,
        aspect=((xlim[1] - xlim[0]) / dg) / ((ylim[1] - ylim[0]) / dr),
    )
    bare_kw = dict(
        marker="s",
        edgecolor=None,
        s=60,
        color=hv.Cycle(phase_colors),
    )
    example_kw = dict(
        marker="s",
        s=60,
        edgecolor="k",
        linewidth=1.5,
        color=hv.Cycle(phase_colors),
    )
    text_init_kw = dict(
        fontsize=11,
        halign="left",
    )
    text_kw = dict(
        c="w",
        weight="normal",
    )

    # Make bare phase diagram
    phasediagram_bare = (
        hv.Scatter(
            data=df,
            kdims=["g_inv_days"],
            vdims=["rho_0", "phase"],
        )
        .groupby("phase")
        .opts(
            **bare_kw,
            **plot_kw,
        )
        .overlay()
    )

    # Phase diagram with examples
    examples = (
        hv.Scatter(
            data=df.loc[df["example"].str.len() > 0],
            kdims=["g_inv_days"],
            vdims=["rho_0", "phase"],
        )
        .groupby("phase")
        .opts(**example_kw)
        .overlay()
    )

    phasediagram_ex = (phasediagram_bare * examples).opts(**plot_kw)

    # Phase labels
    text_labels = [
        hv.Text(x, y, l, **text_init_kw)
        for x, y, l in zip(ex_xlabel, ex_ylabel, ex_name)
    ]

    # Labeled phase diagram
    phasediagram_labeled = hv.Overlay([phasediagram_bare, examples, *text_labels]).opts(
        hv.opts.Text(**text_kw),
        hv.opts.Overlay(**plot_kw),
    )

    if save:
        for name, hvplot in [
            ("bare", phasediagram_bare),
            ("examples", phasediagram_ex),
            ("labeled", phasediagram_labeled),
        ]:
            _fname = save_dir.joinpath(f"{prefix}_{name}.{fmt}")
            print(f"Writing to: {_fname.resolve().absolute()}")
            hv.save(hvplot, _fname, fmt=fmt, dpi=dpi)

    # Isolate aggregate data for examples
    ex_df = df.loc[df["example"].str.len() > 0]

    # Read data for phase examples
    ex_dirs = [d for d in examples_dir.glob("*") if d.joinpath("config.json").exists()]
    ex_dfs = []
    for rd_idx, rd in enumerate(ex_dirs):
        # Get some info from the run configuration
        with rd.joinpath("config.json").open("r") as c:
            config = json.load(c)

            # Expression threshold
            k = config["k"]

            # Intrinsic growth rate, initial density, carrying capacity
            g = config["g"]
            rho_0 = config["rho_0"]

        # Get remaining info from run's data dump
        with h5py.File(rd.joinpath("results.hdf5"), "r") as f:
            # Time-course and density
            t = np.asarray(f["t"])
            t_days = lsig.t_to_units(t)
            rho_t = np.asarray(f["rho_t"])

            # Expression
            S_t = np.asarray(f["S_t"])

        # Get phase
        phase, label = ex_df.loc[
            np.isclose(ex_df["g"].values, g) & np.isclose(ex_df["rho_0"].values, rho_0)
        ][["phase", "example"]].values.flat

        # Calculate sqrt(Area) over time
        n_act_t = (S_t > k).sum(axis=1) - 1
        sqrtA_mm_t = np.sqrt(lsig.ncells_to_area(n_act_t, rho_t))

        # Store data for this example
        _df = pd.DataFrame(
            dict(
                g=g,
                rho_0=rho_0,
                phase=phase,
                label=label,
                t_days=t_days,
                sqrtA_mm_t=sqrtA_mm_t,
            )
        )
        ex_dfs.append(_df)

    # Assemble data
    example_data = pd.concat(ex_dfs).reset_index(drop=True)

    ## Make plot
    # Plotting options
    ex_plot_kw = lambda i: dict(
        linewidth=10,
        color=phase_colors[i],
        xlabel=("", "days", "")[i],
        xlim=(0, 8),
        xticks=(0, 8),
        ylabel=(r"$\sqrt{Area}$ ($mm$)", "", "")[i],
        yticks=([0, 0.2, 0.4, 0.6], 0, 0)[i],
        ylim=(-0.05, 0.75),
        fontscale=3,
    )
    layout_kw = dict(
        hspace=0.25,
        sublabel_size=0,
    )

    # Plot
    example_plots = [
        hv.Curve(
            data=example_data.loc[example_data.phase == i],
            kdims=["t_days"],
            vdims=["sqrtA_mm_t"],
        ).opts(**ex_plot_kw(i))
        for i in np.sort(example_data.phase.unique())
    ]
    examples_layout = hv.Layout(example_plots).opts(**layout_kw).cols(3)

    if save:
        _fname = save_dir.joinpath(f"{prefix}_example_curves.{fmt}")
        print(f"Writing to: {_fname.resolve().absolute()}")
        hv.save(examples_layout, _fname, fmt=fmt, dpi=dpi)

    ## Plot maximum area of a signaling spot

    # Convert to square microns
    df["max_area_um2"] = df["max_area_mm2"] * 1e6

    # Convert to length units
    df["max_sqrtarea_um"] = np.sqrt(df["max_area_um2"])
    df["max_sqrtarea_mm"] = np.sqrt(df["max_area_mm2"])

    # Truncate areas above a ceiling value
    df[marker_dim + "_trunc"] = np.minimum(df[marker_dim], area_ceiling)

    ## Plotting options

    # Get size of marker for each point (scales with truncated area)
    df["marker_size"] = marker_scale * df[marker_dim + "_trunc"] / area_ceiling

    # Make background versions of phase colors
    phase_bgcolors = lsig.viz.hexa2hex(phase_colors, bg_alpha).tolist()
    bare_kw["color"] = hv.Cycle(phase_bgcolors)

    # Set options for plotting
    marker_kw = dict(
        marker=".",
        c="w",
        edgecolor="k",
        linewidth=0.25,
    )

    ## Set vertices for a custom legend box
    legend_xmin = xlim[0] + x_range * (1 - legend_width) / 2
    legend_xmax = xlim[1] - x_range * (1 - legend_width) / 2
    legend_ymin = ylim[1] - y_range * 0.20
    legend_ymax = ylim[1] - y_range * 0.01
    legend_verts = np.array(
        [
            (legend_xmin, legend_ymin),
            (legend_xmin, legend_ymax),
            (legend_xmax, legend_ymax),
            (legend_xmax, legend_ymin),
        ]
    )

    ## Make plot
    phasediagram_bg = (
        hv.Scatter(
            data=df,
            kdims=["g_inv_days"],
            vdims=["rho_0", "phase"],
        )
        .groupby("phase")
        .opts(
            **bare_kw,
        )
        .overlay()
    )

    stim_pts = hv.Scatter(
        data=df,
        kdims=["g_inv_days"],
        vdims=["rho_0", "marker_size"],
    ).opts(s="marker_size", **marker_kw)

    legend_bg = hv.Polygons(
        legend_verts,
    ).opts(
        edgecolor="k",
        linewidth=1,
        facecolor=legend_bgcol,
    )

    legend_pt_verts = np.array(
        [
            np.linspace(legend_xmin, legend_xmax, 13)[1:-1],
            np.repeat(
                (1 - legend_pt_ypos) * legend_ymin + legend_pt_ypos * legend_ymax, 11
            ),
        ]
    ).T
    legend_pts = hv.Scatter(legend_pt_verts).opts(
        s=np.linspace(0, marker_scale, 11), **marker_kw
    )

    spot_size_plot = hv.Overlay(
        [
            phasediagram_bg,
            stim_pts,
            legend_bg,
            legend_pts,
        ]
    ).opts(
        **plot_kw,
    )

    if save:
        _fname = save_dir.joinpath(f"{prefix}_spot_size.{fmt}")
        print(f"Writing to: {_fname.resolve().absolute()}")
        hv.save(spot_size_plot, _fname, fmt=fmt, dpi=dpi)

    ## Make phase diagram with perturbations
    with open(pert_clr_json, "r") as f:
        ks, vs = json.load(f)

        # Get colors of perturbations from file
        ks = [tuple(_k) for _k in ks]
        color_dict = dict(zip(ks, vs))

        # Get densities
        rhos = np.sort([k[1] for k in ks])

    # Make dataframe with growth parameters for all perturbations
    pdf = pd.read_csv(mle_fpath, index_col=0)
    pdf = pdf.merge(pd.DataFrame(ks, columns=["treatment", "rho_0"]))

    # Assign colors
    pdf["color"] = [
        color_dict[(t, d)] for t, d in pdf.loc[:, ["treatment", "rho_0"]].values
    ]

    # Calculate difference between CI bounds and mean
    pdf["g_inv_days_90CIdiff_lo"] = pdf["g_inv_days"] - pdf["g_inv_days_90CI_lo"]
    pdf["g_inv_days_90CIdiff_hi"] = pdf["g_inv_days_90CI_hi"] - pdf["g_inv_days"]

    pert_kw = dict(
        marker="o",
        color="color",
        s=55,
        ec=lsig.viz.black,
    )
    err_kw = dict(
        # linewidth=3,
        capsize=4,
        color="k",
    )

    pert_pts = hv.Points(
        data=pdf,
        kdims=["g_inv_days", "rho_0"],
        vdims=["color"],
    ).opts(**pert_kw)
    pert_err = hv.ErrorBars(
        data=pdf,
        kdims=["g_inv_days"],
        vdims=["rho_0", "g_inv_days_90CIdiff_lo", "g_inv_days_90CIdiff_hi"],
        horizontal=True,
    ).opts(**err_kw)

    pert_extend = 0.1
    pert_overlay = (phasediagram_bare * pert_err * pert_pts).opts(
        xaxis=None,
        yaxis=None,
        xlim=(xlim[0], xlim[1] + pert_extend * (xlim[1] - xlim[0])),
        ylim=(ylim[0], ylim[1] + pert_extend * (ylim[1] - ylim[0])),
    )

    if save:
        _fname = save_dir.joinpath(f"{prefix}_perturbations.{fmt}")
        print(f"Writing to: {_fname.resolve().absolute()}")
        hv.save(pert_overlay, _fname, fmt=fmt, dpi=dpi)


if __name__ == "__main__":
    main(
        save=True,
    )
