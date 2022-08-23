import os
from glob import glob
import json
from pathlib import Path
import h5py

import numpy as np
import pandas as pd
from tqdm import tqdm

import colorcet as cc
import cmocean.cm as cmo

import holoviews as hv

hv.extension("matplotlib")

import matplotlib.pyplot as plt
import matplotlib as mpl

import lateral_signaling as lsig


# Reading simulated data
data_dir = os.path.abspath("../data/simulations/")
sacred_dir = os.path.join(data_dir, "20211209_phase_2D/sacred")
thresh_fpath = os.path.join(data_dir, "phase_threshold.json")
examples_json = os.path.join(data_dir, "phase_examples.json")
examples_dir = os.path.join(data_dir, "20211209_phase_examples/sacred")

# Reading growth parameter estimation data
mle_dir = os.path.abspath("../data/growth_curves_MLE")
mle_fpath = os.path.join(mle_dir, "growth_parameters_MLE.csv")
pert_clr_json = os.path.join(mle_dir, "perturbation_colors.json")

# Writing
save_dir = os.path.abspath("../plots/tmp")
fpath_pfx = os.path.join(save_dir, "phase_diagram_2D_")


def main(
    pad=0.05,
    area_ceiling=1,
    bg_alpha=0.8,
    marker_dim="max_sqrtarea_mm",
    marker_scale=130,
    legend_bgcol="#bebebe",
    legend_width=0.8,
    legend_pt_ypos=0.5,
    save=False,
    prefix=fpath_pfx,
    suffix="_",
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

    # Get threshold for v_init
    with open(thresh_fpath, "r") as f:
        threshs = json.load(f)
        v_init_thresh = float(threshs["v_init_thresh"])

    # Read in phase metric data
    run_dirs = glob(os.path.join(sacred_dir, "[0-9]*"))

    # Store each run's data and in a DataFrame
    dfs = []
    for rd_idx, rd in enumerate(tqdm(run_dirs)):

        _config_file = os.path.join(rd, "config.json")
        _results_file = os.path.join(rd, "results.hdf5")

        if (not os.path.exists(_config_file)) or (not os.path.exists(_results_file)):
            continue

        # Get some info from the run configuration
        with open(_config_file, "r") as c:
            config = json.load(c)

            # Intrinsic growth rate, initial density, carrying capacity
            rho_0 = config["rho_0"]
            g = config["g"]
            rho_max = config["rho_max"]
            g_space = config["g_space"]

        if not (1.0 <= rho_0 <= rho_max):
            continue

        # Get remaining info from run's data dump
        with h5py.File(_results_file, "r") as f:

            t = np.asarray(f["t"])

            # Number of activated cells and density vs. time
            ncells_t_g = np.asarray(f["S_t_g_actnum"])
            rho_t_g = np.asarray(f["rho_t_g"])

            # Phase metrics
            v_init = np.asarray(f["v_init_g"])
            n_act_fin = np.asarray(f["n_act_fin_g"])

        # Calculate maximum area for each param set in batch
        maxArea_g = lsig.ncells_to_area(ncells_t_g, rho_t_g).max(axis=1)

        # Assemble dataframe
        _df = pd.DataFrame(
            dict(
                g=g_space,
                rho_0=rho_0,
                rho_max=rho_max,
                v_init=v_init,
                n_act_fin=n_act_fin,
                max_area_mm2=maxArea_g,
            )
        )
        dfs.append(_df)

    # Concatenate into one dataset
    df = pd.concat(dfs).reset_index(drop=True)
    df["g_inv_days"] = lsig.g_to_units(df["g"].values)

    # Assign phases and sort by phase
    df["phase"] = (df.v_init > v_init_thresh).astype(int) * (
        1 + (df.n_act_fin > 0).astype(int)
    )
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
            rho_0_space[-1] + pad * rho_0_range,
        ]
    )
    y_range = ylim[1] - ylim[0]

    # Colors for phase regions
    phase_colors = lsig.cols_blue[::-1]

    # Options for different plot types
    plot_kw = dict(
        xlim=xlim,
        ylim=ylim,
        xlabel=r"proliferation rate ($days^{-1}$)",
        xticks=(0.5, 1.0, 1.5),
        ylabel=r"init. density (x 100% confl.)",
        yticks=(0, 1, 2, 3, 4, 5, 6),
        hooks=[lsig.remove_RT_spines],
        fontscale=1.0,
        show_legend=False,
        #        aspect=1.,
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

    data = df.pivot(columns="g_inv_days", index="rho_0", values="phase")
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

    fig1 = plt.figure(1, figsize=(2, 2))
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

    _fname = str(Path(save_dir).joinpath(prefix + "highlighted" + "." + fmt))
    print(_fname)
    plt.savefig(_fname, dpi=dpi)

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

        fpath = prefix + "bare" + suffix
        _fpath = fpath + "." + fmt
        print(f"Writing to: {_fpath}")
        hv.save(phasediagram_bare, fpath, fmt=fmt, dpi=dpi)

        fpath = prefix + "examples" + suffix
        _fpath = fpath + "." + fmt
        print(f"Writing to: {_fpath}")
        hv.save(phasediagram_ex, fpath, fmt=fmt, dpi=dpi)

        fpath = prefix + "labeled" + suffix
        _fpath = fpath + "." + fmt
        print(f"Writing to: {_fpath}")
        hv.save(phasediagram_labeled, fpath, fmt=fmt, dpi=dpi)

    # Isolate aggregate data for examples
    ex_df = df.loc[df["example"].str.len() > 0]

    # Read data for phase examples
    ex_dirs = glob(os.path.join(examples_dir, "[0-9]*"))
    ex_dfs = []
    for rd_idx, rd in enumerate(ex_dirs):
        _config_file = os.path.join(rd, "config.json")
        _results_file = os.path.join(rd, "results.hdf5")

        # Get some info from the run configuration
        with open(_config_file, "r") as c:
            config = json.load(c)

            # Expression threshold
            k = config["k"]

            # Intrinsic growth rate, initial density, carrying capacity
            g = config["g"]
            rho_0 = config["rho_0"]

        # Get remaining info from run's data dump
        with h5py.File(_results_file, "r") as f:

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

        fpath = prefix + "example_curves" + suffix
        _fpath = fpath + "." + fmt
        print(f"Writing to: {_fpath}")
        hv.save(examples_layout, fpath, fmt=fmt, dpi=dpi)

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
    phase_bgcolors = lsig.hexa2hex(phase_colors, bg_alpha).tolist()
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

    legend_bg = hv.Polygons(legend_verts,).opts(
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

        fpath = prefix + "spot_size" + suffix
        _fpath = fpath + "." + fmt
        print(f"Writing to: {_fpath}")
        hv.save(spot_size_plot, fpath, fmt=fmt, dpi=dpi)

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

    #    # Add rows for different density treatments
    #    pdf["rho_0"] = rhos[0]
    #    untreated_row = pdf.loc[pdf.treatment=="untreated", :]
    #    untreated_df = pd.concat([untreated_row] * len(rhos))
    #    untreated_df["rho_0"] = rhos
    #    pdf = pd.concat([pdf, untreated_df])
    #    pdf = pdf.drop_duplicates().reset_index(drop=True)

    #    # Select data for plotting
    #    pert_columns = [
    #        "treatment",
    #        r"$\rho_0$",
    #        "g_inv_days",
    #        "g_inv_days_90CIdiff_lo",
    #        "g_inv_days_90CIdiff_hi",
    #    ]
    #    pdf = pdf.loc[:, pert_columns]

    #    # Remove axis labels and phase examples
    #    phasediagram_bare = phasediagram_bare.options(
    #    )

    # Set plotting options
    #    axis_off_args = (
    #        hv.opts.Scatter(
    #            edgecolor=None,
    #        ),
    #        hv.opts.Overlay(
    #            xaxis=None,
    #            yaxis=None,
    #        ),
    #    )
    pert_kw = dict(
        marker="^",
        color="color",
        s=250,
        ec=lsig.black,
    )
    err_kw = dict(
        #        linewidth=3,
        capsize=4,
        color="k",
    )

    # Plot growth and density perturbations
    #    g_pts = hv.Points(
    #        data=pdf.loc[pdf["rho_0"]==rhos[0], :],
    #        kdims=["g_inv_days", "rho_0"],
    #        vdims=["color"],
    #    ).opts(**pert_kw)
    #    g_err = hv.ErrorBars(
    #        data=pdf.loc[pdf["rho_0"]==rhos[0], :],
    #        kdims=["g_inv_days"],
    #        vdims=["rho_0", "g_inv_days_90CIdiff_lo", "g_inv_days_90CIdiff_hi"],
    #        horizontal=True,
    #    ).opts(**err_kw)
    #    rho0_pts = hv.Points(
    #        data=pdf.loc[pdf["treatment"]=="untreated", :],
    #        kdims=["g_inv_days", "rho_0"],
    #        vdims=["color"],
    #    ).opts(**pert_kw)

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

    pert_overlay = (phasediagram_bare * pert_pts * pert_err).opts(
        xaxis=None, yaxis=None
    )
    #    drug_overlay      = (phasediagram_bare * g_pts * g_err).opts(xaxis=None, yaxis=None)
    #    dens_overlay      = (phasediagram_bare * rho0_pts).opts(xaxis=None, yaxis=None)
    #    drug_dens_overlay = (phasediagram_bare * g_pts * rho0_pts).opts(xaxis=None, yaxis=None)

    #    drug_overlay      = (phasediagram_bare * g_pts * g_err).opts(*axis_off_args)
    #    dens_overlay      = (phasediagram_bare * rho0_pts).opts(*axis_off_args)
    #    drug_dens_overlay = (phasediagram_bare * g_pts * rho0_pts).opts(*axis_off_args)

    if save:

        #        fpath = prefix + "growth_perturbations" + suffix
        #        _fpath = fpath + "." + fmt
        #        print(f"Writing to: {_fpath}")
        #        hv.save(drug_overlay, fpath, fmt=fmt, dpi=dpi)
        #
        #        fpath = prefix + "initdensity_perturbations" + suffix
        #        _fpath = fpath + "." + fmt
        #        print(f"Writing to: {_fpath}")
        #        hv.save(dens_overlay, fpath, fmt=fmt, dpi=dpi)
        #
        #        fpath = prefix + "both_perturbations" + suffix
        #        _fpath = fpath + "." + fmt
        #        print(f"Writing to: {_fpath}")
        #        hv.save(drug_dens_overlay, fpath, fmt=fmt, dpi=dpi)

        fpath = prefix + "perturbations" + suffix
        _fpath = fpath + "." + fmt
        print(f"Writing to: {_fpath}")
        hv.save(pert_overlay, fpath, fmt=fmt, dpi=dpi)


if __name__ == "__main__":

    main(
        save=False,
        suffix="_",
    )
