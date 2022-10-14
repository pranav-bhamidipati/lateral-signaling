import os
from glob import glob
import json
from pathlib import Path
import h5py

import numpy as np
import pandas as pd
from tqdm import tqdm

import holoviews as hv

hv.extension("matplotlib")

import lateral_signaling as lsig

# Reading
data_dir = Path("../data/simulations")
sacred_dir = data_dir.joinpath("20211201_singlespotphase/sacred")
thresh_fpath = data_dir.joinpath("phase_threshold.json")
slices_fpath = data_dir.joinpath("phase_slices.json")


def main(
    pad=0.05,
    shrink=0.1,
    prefix="phase_diagram_3Dslice",
    save_dir=lsig.plot_dir,
    save=False,
    fmt="png",
    dpi=300,
):

    ## Read in and assemble data
    # Get threshold for v_init
    with open(slices_fpath, "r") as f:
        slices = json.load(f)
        slice_var = slices["var"]
        slice_val = slices["val"]
        slice_title = slices["title"]
        slice_x = slices["x"]
        slice_y = slices["y"]

    # Get threshold for v_init
    with open(thresh_fpath, "r") as f:
        threshs = json.load(f)
        v_init_thresh = float(threshs["v_init_thresh"])

    # Read in phase metric data
    run_dirs = [d for d in sacred_dir.glob("*") if d.joinpath("config.json").exists()]

    # Store each run's data in a DataFrame
    dfs = []
    for rd_idx, rd in enumerate(tqdm(run_dirs)):

        # Get some info from the run configuration
        with rd.joinpath("config.json").open("r") as c:
            config = json.load(c)

            # Initial density, carrying capacity
            rho_max = config["rho_max"]
            rho_0 = config["rho_0"]
            g_space = np.array(config["g_space"])
            g = g_space.tolist()

        if rho_0 < 1.0:
            continue

        # Get remaining info from run's data dump
        with h5py.File(rd.joinpath("results.hdf5"), "r") as f:

            # Phase metrics
            v_init = np.asarray(f["v_init_g"])
            n_act_fin = np.asarray(f["n_act_fin_g"])

            # Proliferation rates
            t = np.asarray(f["t"])

        # Assemble dataframe
        _df = pd.DataFrame(
            dict(
                v_init=v_init,
                n_act_fin=n_act_fin,
                g=g,
                rho_0=rho_0,
                rho_max=rho_max,
            )
        )
        dfs.append(_df)

    # Concatenate into one dataset
    df = pd.concat(dfs).reset_index(drop=True)
    df["g_inv_days"] = lsig.g_to_units(df["g"].values)

    # Assign phases and corresponding plot colors
    df["phase"] = (df.v_init > v_init_thresh).astype(int) * (
        1 + (df.n_act_fin > 0).astype(int)
    )
    df = df.sort_values("phase")

    # Extract data ranges
    # g_space = np.unique(df["g_inv_days"])
    g_range = g_space[-1] - g_space[0]
    rho_0_space = np.unique(df["rho_0"])
    rho_0_range = rho_0_space[-1] - rho_0_space[0]
    rho_max_space = np.unique(df["rho_max"])
    rho_max_range = rho_max_space[-1] - rho_max_space[0]

    # Get data for each slice
    slice_data = [
        df.loc[np.isclose(df[var], val), :] for var, val in zip(slice_var, slice_val)
    ]

    ## Plot 2D slices

    # Set axis limits
    axis_lims = {
        "g_inv_days": tuple(
            [
                g_space[0] - pad * g_range,
                g_space[-1] + pad * g_range,
            ]
        ),
        "rho_0": tuple(
            [
                rho_0_space[0] - pad * rho_0_range,
                rho_0_space[-1] + pad * rho_0_range,
            ]
        ),
        "rho_max": tuple(
            [
                rho_max_space[0] - pad * rho_max_range,
                rho_max_space[-1] + pad * rho_max_range,
            ]
        ),
    }
    axis_labels = {
        "g_inv_days": r"proliferation rate ($days^{-1}$)",
        "rho_0": r"init. density (x 100% confl.)",
        "rho_max": r"carrying capacity (x 100% confl.)",
    }
    axis_ticks = {
        "g_inv_days": (0.5, 1.0, 1.5),
        "rho_0": (0, 1, 2, 3, 4, 5, 6),
        "rho_max": (0, 1, 2, 3, 4, 5, 6),
    }

    # Colors for phase regions
    phase_colors = lsig.cols_blue[::-1]

    # Options for different plot types
    kw = dict(
        hooks=[lsig.remove_RT_spines],
        fontscale=1.0,
        show_legend=False,
        marker="s",
        edgecolor=None,
        s=60,
        # color=hv.Cycle(phase_colors),
        color=hv.dim("phase").categorize({i: c for i, c in enumerate(phase_colors)}),
    )

    # Get kwargs for each slice
    def update_kw(x, y):
        kw.update(
            dict(
                xlabel=axis_labels[x],
                xticks=axis_ticks[x],
                xlim=axis_lims[x],
                ylabel=axis_labels[y],
                yticks=axis_ticks[y],
                ylim=axis_lims[y],
            )
        )

    labels = [
        [
            {"name": "Attenuated", "c": "k", "xy": (3.0, 4.0)},
            {"name": "Unlimited", "c": "w", "xy": (2.5, 2.25)},
            {
                "name": "Limited",
                "c": "w",
                "xy": (4.5, 1.5),
                "line": dict(dst=(5.5, 2.25), lw=1.5),
            },
        ],
        [
            {"name": "Limited", "c": "k", "xy": (1.5, 5.5)},
            {"name": "Unlimited", "c": "w", "xy": (1.0, 3.0)},
        ],
    ]

    # Make 2D slices of phase space
    pd_slices = []
    for t, x, y, d, l in zip(slice_title, slice_x, slice_y, slice_data, labels):
        # Get kwargs
        update_kw(x, y)

        # Plot
        p = (
            hv.Scatter(
                data=d,
                kdims=[x],
                vdims=[y, "phase"],
            )
            .groupby("phase")
            .opts(title=t, **kw)
            .overlay()
        )

        # Phase label text
        t = hv.Overlay(
            [hv.Text(*d["xy"], d["name"], halign="right").opts(color=d["c"]) for d in l]
        )
        # Annotation line
        for d in l:
            if "line" in d.keys():
                lin = d["line"]

                src = np.array(d["xy"])
                dst = np.array(lin["dst"])
                diff = dst - src
                src = src + diff * shrink
                dst = dst - diff * shrink

                xs, ys = np.array([src, dst]).T

                t = t * hv.Curve(dict(x=xs, y=ys)).opts(
                    color=d["c"], linewidth=lin["lw"]
                )

        pd_slices.append(p * t)

    if save:
        for i, p in enumerate(pd_slices):
            fpath = save_dir.joinpath(f"{prefix}_{i}.{fmt}")
            print(f"Writing to: {fpath.resolve().absolute()}")
            hv.save(p, fpath, fmt=fmt, dpi=dpi)


if __name__ == "__main__":
    main(
        save=True,
    )
