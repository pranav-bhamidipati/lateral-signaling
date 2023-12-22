from copy import deepcopy
from functools import partial
from typing import Tuple

import numba

import lateral_signaling as lsig

from pathlib import Path
import json

import numpy as np
import pandas as pd

import skimage.io as io

import colorcet as cc

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib_scalebar.scalebar import ScaleBar

import holoviews as hv

hv.extension("matplotlib")

lsig.set_simulation_params()
lsig.set_growth_params()
lsig.set_steady_state_data()
lsig.viz.default_rcParams()


image_dir = lsig.data_dir.joinpath("imaging/signaling_gradient")
img_analysis_dir = lsig.analysis_dir.joinpath("signaling_gradient")
circle_data_csv = img_analysis_dir.joinpath("roi_circle_data.csv")
lp_params_json = img_analysis_dir.joinpath("line_profile.json")
lp_data_json = img_analysis_dir.joinpath("line_profile_data.json")


def windowed_mean_1d(arr, winsize):
    """Calculate a running mean over a 1d Numpy array `arr` with window size `winsize`."""
    left = -(winsize - 1) // 2
    right = winsize + left

    @numba.stencil(neighborhood=((left, right - 1),))
    def kernel_avg(a):
        cumul = 0
        for i in range(left, right):
            cumul += a[i]
        return cumul / (right - left)

    new_arr = kernel_avg(arr)

    return new_arr[-left : -right + 1]


def main(
    t_days: float = 2.7,
    mean_dens_t0: float = 2.0,
    nx: int = 201,
    figsize: Tuple = (3, 2.25),
    winsize: int = 201,
    save_dir: Path = lsig.plot_dir,
    save_prefix: str = "signaling_gradient",
    save: bool = False,
    fmt: str = "png",
    dpi: int = 300,
):

    files_B = sorted(image_dir.glob("*BFP*.tif*"))
    files_G = sorted(image_dir.glob("*GFP*.tif*"))

    # Get unique name for each image
    im_names = [p.stem for p in files_B + files_G]

    # Load images and convert to image collections
    load_func = lambda f, channel: io.imread(f).astype(np.uint8)[:, :, channel]
    ims_B = io.ImageCollection(
        [str(f) for f in files_B], load_func=partial(load_func, channel=2)
    )
    ims_G = io.ImageCollection(
        [str(f) for f in files_G], load_func=partial(load_func, channel=1)
    )
    ims = list(ims_B) + list(ims_G)

    # Load circular ROIs
    circle_verts_df = pd.read_csv(circle_data_csv)
    circle_verts = circle_verts_df.iloc[:, 1:].values

    # Parameters used for line profile
    with open(lp_params_json, "r") as f:
        lp_params = json.load(f)
        src = np.array([lp_params["x_src"], lp_params["y_src"]])
        dst = np.array([lp_params["x_dst"], lp_params["y_dst"]])
        lp_width = int(lp_params["width"])

        # Which image was used to draw line profile
        im_for_lp = int(lp_params["im_for_lp"])

    center = np.array([circle_verts[im_for_lp, :2]])
    radius = circle_verts[im_for_lp, 2]

    # Fluorescence data along the line
    with open(lp_data_json, "r") as f:
        lp_data = json.load(f)
        lp_length_mm = lp_data["lp_length"]
        # t_hours = lp_data["t_hours"]
        line_profiles = [np.asarray(lp_data[n]) for n in im_names]
        mean_BFP_first = lp_data["first_BFP_image_mean_fluor"]

    # Set scalebar parameters (modify from defaults)
    sbar_kw = deepcopy(lsig.viz.sbar_kwargs)
    sbar_kw.update(
        dict(
            units="mm",
            fixed_value=2.0,
            fixed_units="mm",
            font_properties=dict(weight=0, size=0),
        )
    )

    # Plot layouts of images
    rows, cols = 2, 5
    cbar_aspect = 10
    gs_kw = dict(width_ratios=[1] * (cols - 1) + [1.25])
    fig, axs = plt.subplots(rows, cols, figsize=(15, 5.5), gridspec_kw=gs_kw)

    for i, ax in enumerate(axs.flat):

        # Plot image
        cmap_ = (cc.cm["kbc"], lsig.viz.kgy)[i // cols]
        ax.imshow(
            ims[i],
            cmap=cmap_,
        )
        ax.axis("off")

        # Get line profile for this image
        *_c, _r = circle_verts[i]
        _src = lsig.transform_point(src, center, radius, _c, _r)
        _dst = lsig.transform_point(dst, center, radius, _c, _r)

        # Get width of each pixel in image
        _dx = lp_length_mm / np.linalg.norm(_src - _dst)

        # Plot scalebar
        sbar_kw["dx"] = _dx
        _scalebar = ScaleBar(**sbar_kw)
        ax.add_artist(_scalebar)

        # Plot line profile in first column
        if i % cols == 0:

            _lp_width = lp_width * _r / radius
            _lp_corners = lsig.get_lp_corners(_src, _dst, _lp_width)

            # Plot
            ax.arrow(
                *_dst,
                *(0.95 * (_src - _dst)),
                color="w",
                width=1,
                head_width=50,
            )
            pc = PatchCollection([Polygon(_lp_corners)])
            pc.set(
                edgecolor="y",
                linewidth=1.5,
                facecolor=(0, 0, 0, 0),
            )
            ax.add_collection(pc)

        if i % cols == cols - 1:
            #            _shrink = (0.9, 0.87)[i // cols]
            plt.colorbar(
                cm.ScalarMappable(cmap=cmap_),
                ax=ax,
                shrink=0.95,
                ticks=[],
                aspect=cbar_aspect,
            )

        ax.invert_yaxis()
        ax.invert_xaxis()

    plt.tight_layout()

    if save:
        _path = save_dir.joinpath(save_prefix + "_images").with_suffix("." + fmt)
        _path = _path.resolve().absolute()
        print("Writing to:", str(_path))
        plt.savefig(_path, format=fmt, dpi=dpi, facecolor="k", transparent=True)

    # Get positions along profile
    lp_positions = [np.linspace(0, lp_length_mm, lp.size) for lp in line_profiles]

    # Overlay intensity profiles at first time-point
    fig, ax = plt.subplots(figsize=figsize)

    ax.set(
        xlabel="Position (mm)",
        ylabel="Fluorescence",
        yticks=(0, 1),
    )

    clr_B = "b"
    lbl_B = "BFP"
    pos_B_unsmoothed = lp_positions[0]
    lp_B_unnorm_unsmoothed = line_profiles[0]
    pos_B_unsmoothed = pos_B_unsmoothed.max() - pos_B_unsmoothed

    pos_B = windowed_mean_1d(pos_B_unsmoothed, winsize=winsize)
    lp_B_unnorm = windowed_mean_1d(lp_B_unnorm_unsmoothed, winsize=winsize)
    lp_B = lsig.normalize(lp_B_unnorm, lp_B_unnorm.min(), lp_B_unnorm.max())
    plt.plot(pos_B, lp_B, color=clr_B, linewidth=2, label=lbl_B)

    # Estimate mean density at future time-point
    rho_x_t0 = mean_dens_t0 * np.linspace(0, 2, nx)
    t = t_days / lsig.t_to_units(1)
    mean_dens = lsig.logistic(t, 1.0, rho_x_t0, lsig.mle_params.rho_max_ratio).mean()

    # lbl_dens = "Density"
    # dens_B = mean_dens * lp_B / mean_BFP_first
    # plt.plot(pos_B, dens_B, color=clr_B, linewidth=2, label=lbl_dens)

    min_dens = mean_dens * lp_B_unnorm.min() / mean_BFP_first  # roughly 0.5
    max_dens = mean_dens * lp_B_unnorm.max() / mean_BFP_first  # roughly 6.2

    # mean_dens_yval = lsig.normalize(mean_dens, min_dens, max_dens)
    # plt.hlines(mean_dens_yval, *plt.xlim(), colors="k", linewidth=2)

    rho_OFF_yval = lsig.normalize(lsig.rho_crit_high, min_dens, max_dens)
    plt.hlines(rho_OFF_yval, *plt.xlim(), colors="k", linewidth=1)

    rho_ON_yval = lsig.normalize(lsig.rho_crit_low, min_dens, max_dens)
    plt.hlines(rho_ON_yval, *plt.xlim(), colors="k", linewidth=1)

    # scan_dens_yval = lsig.normalize(np.arange(1, 6), min_dens, max_dens)
    # plt.hlines(
    #     scan_dens_yval, *plt.xlim(), colors="gray", linewidth=1, linestyles="dotted"
    # )

    clr_G = "g"
    lbl_G = "GFP"
    pos_G_unsmoothed = lp_positions[5]
    lp_G_unnorm_unsmoothed = line_profiles[5]
    pos_G_unsmoothed = pos_G_unsmoothed.max() - pos_G_unsmoothed

    pos_G = windowed_mean_1d(pos_G_unsmoothed, winsize=winsize)
    lp_G_unnorm = windowed_mean_1d(lp_G_unnorm_unsmoothed, winsize=winsize)
    lp_G = lsig.normalize(lp_G_unnorm, lp_G_unnorm.min(), lp_G_unnorm.max())
    plt.plot(pos_G, lp_G, color=clr_G, linewidth=2, label=lbl_G)

    plt.tight_layout()

    plt.legend(loc="center left", bbox_to_anchor=(ax.get_xlim()[1], 0.5))

    plt.tight_layout()

    lp_B_unsmoothed = lsig.normalize(
        lp_B_unnorm_unsmoothed, lp_B_unnorm.min(), lp_B_unnorm.max()
    )
    plt.plot(
        pos_B_unsmoothed,
        lp_B_unsmoothed,
        color=clr_B,
        linewidth=0.5,
        alpha=0.5,
    )
    lp_G_unsmoothed = lsig.normalize(
        lp_G_unnorm_unsmoothed, lp_G_unnorm.min(), lp_G_unnorm.max()
    )
    plt.plot(
        pos_G_unsmoothed,
        lp_G_unsmoothed,
        color=clr_G,
        linewidth=0.5,
        alpha=0.5,
    )

    # rho_bar = 2.0
    # rho_pos = lp_B_unnorm / lp_B_unnorm.mean() * rho_bar
    # SS_mean_pos = lsig.get_steady_state_mean(rho_pos)
    # SS_std_pos = lsig.get_steady_state_std(rho_pos)
    # twinax = ax.twinx()
    # twinax.set(
    #     ylabel=r"$[GFP]_{SS}$",
    #     # ylim=(),
    # )
    # twinax.plot(pos_B, SS_mean_pos, color="r", linewidth=0.5)

    # if _debug:
    #     return SS_mean_pos, SS_std_pos

    # twinax.plot(pos_B, SS_mean_pos - SS_std_pos, color="purple", lw=0.5)
    # twinax.plot(pos_B, SS_mean_pos + SS_std_pos, color="yellow", lw=0.5)

    # twinax.fill_between(
    #     pos_B,
    #     SS_mean_pos - SS_std_pos,
    #     SS_mean_pos + SS_std_pos,
    #     # fc="gray",
    #     # ec="None",
    #     # alpha=0.2,
    # )

    if save:
        _path = save_dir.joinpath(save_prefix + "_line_profiles").with_suffix("." + fmt)
        _path = _path.resolve().absolute()
        print("Writing to:", str(_path))
        plt.savefig(_path, format=fmt, dpi=dpi)


if __name__ == "__main__":
    main(
        # save=True,
        # save_dir = lsig.temp_plot_dir,
    )
