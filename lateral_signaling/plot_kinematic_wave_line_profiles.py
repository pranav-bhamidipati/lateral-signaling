import json
from copy import deepcopy

import numpy as np
import skimage.io as io

import colorcet as cc

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib_scalebar.scalebar import ScaleBar

import holoviews as hv

import lateral_signaling as lsig

hv.extension("matplotlib")


image_dir = lsig.data_dir.joinpath("imaging/kinematic_wave")
img_analysis_dir = lsig.analysis_dir.joinpath("kinematic_wave")

lp_param_json = img_analysis_dir.joinpath("line_profile.json")
lp_data_json = img_analysis_dir.joinpath("line_profile_data.json")


def main(
    save_dir=lsig.plot_dir,
    save=False,
    fmt="png",
    dpi=300,
):

    # Get images and filenames
    files_B = sorted(image_dir.glob("*BFP*.tif*"))
    files_G = sorted(image_dir.glob("*GFP*.tif*"))

    im_names_B = [f.stem for f in files_B]
    im_names_G = [f.stem for f in files_G]
    im_names = im_names_B + im_names_G

    files_B_abs = [str(f.resolve().absolute()) for f in files_B]
    files_G_abs = [str(f.resolve().absolute()) for f in files_G]

    # Get time points as days/hours
    t_hours = [float(imn[-6:-3]) for imn in im_names]
    t_days = [h / 24 for h in t_hours]

    # Load images
    load_B = lambda f: io.imread(f).astype(np.uint8)[:, :, 2]
    ims_B = io.ImageCollection(files_B_abs, load_func=load_B).concatenate()

    load_G = lambda f: io.imread(f).astype(np.uint8)[:, :, 1]
    ims_G = io.ImageCollection(files_G_abs, load_func=load_G).concatenate()

    # Get images as Numpy array
    ims = np.concatenate([ims_B, ims_G])

    # Load line profile data
    with lp_param_json.open("r") as f:
        j = json.load(f)
        src = np.array([j["x_src"], j["y_src"]])
        dst = np.array([j["x_dst"], j["y_dst"]])
        lp_width = int(j["width"])

    with lp_data_json.open("r") as f:
        j = json.load(f)
        lp_length_mm = j["lp_length"]
        im_names = j["im_names"]
        line_profiles = [np.array(j[imn]) for imn in im_names]

    # Calculate corners given the endpoints and width
    lp_corners = lsig.get_lp_corners(src, dst, lp_width)

    # Set scalebar parameters (modify from defaults)
    sbar_kw = deepcopy(lsig.viz.sbar_kwargs)
    dx = lp_length_mm / np.linalg.norm(src - dst)  # width of each pixel
    sbar_kw.update(
        dict(
            dx=dx,
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
    fig, axs = plt.subplots(rows, cols, figsize=(15, 4.5), gridspec_kw=gs_kw)

    for i, ax in enumerate(axs.flat):

        # Plot image
        cmap_ = (cc.cm["kbc"], lsig.viz.kgy)[i // cols]
        ax.imshow(
            ims[i],
            cmap=cmap_,
        )
        ax.axis("off")

        # Plot scalebar
        _scalebar = ScaleBar(**sbar_kw)
        ax.add_artist(_scalebar)

        # Plot line-profile in first column
        if i % cols == 0:

            # ax.plot(*np.array([src, dst]).T, color="w", linewidth=1)
            ax.arrow(
                *src,
                *(0.92 * (dst - src)),
                color="w",
                width=1,
                head_width=50,
            )

            pc = PatchCollection([Polygon(lp_corners)])
            pc.set(
                edgecolor="y",
                linewidth=1.0,
                facecolor=(0, 0, 0, 0),
            )
            ax.add_collection(pc)

        # Add colorbar to last column
        if i % cols == cols - 1:
            plt.colorbar(
                cm.ScalarMappable(cmap=cmap_),
                ax=ax,
                shrink=0.95,
                ticks=[],
                aspect=cbar_aspect,
            )

    plt.tight_layout()

    if save:
        _fpath = save_dir.joinpath(f"kinematic_wave_images.{fmt}")
        print("Writing to:", _fpath.resolve().absolute())
        plt.savefig(_fpath, format=fmt, facecolor="k", dpi=dpi)

    # Get positions along profile
    lp_positions = [np.linspace(0, lp_length_mm, lp.size) for lp in line_profiles]

    # Plot options
    curve_opts = dict(
        padding=0.05,
        xlabel="Position (mm)",
        c="k",
        linewidth=1,
        fontscale=1.6,
    )

    # Titles for line profiles
    titles = []
    for n in im_names:
        *_, p, hr = n.split("_")
        t = " ".join([p + ",", str(int(hr[:3])), "hr"])
        titles.append(t)

    # Plot intensity values along line profiles
    curves = [
        hv.Curve((pos, lp)).opts(title=t)
        for pos, lp, t in zip(lp_positions, line_profiles, titles)
    ]
    for i, curve in enumerate(curves):
        curve.opts(ylabel=("BFP (AU)", "GFP (AU)")[i // 5])

    curves_layout = (
        hv.Layout(curves)
        .opts(
            hv.opts.Curve(**curve_opts),
        )
        .cols(5)
    )
    curves_layout.opts(sublabel_format="")

    if save:
        _fpath = save_dir.joinpath(f"kinematic_wave_line_profiles.{fmt}")
        print("Writing to:", _fpath)
        hv.save(curves_layout, _fpath, fmt=fmt, dpi=dpi)


if __name__ == "__main__":
    main(
        save_dir=lsig.temp_plot_dir,
        # save=True,
    )
