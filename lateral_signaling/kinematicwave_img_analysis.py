import lateral_signaling as lsig

import os
from glob import glob
import json
from copy import deepcopy

import numpy as np
import pandas as pd

import skimage
import skimage.io as io
import skimage.filters as filt
import skimage.measure as msr

from tqdm import tqdm

import colorcet as cc

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib_scalebar.scalebar import ScaleBar

import holoviews as hv
hv.extension("matplotlib")

# Paths to read data
data_dir          = os.path.abspath("../data/imaging/kinematic_wave")
image_dir         = os.path.join(data_dir, "processed")
circle_data_path  = os.path.join(data_dir, "roi_circle.json")
lp_param_path     = os.path.join(data_dir, "line_profile.json")


# Paths to save data
save_dir          = os.path.abspath("../plots")
im_layout_path    = os.path.join(save_dir, "kinematic_wave_images__")
lp_plot_path      = os.path.join(save_dir, "kinematic_wave_line_profiles__")
lp_data_path      = os.path.join(data_dir, "__line_profile_data.json")

def main(
    save_data = False,
    save_figs = False,
    fmt       = "png",
    dpi       = 300,
):

    # Get image filenames
    files = glob(os.path.join(image_dir, "*.tif*"))
    files = [os.path.realpath(f) for f in np.sort(files)]
    n_ims = len(files)

    # Select blue and green fluorescence images (BFP and GFP)
    files_B = [f for f in files if "BFP" in f]
    files_G = [f for f in files if "GFP" in f]

    # Get unique name for each image
    im_names = []
    for f in files:
        end = os.path.split(f)[1]
        im_names.append(end[:end.index(".")])

    # Get time points as days/hours
    t_hours = [float(imn[-6:-3]) for imn in im_names]
    t_days  = [h / 24 for h in t_hours]

    # Load images
    load_B = lambda f: io.imread(f).astype(np.uint8)[:, :, 2]
    ims_B = io.ImageCollection(files_B, load_func=load_B).concatenate()

    load_G = lambda f: io.imread(f).astype(np.uint8)[:, :, 1]
    ims_G = io.ImageCollection(files_G, load_func=load_G).concatenate()

    # Get images as Numpy array
    ims = np.concatenate([ims_B, ims_G])

    # Save shape of each image
    imshape = ims.shape[1:]

    # Load circular ROI
    with open(circle_data_path, "r") as f: 
        circle_data = json.load(f)
        
        # Center of circle
        center = np.array([circle_data["x_center"], circle_data["y_center"]])
        
        # Radius of whole well
        radius = circle_data["radius"]
        
        # Diameter of the well
        well_diameter_mm = circle_data["well_diameter_mm"]


    # Load parameters for line profile
    with open(lp_param_path, "r") as f: 
        lp_data = json.load(f)
        src = np.array([lp_data["x_src"], lp_data["y_src"]])
        dst = np.array([lp_data["x_dst"], lp_data["y_dst"]])
        lp_width = int(lp_data["width"])

    # Calculate corners given the endpoints and width
    lp_corners = lsig.get_lp_corners(src, dst, lp_width)

    # Calculate length of line profile in mm
    lp_length_mm = np.linalg.norm(src - dst) / (2 * radius) * well_diameter_mm

    # Set scalebar parameters (modify from defaults)
    sbar_kw = deepcopy(lsig.sbar_kwargs)
    dx = lp_length_mm / np.linalg.norm(src - dst)   # width of each pixel
    sbar_kw.update(dict(
        dx=dx,
        units="mm",
        fixed_value=2.,
        fixed_units="mm",
        font_properties=dict(weight=0, size=0),
    ))

    # Plot layouts of images
    rows, cols = 2, 5
    cbar_aspect = 10
    gs_kw = dict(width_ratios = [1] * (cols - 1) + [1.25])
    fig, axs = plt.subplots(rows, cols, figsize=(15, 4.5), gridspec_kw=gs_kw)

    for i, ax in enumerate(axs.flat):
        
        # Plot image
        cmap_ = (cc.cm["kbc"], lsig.kgy)[i // cols]    
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
            
            #ax.plot(*np.array([src, dst]).T, color="w", linewidth=1)
            ax.arrow(*src, *(0.92 * (dst - src)), color="w", width=1, head_width=50,)
            
            pc = PatchCollection([Polygon(lp_corners)])
            pc.set(edgecolor="y", linewidth=1.0, facecolor=(0, 0, 0, 0), )
            ax.add_collection(pc)
        
        # Add colorbar to last column
        if i % cols == cols - 1:
            plt.colorbar(cm.ScalarMappable(cmap=cmap_), ax=ax, shrink=0.95, ticks=[], aspect=cbar_aspect)

    plt.tight_layout()

    # Save
    if save_figs:

        _path = im_layout_path + "." + fmt
        print("Writing to:", _path)
        plt.savefig(_path, format=fmt, dpi=dpi)

    # Compute expression profile along line (line profile)
    line_profiles = []
    for i, im in enumerate(tqdm(ims)):
        
        # Normalize image
        _im = im.copy()
        _im = lsig.rescale_img(_im)
        
        # Get profile
        prof = msr.profile_line(
            image=_im.T, 
            src=src, 
            dst=dst, 
            linewidth=lp_width,
            mode="constant", 
            cval=-200, 
        )
        line_profiles.append(prof)

    # Get positions along profile
    lp_positions = [np.linspace(0, lp_length_mm, lp.size) for lp in line_profiles]

    # Plot options
    curve_opts=dict(
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
    curves = [hv.Curve((pos, lp)).opts(title=t) for pos, lp, t in zip(lp_positions, line_profiles, titles)]
    for i, curve in enumerate(curves):
        curve.opts(ylabel=("BFP (AU)", "GFP (AU)")[i // 5])

    curves_layout = hv.Layout(curves).opts(
        hv.opts.Curve(**curve_opts),
    ).cols(5)
    curves_layout.opts(sublabel_format="")

    # Save
    if save_figs:
        _path = lp_plot_path + "." + fmt
        print("Writing to:", _path)
        hv.save(curves_layout, lp_plot_path, fmt=fmt, dpi=dpi)

    # Store line profile data
    lp_data_dict = {
        "lp_length": lp_length_mm, 
        "im_names": im_names,
        "t_hours": list(t_hours),
    }
    for imn, lp in zip(im_names, line_profiles):
        lp_data_dict[imn] = list(lp)

    # Save as JSON
    if save_data:
        with open(lp_data_path, "w") as f:
            json.dump(lp_data_dict, f, indent=4)


main(
    save_data = True,
    save_figs = True,
)

