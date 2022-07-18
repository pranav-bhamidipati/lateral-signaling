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

lsig.default_rcParams()

# Paths to read
data_dir           = os.path.abspath("../data/imaging/signaling_gradient")
image_dir          = os.path.join(data_dir, "processed")
circle_data_path   = os.path.join(data_dir, "roi_circle_data.csv")
circle_params_path = os.path.join(data_dir, "roi_circle_params.json")
lp_params_path     = os.path.join(data_dir, "line_profile.json")

# Paths to write data
save_dir       = os.path.abspath("../plots/tmp")
im_layout_path = os.path.join(save_dir, "signaling_gradient_images_")
lp_plot_path   = os.path.join(save_dir, "signaling_gradient_line_profiles_")
lp_data_path   = os.path.join(data_dir, "__line_profile_data.json")

def main(
    figsize=(3, 3),
    save_data=False,
    save_figs=False,
    fmt="png",
    dpi=300,
):

    # Get image filenames
    files = glob(os.path.join(image_dir, "*.tif*"))
    files = [os.path.realpath(f) for f in files]
    n_ims = len(files)

    # Select blue and green fluorescence images (BFP and GFP)
    files_B = [f for f in files if "BFP" in f]
    files_G = [f for f in files if "GFP" in f]

    # Get unique name for each image
    im_names = []
    for f in (*files_B, *files_G):
        end = os.path.split(f)[1]
        im_names.append(end[:end.index(".")])
        
    # Get time points as days/hours
    t_hours = [float(imn[-6:-3]) for imn in im_names]
    t_days  = [h / 24 for h in t_hours]

    # Load images and convert to image collections
    load_B = lambda f: io.imread(f).astype(np.uint8)[:, :, 2]
    ims_B = io.ImageCollection(files_B, load_func=load_B)

    load_G = lambda f: io.imread(f).astype(np.uint8)[:, :, 1]
    ims_G = io.ImageCollection(files_G, load_func=load_G)

    # Get images as Numpy array
    ims = list(ims_B) + list(ims_G)

    # Save shape of each image
    imshapes = (ims_B[0].shape, ims_G[0].shape)

    # Load circular ROIs
    circle_verts_df = pd.read_csv(circle_data_path)
    circle_verts = circle_verts_df.iloc[:, 1:].values

    # Diameter of the well
    with open(circle_params_path, "r") as f: 
        circle_params = json.load(f)
        well_diameter_mm = circle_params["well_diameter_mm"]

    # Load parameters for line profile
    with open(lp_params_path, "r") as f: 
        lp_data = json.load(f)
        src = np.array([lp_data["x_src"], lp_data["y_src"]])
        dst = np.array([lp_data["x_dst"], lp_data["y_dst"]])
        lp_width = int(lp_data["width"])
        
        # Which image was used to draw line profile
        im_for_lp = int(lp_data["im_for_lp"])

        # Get center and radius of the well used to draw the line profile
        center = np.array([circle_verts[im_for_lp, :2]])
        radius = circle_verts[im_for_lp, 2]

    # Calculate corners given the endpoints and width
    lp_corners = lsig.get_lp_corners(src, dst, lp_width)

    # Calculate length of line profile in mm
    lp_length_mm = np.linalg.norm(src - dst) / (2 * radius) * well_diameter_mm

    # Set scalebar parameters (modify from defaults)
    sbar_kw = deepcopy(lsig.sbar_kwargs)
    sbar_kw.update(dict(
        units="mm",
        fixed_value=2.,
        fixed_units="mm",
        font_properties=dict(weight=0, size=0),
    ))

    # Plot layouts of images
    rows, cols = 2, 5
    cbar_aspect = 10
    gs_kw = dict(width_ratios = [1] * (cols - 1) + [1.25])
    fig, axs = plt.subplots(rows, cols, figsize=(15, 5.5), gridspec_kw=gs_kw)

    for i, ax in enumerate(axs.flat):
        
        # Plot image
        cmap_ = (cc.cm["kbc"], lsig.kgy)[i // cols]
        ax.imshow(
            ims[i],
            cmap=cmap_,
        )
        ax.axis("off")
        
        # Get line profile for this image
        *_c, _r = circle_verts[i]
        _src  = lsig.transform_point(src, center, radius, _c, _r)
        _dst  = lsig.transform_point(dst, center, radius, _c, _r)
        
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
            ax.arrow(*_dst, *(0.95 * (_src - _dst)), color="w", width=1, head_width=50,)
            pc = PatchCollection([Polygon(_lp_corners)])
            pc.set(edgecolor="y", linewidth=1.5, facecolor=(0, 0, 0, 0), )
            ax.add_collection(pc)

        if i % cols == cols - 1:
#            _shrink = (0.9, 0.87)[i // cols]
            plt.colorbar(cm.ScalarMappable(cmap=cmap_), ax=ax, shrink=0.95, ticks=[], aspect=cbar_aspect)
        
        # Re-orient
        ax.invert_yaxis()
        ax.invert_xaxis()
        
    plt.tight_layout()

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
        
        # Get line profile in image-specific coordinates
        *_c, _r = circle_verts[i]
        _src  = lsig.transform_point(src, center, radius, _c, _r)
        _dst  = lsig.transform_point(dst, center, radius, _c, _r)
        _lp_width = int(lp_width * _r / radius)
        
        # Measure fluorescence along profile
        prof = msr.profile_line(
            image=_im.T, 
            src=_src, 
            dst=_dst, 
            linewidth=_lp_width,
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

    if save_figs:
        _path = lp_plot_path  + "." + fmt
        print("Writing to:", _path)
        hv.save(curves_layout, lp_plot_path, fmt=fmt, dpi=dpi)

    # Overlay intensity profiles at first time-point
#    overlay = hv.Overlay(
#        [curves[0], curves[5]]
#    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.set(
        xlabel="Position (mm)",
        ylabel="Norm. Fluorescence",
        yticks=(0, 1),
    )
   
    idx = (0, 5)
    colors = ("b", "g")
    labels = ("BFP", "GFP")
    for i, clr, lbl in zip(idx, colors, labels):
        pos = lp_positions[i]
        pos = pos.max() - pos   # flip direction
        lp  = line_profiles[i]
        lp  = (lp - lp.min()) / (lp.max() - lp.min())
        plt.plot(pos, lp, color=clr, linewidth=0.5, label=lbl)
    plt.legend(loc="center left", bbox_to_anchor=(0.0, 0.35))

#    twinax=ax.twinx() 
#    twinax.set(
#        ylabel="norm. GFP",
#    )

    plt.tight_layout()

    if save_figs:
        
        overlay_path = lp_plot_path + "overlay"
        _path = overlay_path + "." + fmt
        print("Writing to:", _path)
#        hv.save(overlay, overlay_path, fmt=fmt, dpi=dpi)
        plt.savefig(_path, dpi=dpi, format=fmt)

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
    save_figs=True, 
    save_data=False,
)

