from pathlib import Path
import lateral_signaling as lsig

import os
from glob import glob
import json

import numpy as np
import pandas as pd

import skimage.io as io
import skimage.measure as msr

from tqdm import tqdm

import holoviews as hv

hv.extension("matplotlib")

lsig.viz.default_rcParams()

image_dir = lsig.data_dir.joinpath("imaging", "signaling_gradient")
img_analysis_dir = lsig.analysis_dir.joinpath("signaling_gradient")

well_diam_path = lsig.data_dir.joinpath("imaging", "well_diameter_mm.json")
circle_data_path = lsig.analysis_dir.joinpath("roi_circle_data.csv")
lp_params_path = lsig.analysis_dir.joinpath("line_profile.json")

save_dir = img_analysis_dir


def main(
    image_dir: Path = image_dir,
    well_diam_path: Path = well_diam_path,
    lp_params_path: Path = lp_params_path,
    save_dir: Path = save_dir,
    save_data: bool = False,
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

    # Load images and convert to image collections
    load_B = lambda f: io.imread(f).astype(np.uint8)[:, :, 2]
    ims_B = io.ImageCollection(files_B_abs, load_func=load_B)

    load_G = lambda f: io.imread(f).astype(np.uint8)[:, :, 1]
    ims_G = io.ImageCollection(files_G_abs, load_func=load_G)

    # Get images as Numpy array
    ims = list(ims_B) + list(ims_G)

    # Load circular ROIs
    circle_verts_df = pd.read_csv(circle_data_path)
    circle_verts = circle_verts_df.iloc[:, 1:].values

    # Diameter of the well
    with open(well_diam_path, "r") as f:
        well_diameter_mm = json.load(f)["signaling_gradient"]

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

    # # Calculate corners given the endpoints and width
    # lp_corners = lsig.get_lp_corners(src, dst, lp_width)

    # Calculate length of line profile in mm
    lp_length_mm = np.linalg.norm(src - dst) / (2 * radius) * well_diameter_mm

    # Compute expression profile along line (line profile)
    line_profiles = []
    for i, im in enumerate(tqdm(ims)):

        # Normalize image
        _im = im.copy()
        _im = lsig.rescale_img(_im)

        # Get line profile in image-specific coordinates
        *_c, _r = circle_verts[i]
        _src = lsig.transform_point(src, center, radius, _c, _r)
        _dst = lsig.transform_point(dst, center, radius, _c, _r)
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

    # Store line profile data
    lp_data_dict = {
        "lp_length": lp_length_mm,
        "im_names": im_names,
        "t_hours": list(t_hours),
    }
    for imn, lp in zip(im_names, line_profiles):
        lp_data_dict[imn] = list(lp)

    if save_data:
        _path = save_dir.joinpath("line_profile_data.json")
        print("Writing data to:", _path.resolve().absolute())
        with _path.open("w") as f:
            json.dump(lp_data_dict, f, indent=4)


if __name__ == "__main__":
    main(
        # save_data=True,
    )
