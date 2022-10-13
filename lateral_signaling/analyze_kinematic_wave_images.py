import json

import numpy as np
from tqdm import tqdm
import skimage.io as io
import skimage.measure as msr

import lateral_signaling as lsig

image_dir = lsig.data_dir.joinpath("imaging/kinematic_wave")
img_analysis_dir = lsig.analysis_dir.joinpath("kinematic_wave")

circle_data_path = img_analysis_dir.joinpath("roi_circle.json")
lp_param_path = img_analysis_dir.joinpath("line_profile.json")

lp_data_path = img_analysis_dir.joinpath("line_profile_data.json")


def main(
    image_dir=image_dir,
    circle_data_path=circle_data_path,
    lp_param_path=lp_param_path,
    save=False,
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

    # Calculate length of line profile in mm
    lp_length_mm = np.linalg.norm(src - dst) / (2 * radius) * well_diameter_mm

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

    # Save as JSON
    lp_data_dict = {
        "lp_length": lp_length_mm,
        "im_names": im_names,
        "t_hours": list(t_hours),
    }
    for imn, lp in zip(im_names, line_profiles):
        lp_data_dict[imn] = list(lp)

    if save:
        with open(lp_data_path, "w") as f:
            json.dump(lp_data_dict, f, indent=4)


if __name__ == "__main__":
    main(
        # save=True,
    )
