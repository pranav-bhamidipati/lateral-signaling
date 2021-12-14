import os
import json

import numpy as np
from scipy.interpolate import interp1d

import holoviews as hv
hv.extension("matplotlib")

import colorcet as cc

import lateral_signaling as lsig

# Options for saving
save_figs = True
fmt       = "png"
dpi       = 300

data_dir  = os.path.abspath("../data/imaging/signaling_gradient")
save_dir  = os.path.abspath("../plots")

# Set paths to data
lp_data_path  = os.path.join(data_dir, "line_profile_data.json")
gfp_kymo_path = os.path.join(save_dir, "signaling_gradient_kymograph_gfpnorm." + fmt)
bfp_kymo_path = os.path.join(save_dir, "signaling_gradient_kymograph_bfpnorm." + fmt)

# Load data from file
with open(lp_data_path, "r") as f:
    lp_data_dict = json.load(f)

# Extract image names
im_names = lp_data_dict["im_names"]

# Extract intensity profile data and normalize
data = [np.array(lp_data_dict[i])[::-1] for i in im_names]

# Get time in days
t_hours = np.array(lp_data_dict["t_hours"])
t_days = t_hours / 24
nt = t_days.size//2
t_hours = t_hours[:nt]
t_days = t_days[:nt]

# Get smallest number of samples along intensity profile
#   All other samples will be downsampled to this value
nd = min([d.size for d in data])

# Specify which positions to sample along profile
lp_length_mm = lp_data_dict["lp_length"]
position = np.linspace(0, lp_length_mm, nd)  #  mm

# Downsample data to the same number of points
data_samp = np.zeros((len(data), nd), dtype=float)
for i, _d in enumerate(data):
    
    # Get positions currently sampled by data
    _pos = np.linspace(0, lp_length_mm, len(_d))
    
    # Construct a nearest-neighbor interpolation
    nnrep = interp1d(_pos, _d, kind="nearest")

    # Sample interpolation at desired points
    data_samp[i] = nnrep(position)

# Split up BFP and GFP data
bfp_data = data_samp[:5].T
gfp_data = data_samp[5:].T

# Normalize
bfp_norm = np.array([lsig.normalize(d, min(d), max(d)) for d in bfp_data.T]).T
gfp_norm = np.array([lsig.normalize(d, min(d), max(d)) for d in gfp_data.T]).T
 
# Get mean position of the signaling wave (mean of distribution)
median_position = np.empty((nt,), dtype=position.dtype)
for i, d in enumerate(gfp_norm.T):
    dist = d[::-1] / d.sum()
    median_position[i] = position[np.searchsorted(np.cumsum(dist), 0.5)]



# Get wave velocities and mean velocity
velocities = np.diff(median_position)
vbar = np.abs(velocities.mean())

## Make kymographs of fluorescence along spatial axis

# Plot kymographs as images
bounds = (t_days.min() - 0.5, position.min(), t_days.max() + 0.5, position.max())
image_opts = dict(
    colorbar=False,
    cbar_ticks=[(0, "0"), (1, "1")], 
    cbar_width=0.1,
)
plot_opts = dict(
    xlabel="Day",
    xticks=(2.7, 3.7, 4.7, 5.7, 6.7),
    yticks=(0, 1, 2, 3, 4, 5),
    ylabel="Position (mm)",
    aspect=0.6,
    fontscale=1.2,
)
gfp_kymo = (
    hv.Image(
        gfp_norm,
        bounds=bounds,
    ).opts(
        cmap=lsig.kgy,
        clabel="GFP (norm.)",
        **image_opts,
    ) * hv.Scatter(
        (t_days, median_position)
    ).opts(
        c="w",
        s=30,
    ) * hv.Curve(
        (t_days, median_position)
    ).opts(
        c="w",
        linewidth=1,
    ) * hv.Text(
        4.5, 4.25, r"$\bar{\mathit{v}} = " + f"{vbar:.2f}" \
            + r"$" + "\n" + r"$mm\, day^{-1}$" ,
        halign="left",
        fontsize=12,
    ).opts(
        c="w"
    )
).opts(**plot_opts)

bfp_kymo = hv.Image(
    bfp_norm,
    bounds=bounds,
).opts(
    cmap=cc.kbc,
    clabel="BFP (norm.)",
    **image_opts,
    **plot_opts,
)

# Save
if save_figs:
    
    hv.save(gfp_kymo, gfp_kymo_path, dpi=dpi, fmt=fmt)
    print(f"Figure saved to: {gfp_kymo_path}")
    
    hv.save(bfp_kymo, bfp_kymo_path, dpi=dpi, fmt=fmt)
    print(f"Figure saved to: {bfp_kymo_path}")
