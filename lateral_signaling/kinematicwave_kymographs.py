import os
import json

import numpy as np

import holoviews as hv
hv.extension("matplotlib")

import colorcet as cc

import lateral_signaling as lsig

# Paths to read data
data_dir  = os.path.abspath("../data/imaging/kinematic_wave")
lp_data_path = os.path.join(data_dir, "line_profile_data.json")

# Paths to write data
save_dir  = os.path.abspath("../plots")
save_prefix = os.path.join(save_dir, "kinematic_wave_kymograph_")

def main(
    save = True,
    fmt  = "png",
    dpi  = 300,
):

    # Load data from file
    with open(lp_data_path, "r") as f:
        lp_data_dict = json.load(f)

    # Extract image names
    im_names = lp_data_dict["im_names"]

    # Extract intensity profile data and normalize
    data = np.array([lp_data_dict[i] for i in im_names])
    data_norm = np.array([lsig.normalize(d, d.min(), d.max()) for d in data])

    # Get position along line profile
    nd = data[0].size
    lp_length_mm = lp_data_dict["lp_length"]
    position = np.linspace(0, lp_length_mm, nd)  #  mm

    # Get time in days
    t_hours = np.array(lp_data_dict["t_hours"])
    t_days = t_hours / 24
    nt = t_days.size//2
    t_hours = t_hours[:nt]
    t_days = t_days[:nt]

    # Get normalized data as 1D arrays
    bfp_norm = np.array(data_norm[:5]).T
    gfp_norm = np.array(data_norm[5:]).T

    # Get mean position of the signaling wave (mean of distribution)
    median_position = np.empty((nt,), dtype=position.dtype)
    for i, d in enumerate(gfp_norm.T):
        dist = d[::-1] / d.sum()
        median_position[i] = position[np.searchsorted(np.cumsum(dist), 0.5)]

    # Get wave velocities and mean velocity
    velocities = np.diff(median_position)
    vbar = np.abs(velocities.mean())

    ## Make kymographs of fluorescence over time

    # Plot kymographs as images
    bounds = (t_days.min() - 0.5, position.min(), t_days.max() + 0.5, position.max())
    gfpimage_opts = dict(
        colorbar=True,
#        cbar_ticks=[(0, "0"), (1, "1")], 
        cbar_ticks=0, 
        cbar_width=0.05,
        cbar_padding=-0.13,
    )
    bfpimage_opts = dict(
        colorbar=True,
    #    cbar_ticks=[(0, "0"), (1, "1")], 
        cbar_ticks=0, 
        cbar_width=0.1,
    )
    plot_opts = dict(
        xlabel="Day",
        xticks=tuple(t_days),
        yticks=(0, 1, 2, 3, 4),
        ylabel="Position (mm)",
        aspect=0.6,
        fontscale=1.2,
    )
    gfp_kymo = hv.Overlay([
        hv.Image(
            gfp_norm,
            bounds=bounds,
        ).opts(
            cmap=lsig.kgy,
    #        clabel="GFP (norm.)",
            clabel="",
            **gfpimage_opts,
        ),
        hv.Scatter(
            (t_days, median_position)
        ).opts(
            c="w",
            s=30,
        ), 
        hv.Curve(
            (t_days, median_position)
        ).opts(
            c="w",
            linewidth=1,
        ), 
        hv.Text(
            2.8, 4.25, r"$\bar{\mathit{v}} = " + f"{vbar:.2f}" \
                + r"$" + "\n" + r"$mm\, day^{-1}$" ,
            halign="left",
            fontsize=12,
        ).opts(
            c="w"
        )
    ]).opts(**plot_opts)

    bfp_kymo = hv.Image(
        bfp_norm,
        bounds=bounds,
    ).opts(
        cmap=cc.kbc,
    #    clabel="BFP (norm.)",
        clabel="",
        **bfpimage_opts,
        **plot_opts,
    )

    # Save
    if save:
        
        gfp_kymo_path = save_prefix + "gfpnorm"
        bfp_kymo_path = save_prefix + "bfpnorm"

        _path = gfp_kymo_path + "." + fmt
        print(f"Writing to: {_path}")
        hv.save(gfp_kymo, gfp_kymo_path, dpi=dpi, fmt=fmt)
        
        _path = bfp_kymo_path + "." + fmt
        print(f"Writing to: {_path}")
        hv.save(bfp_kymo, bfp_kymo_path, dpi=dpi, fmt=fmt)

main(
    save = True,
)

