import json

import numpy as np
from scipy.interpolate import interp1d

import holoviews as hv

hv.extension("matplotlib")

import colorcet as cc

import lateral_signaling as lsig


# Paths to read data
lp_data_json = lsig.analysis_dir.joinpath("signaling_gradient/line_profile_data.json")


def main(
    prefix="signaling_gradient_kymograph",
    save_dir=lsig.plot_dir,
    save=True,
    fmt="png",
    dpi=300,
):

    # Load data from file
    lp_data_dict = json.load(lp_data_json.open("r"))

    # Extract image names
    im_names = lp_data_dict["im_names"]

    # Extract intensity profile data and normalize
    data = [np.array(lp_data_dict[i])[::-1] for i in im_names]

    # Get time in days
    t_hours = np.array(lp_data_dict["t_hours"])
    t_days = t_hours / 24
    nt = t_days.size // 2
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

    # Get median position of the signaling wave (median of distribution)
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
    gfpimage_opts = dict(
        colorbar=True,
        #    cbar_ticks=[(0, "0"), (1, "1")],
        cbar_ticks=0,
        cbar_width=0.07,
        cbar_padding=-0.11,
    )
    bfpimage_opts = dict(
        colorbar=True,
        #    cbar_ticks=[(0, "0"), (1, "1")],
        cbar_ticks=0,
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
    gfp_kymo = hv.Overlay(
        [
            hv.Image(gfp_norm, bounds=bounds,).opts(
                cmap=lsig.viz.kgy,
                #        clabel="GFP (norm.)",
                clabel="",
                **gfpimage_opts,
            ),
            hv.Scatter((t_days, median_position)).opts(
                c="w",
                s=30,
            ),
            hv.Curve((t_days, median_position)).opts(
                c="w",
                linewidth=1,
            ),
            hv.Text(
                4.5,
                4.75,
                r"$\bar{\mathit{v}} = "
                + f"{vbar:.2f}"
                + r"$"
                + "\n"
                + r"$mm\, day^{-1}$",
                halign="left",
                fontsize=12,
            ).opts(c="w"),
        ]
    ).opts(**plot_opts)

    bfp_kymo = hv.Image(bfp_norm, bounds=bounds,).opts(
        cmap=cc.kbc,
        #    clabel="BFP (norm.)",
        clabel="",
        **bfpimage_opts,
        **plot_opts,
    )

    if save:

        _fname = save_dir.joinpath(f"{prefix}_gfpnorm.{fmt}")
        print(f"Writing to: {_fname.resolve().absolute()}")
        hv.save(gfp_kymo, _fname, dpi=dpi, fmt=fmt)

        _fname = save_dir.joinpath(f"{prefix}_bfpnorm.{fmt}")
        print(f"Writing to: {_fname.resolve().absolute()}")
        hv.save(bfp_kymo, _fname, dpi=dpi, fmt=fmt)


if __name__ == "__main__":
    main(
        save=True,
    )
