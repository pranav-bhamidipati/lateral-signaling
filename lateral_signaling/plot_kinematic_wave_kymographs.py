import json

import numpy as np
import colorcet as cc
import holoviews as hv

import lateral_signaling as lsig

hv.extension("matplotlib")

image_dir = lsig.data_dir.joinpath("imaging/kinematic_wave")
img_analysis_dir = lsig.analysis_dir.joinpath("kinematic_wave")

lp_data_path = img_analysis_dir.joinpath("line_profile_data.json")


def main(
    save_dir=lsig.plot_dir,
    save=False,
    fmt="png",
    dpi=300,
):

    # Load data from file
    with lp_data_path.open("r") as f:
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
    nt = t_days.size // 2
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
    vbar_sd = np.std(velocities)

    ## Make kymographs of fluorescence over time

    # Plot kymographs as images
    bounds = (t_days.min() - 0.5, position.min(), t_days.max() + 0.5, position.max())
    gfpimage_opts = dict(
        # colorbar=True,
        # #        cbar_ticks=[(0, "0"), (1, "1")],
        # cbar_ticks=0,
        # cbar_width=0.05,
        # cbar_padding=-0.13,
    )
    bfpimage_opts = dict(
        # colorbar=True,
        # #    cbar_ticks=[(0, "0"), (1, "1")],
        # cbar_ticks=0,
        # cbar_width=0.1,
    )
    plot_opts = dict(
        xlabel="Day",
        xticks=tuple(t_days),
        yticks=(0, 1, 2, 3, 4),
        ylabel="Position (mm)",
        aspect=0.75,
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
                5.25,
                4.2,
                r"$\bar{\mathit{v}} = "
                + f"{vbar:.2f} \pm {vbar_sd:.2f}"
                + r"$"
                + "\n"
                + r"mm/day",
                halign="right",
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

        _path = save_dir.joinpath(f"kinematic_wave_kymograph_gfpnorm.{fmt}")
        print(f"Writing to:", _path.resolve().absolute())
        hv.save(gfp_kymo, _path, dpi=dpi, fmt=fmt)

        _path = save_dir.joinpath(f"kinematic_wave_kymograph_bfpnorm.{fmt}")
        print(f"Writing to:", _path.resolve().absolute())
        hv.save(bfp_kymo, _path, dpi=dpi, fmt=fmt)


if __name__ == "__main__":
    main(
        save_dir=lsig.temp_plot_dir,
        save=True,
    )
