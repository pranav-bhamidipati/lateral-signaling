import lateral_signaling as lsig

import os

import numpy as np
import scipy.stats as stats
import pandas as pd
from tqdm import tqdm
import numba

import holoviews as hv
import colorcet as cc
import cmocean.cm as cmo
hv.extension("matplotlib")

# Parameters for saving
save_dir = os.path.abspath("../plots")
save_pfx = os.path.join(save_dir, "theoretical_patterning_")

### Function definitions
###   See supplemental text for derivation and exposition on the below functions

@numba.njit
def rho_y_0(y, psi, rho_bar):
    """
    Density of a deforming lattice over time.
    Initial condition is an exponential gradient, and growth
    follows the logistic equation.
    """
    return np.log(psi) / (psi - 1) * rho_bar * psi ** y


@numba.njit
def rho_y_t(y, t, psi, rho_bar, rho_max):
    """
    Density of a deforming lattice over time.
    Initial condition is an exponential gradient, and growth
    follows the logistic equation.
    """
    rho_y_0 = np.log(psi) / (psi - 1) * rho_bar * psi ** y
    return rho_max * rho_y_0 * np.exp(t) / (rho_max + rho_y_0 * (np.exp(t) - 1))


@numba.njit
def y_t_fixedrho(t, fixedrho, psi, rho_bar, rho_max):
    """
    Returns the y-position with a particular rho value (fixedrho) over time.
    Initial condition is an exponential gradient, and growth
    follows the logistic equation.
    """
    rho_0 = np.log(psi) / (psi - 1) * rho_bar
    return np.log(
        fixedrho
        * rho_max
        * np.exp(-t)
        / (rho_max - fixedrho * (1 - np.exp(-t)))
        / rho_0
    ) / np.log(psi)


def signaling_activity(rho, rho_opt, scale):
    """Activity of a hypothetical effector in response to morphogen"""
    return stats.norm.pdf(rho, rho_opt, scale) / stats.norm.pdf(rho_opt, rho_opt, scale)


## The default parameters below were chosen for demonstration purposes.
##   They are not meant to estimate the experiments shown in
##   Figure 6, but were rather chosen to show how realistic patterning
##   behaviors can arise from simple non-monotonic morphogen-effector 
##   relationships.

def main(
    tmax = 5,
    ny = 101,
    nt = 101,
    rho_max = 6,
    rho_bar_lo = 0.7,
    rho_bar_hi = 4.0,
    gradient_steepness = 20,
    clevels_M=15,
    clevels_E=10,
    scale = 1.,
    act_thresh = 0.25,
#    rho_opt = 2.6,
    prefix=save_pfx,
    suffix="",
    save=False,
    fmt="png",
    dpi=300,
):
    
    # Set optimal morphogen range
    #rho_opt_lo, rho_opt_hi = 1.4, 3.8
    rho_opt = rho_max / 2

    ## Spatiotemporal parameters
    # Sample along y-direction
    y_space = np.linspace(0, 1, ny)

    # Sample which time-points to visualize
    t_space = np.linspace(0, tmax, nt)
    dt = t_space[1] - t_space[0]

    ## Initial conditions
    psi_flat = 1 - 1e-10            # Uniform density profile
    psi_grad = 1/gradient_steepness # Steep gradient

    # Calculate density dynamics and normalize 
    rho_yt_grad = rho_y_t(
        np.tile(y_space, nt), 
        np.repeat(t_space, ny), 
        psi_grad, 
        rho_bar_lo,
        rho_max,
    ).reshape(nt, ny).T
    rho_yt_grad = rho_yt_grad[::-1]
    rho_yt_grad_norm = rho_yt_grad / rho_yt_grad.max(axis=0)
    
    # Repeat for higher initial denisty
    rho_yt_grad2 = rho_y_t(
        np.tile(y_space, nt), 
        np.repeat(t_space, ny), 
        psi_grad, 
        rho_bar_hi,
        rho_max,
    ).reshape(nt, ny).T
    rho_yt_grad2 = rho_yt_grad2[::-1]
    rho_yt_grad2_norm = rho_yt_grad2 / rho_yt_grad2.max(axis=0)

    ## Plotting options
    # Bounds for plotting kymograph as an image
    bounds = (0, 0, tmax, 1,)
    
    # Colormap
    cmap_M = lsig.sample_cycle(cc.kbc, clevels_M).values

    # Other plot options
    kymo_opts = dict(
        xlabel="Time",
        xlim=(0, tmax),
        xticks=0,
        ylabel="Space",
        ylim=(1, 0),
        yticks=0,
        colorbar=True,
        aspect=1,
        fontscale=2.0,
#        colorbar_opts=dict(shrink=0.5, aspect=3, pad=0.2),
    )

    ## Plot kymograph as image
    rho_kymo_grad = hv.Image(
        rho_yt_grad,
        bounds=bounds,
    ).opts(
        cmap=cmap_M,
        clabel=r"$\mathrm{[Morphogen]}$",
        clim=(0, rho_max),
#        cbar_ticks=[(0, "0"), (rho_max, r"$\mathrm{[Morphogen]}_\mathrm{max}$")],
        cbar_ticks=0,
        **kymo_opts
    )

    rho_kymo_grad2 = hv.Image(
        rho_yt_grad2,
        bounds=bounds,
    ).opts(
        cmap=cmap_M,
        clabel=r"$\mathrm{[Morphogen]}$",
        clim=(0, rho_max),
#        cbar_ticks=[(0, "0"), (rho_max, r"$\mathrm{[Morphogen]}_\mathrm{max}$")],
        cbar_ticks=0,
        **kymo_opts
    )

    # Make curve of effector activity vs morphogen (density)
    rho_space = np.linspace(0, rho_max, 100)
    activity_curve_data = signaling_activity(rho_space, rho_opt, scale)
    
    opt_bounds_idx = np.diff(activity_curve_data > act_thresh).nonzero()
    opt_bounds = rho_space[opt_bounds_idx]
#    opt_range_lo =  np.log(act_thresh) + rho_opt
#    opt_range_hi = -np.log(act_thresh) + rho_opt

    curve_opts = dict(
        linewidth=4,
        c="k",
    )
    vline_opts = dict(
        linewidth=2,
        c="gray",
    )
    signaling_plot_opts = dict(
        xlabel=r"$\mathrm{[Morphogen]}$",
        xticks=0,
        ylabel=r"$[\mathrm{Eff}]_{eq}$",
        ylim=(0, None),
#        yticks=(0,1),
        yticks=0,
        fontscale=2.0,
        padding=0.05,
        aspect=1.5,
    )

    activity_curve = hv.Overlay([
        *[hv.VLine(b) for b in opt_bounds],
#        hv.VLine(opt_range_lo),
#        hv.VLine(opt_range_hi),
        hv.Curve(
            {"x": rho_space, "y":activity_curve_data}
        ),
    ]).opts(
        hv.opts.Curve(**curve_opts),
        hv.opts.VLine(**vline_opts),
        hv.opts.Overlay(**signaling_plot_opts),
    )

    Act_yt_grad = signaling_activity(rho_yt_grad, rho_opt, scale)
    Act_yt_grad_norm = Act_yt_grad / Act_yt_grad.max(axis=0)

    Act_yt_grad2 = signaling_activity(rho_yt_grad2, rho_opt, scale)
    Act_yt_grad2_norm = Act_yt_grad2 / Act_yt_grad2.max(axis=0)
    
    # Plot kymograph as image
    kgy_list = [lsig.rgba2hex(c) for c in lsig.kgy.colors]
    cmap_TF = lsig.sample_cycle(kgy_list, clevels_TF).values

    act_kymo_grad = hv.Image(
        Act_yt_grad,
        bounds=bounds,
    ).opts(
        cmap=cmap_TF,
        clabel=r"$[\mathrm{TF}]_{eq}$",
#        clim=(0, 1),
#        cbar_ticks=[0, 1],
        cbar_ticks=0,
        **kymo_opts
    )

    act_kymo_grad_norm = hv.Image(
        Act_yt_grad_norm,
        bounds=bounds,
    ).opts(
        cmap=cmap_TF,
        clabel=r"norm. $[\mathrm{TF}]_{eq}$",
#        clim=(0, 1),
#        cbar_ticks=[0, 1],
        cbar_ticks=0,
        **kymo_opts
    )
    
    act_kymo_grad2 = hv.Image(
        Act_yt_grad2,
        bounds=bounds,
    ).opts(
        cmap=cmap_TF,
        clabel=r"$[\mathrm{TF}]_{eq}$",
#        clim=(0, 1),
#        cbar_ticks=[0, 1],
        cbar_ticks=0,
        **kymo_opts
    )

    act_kymo_grad2_norm = hv.Image(
        Act_yt_grad2_norm,
        bounds=bounds,
    ).opts(
        cmap=cmap_TF,
        clabel=r"norm. $[\mathrm{TF}]_{eq}$",
#        clim=(0, 1),
#        cbar_ticks=[0, 1],
        cbar_ticks=0,
        **kymo_opts
    )


    if save:
        
        plots = [
            activity_curve, 
            rho_kymo_grad, 
            act_kymo_grad_norm,
            rho_kymo_grad2, 
            act_kymo_grad2_norm,
        ]
        names = [
            "TF_activation_vs_morphogen",
            "morphogen_kymograph",
            "TF_kymograph",
            "morphogen_kymograph2",
            "TF_kymograph2",
        ]
        for plot, name in zip(plots, names):
            fpath = prefix + name + suffix
            _fpath = fpath + "." + fmt
            print(f"Writing to: {_fpath}")
            hv.save(plot, fpath, fmt=fmt, dpi=dpi)


main(
    save=False,
    suffix="__",
)

