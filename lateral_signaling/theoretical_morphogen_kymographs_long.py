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
def M_y_t(y, t, psi, M_bar, M_max):
    """
    Concentration of self-activating morphogen over time.
    Initial condition is an exponential gradient, and self-activation
    follows the logistic equation.
    Diffusion and advection are assumed to be negligible.
    """
    M_y_0 = np.log(psi) / (psi - 1) * M_bar * psi ** y
    return M_max * M_y_0 * np.exp(t) / (M_max + M_y_0 * (np.exp(t) - 1))


@numba.njit
def y_t_fixedM(t, fixedM, psi, M_bar, M_max):
    """
    Returns the y-position with a particular M value (fixedM) over time.
    Initial condition is an exponential gradient, and growth
    follows the logistic equation.
    """
    M_0 = np.log(psi) / (psi - 1) * M_bar
    return np.log(
        fixedM
        * M_max
        * np.exp(-t)
        / (M_max - fixedM * (1 - np.exp(-t)))
        / M_0
    ) / np.log(psi)


def effector_response(M, M_opt, scale, width):
    """Equilibrium concentration of a hypothetical effector in response to morphogen"""
#    E = stats.norm.pdf(M, M_opt, scale)
#    E = stats.cauchy.pdf(M, M_opt, scale)
    E = 1 - np.abs(M_opt - M) / M_opt
#    E = 1 / (1 + scale * (M_opt - M) ** 2)
#    Ea = M ** scale / ((M_opt - width/2) ** scale + M ** scale)
#    Ei = (M_opt + width/2) ** scale / ((M_opt + width/2) ** scale + M ** scale)
#    E = Ea * Ei
    
    return E / E.max()

## The default parameters below were chosen for demonstration purposes.
##   They are not meant to estimate the experiments shown in
##   Figure 6, but were rather chosen to show how realistic patterning
##   behaviors can arise from simple non-monotonic morphogen-effector 
##   relationships.

def main(
    tmax = 6.5,
    ny = 101,
    nt = 201,
    M_max = 6,
    M_bar = 1.,
    gradient_steepness = 20,
    clevels_M=15,
    clevels_E=10,
    scale = 1.,
    width = 1.,
    act_thresh = 0.5,
    prefix=save_pfx,
    suffix="",
    save=False,
    fmt="png",
    dpi=300,
):
    
    # Set optimal morphogen range
    #M_opt_lo, M_opt_hi = 1.4, 3.8
    M_opt = M_max / 2

    ## Spatiotemporal parameters
    # Sample along y-direction
    y_space = np.linspace(0, 1, ny)

    # Sample which time-points to visualize
    t_space = np.linspace(0, tmax, nt)
    dt = t_space[1] - t_space[0]

    ## Initial conditions
    psi_flat = 1 - 1e-10            # Uniform density profile
    psi_grad = 1/gradient_steepness # Steep gradient

    # Calculate morphogen dynamics and normalize 
    M_yt = M_y_t(
        np.tile(y_space, nt), 
        np.repeat(t_space, ny), 
        psi_grad, 
        M_bar,
        M_max,
    ).reshape(nt, ny).T
    M_yt = M_yt[::-1]
    M_yt_norm = M_yt / M_yt.max(axis=0)

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
        aspect=2,
        fontscale=1.5,
        hooks=[lsig.xaxis_label_left, lsig.yaxis_label_bottom]
#        colorbar_opts=dict(shrink=0.5, aspect=3, pad=0.2),
    )

    ## Plot kymograph as image
    M_kymo = hv.Image(
        M_yt,
        bounds=bounds,
    ).opts(
        cmap=cmap_M,
        clabel=r"$\mathrm{[Morphogen]}$",
        clim=(0, M_max),
        cbar_ticks=0,
        **kymo_opts
    )
    
    # Make curve of morphogen self-activation
    M_t = lsig.logistic(t_space, 1., M_bar, M_max)

    # Make curve of effector vs morphogen
    M_space = np.linspace(0, M_max, 100)
    effector_curve_data = effector_response(M_space, M_opt, scale, width)
    
    opt_bounds_idx = np.diff(effector_curve_data > act_thresh).nonzero()
    opt_bounds = M_space[opt_bounds_idx]
#    opt_range_lo =  np.log(act_thresh) + M_opt
#    opt_range_hi = -np.log(act_thresh) + M_opt

    curve_opts = dict(
        linewidth=4,
        c="k",
    )
    vline_opts = dict(
        linewidth=2,
        c="gray",
    )
    effector_plot_opts = dict(
        xlabel=r"$\mathrm{[Morphogen]}$",
        xticks=0,
        ylabel=r"$[\mathrm{Effector}]$",
        ylim=(0, None),
#        yticks=(0,1),
        yticks=0,
        fontscale=2.0,
        padding=0.05,
        aspect=1.3,
    )
    hline_opts = dict(
        linewidth=2,
        linestyle="dashed",
        c="gray",
    )
    morphogen_plot_opts = dict(
        xlabel="Time",
        xticks=0,
        ylabel=r"$[\mathrm{Morphogen}]$",
        ylim=(0, None),
#        yticks=(0,1),
        yticks=0,
        fontscale=2.0,
        padding=0.05,
        aspect=1.3,
    )

    effector_curve = hv.Overlay([
#        *[hv.VLine(b) for b in opt_bounds],
#        hv.VLine(opt_range_lo),
#        hv.VLine(opt_range_hi),
        hv.Curve(
            {"x": M_space, "y":effector_curve_data}
        ),
    ]).opts(
        hv.opts.Curve(**curve_opts),
        hv.opts.VLine(**vline_opts),
        hv.opts.Overlay(**effector_plot_opts),
    )

    morphogen_curve = hv.Overlay([
        hv.HLine(M_max),
        hv.Curve(
            {"x": t_space, "y": M_t}
        ),
    ]).opts(
        hv.opts.HLine(**hline_opts),
        hv.opts.Curve(**curve_opts),
        hv.opts.Overlay(**morphogen_plot_opts),
    )

    E_yt = effector_response(M_yt, M_opt, scale, width)
    E_yt_norm = E_yt / E_yt.max(axis=0)

    # Plot kymograph as image
    kgy_list = [lsig.rgba2hex(c) for c in lsig.kgy.colors]
    cmap_E = lsig.sample_cycle(kgy_list, clevels_E).values

    E_kymo = hv.Image(
        E_yt,
        bounds=bounds,
    ).opts(
        cmap=cmap_E,
        clabel=r"$[\mathrm{Effector}]$",
#        clim=(0, 1),
#        cbar_ticks=[0, 1],
        cbar_ticks=0,
        **kymo_opts
    )

    E_kymo_norm = hv.Image(
        E_yt_norm,
        bounds=bounds,
    ).opts(
        cmap=cmap_E,
        clabel=r"norm. $[\mathrm{Eff}]$",
#        clim=(0, 1),
#        cbar_ticks=[0, 1],
        cbar_ticks=0,
        **kymo_opts
    )
    
    if save:
        
        plots = [
            morphogen_curve, 
            effector_curve, 
            M_kymo, 
            E_kymo,
            E_kymo_norm,
        ]
        names = [
            "morphogen_vs_time",
            "effector_vs_morphogen",
            "morphogen_kymograph_long",
            "effector_kymograph_long",
            "effector_kymograph_long_norm",
        ]
        for plot, name in zip(plots, names):
            fpath = prefix + name + suffix
            _fpath = fpath + "." + fmt
            print(f"Writing to: {_fpath}")
            hv.save(plot, fpath, fmt=fmt, dpi=dpi)


main(
    scale=1,
#    scale=8,
    width=1.,
    save=True,
    suffix="__",
)

