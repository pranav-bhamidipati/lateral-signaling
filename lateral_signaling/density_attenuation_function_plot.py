import os

import pandas as pd
import numpy as np
import scipy.stats as stats
from tqdm import tqdm

import colorcet as cc
import holoviews as hv
hv.extension("matplotlib")

import matplotlib.pyplot as plt

import lateral_signaling as lsig

save_dir = os.path.abspath("../plots")
fname    = "signaling_vs_density_curve_"

def main(
    rows=12,
    m=1.,
    rhomin=1,
    rhomax=7,
    nrho=101,
    cmap="viridis",
    figsize=(6.2, 6.5),
    save=False,
    fmt="png",
    dpi=300,
):
    
    ## Construct data for example
    # make cell sheet
    cols = rows
    X = lsig.hex_grid(rows, cols)
    center = lsig.get_center_cells(X)
    X = X - X[center]

    # Store cell type
    var = np.zeros(X.shape[0], dtype=np.float32)
    var[center] = 1.

    # Sample beta(rho, m) in the defined space
    rho_space = np.linspace(rhomin, rhomax, nrho)
    beta_space = lsig.beta_rho_exp(rho_space, m)
    
    # Set which rhos to use as examples
    rhos = np.array([1., 2., 4.])
    betas = lsig.beta_rho_exp(rhos, m)
    X_rho = np.multiply.outer(1/np.sqrt(rhos), X)
    
    ## Plotting options
    # Set axis limits for inset plots
    _extent = np.min([-X_rho[-1].min(axis=0).max(), X_rho[-1].max(axis=0).min()])
    xlim = -_extent, _extent
    ylim = -_extent, _extent
    
    # Set locations for insets
    inset_locs = np.array([
        [1.34, 0.72 ],
        [2.59, 0.335],
        [4.85, 0.195],
    ])
    ins_width, ins_height = 1.75, 0.275

    # Set axis limits for larger plot
    plot_xlim = rhomin - 0.5, rhomax + 0.5
    plot_ylim = -0.05, 1.05
    
    ## Make plot
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot curve
    ax.plot(rho_space, beta_space)

    # Plot lattices as inset plots
    for i, _ in enumerate(rhos):
        
        # Make inset axis
        x_ins, y_ins = inset_locs[i] 
        axins = ax.inset_axes(
            [x_ins, y_ins, ins_width, ins_height],
            transform=ax.transData,
        )
        
        # Draw line from inset to curve
        endpoint_x = x_ins + ins_width/6
        endpoint_y = y_ins + ins_height/2
        ax.plot(
            (endpoint_x, rhos[i]), 
            (endpoint_y, betas[i]), 
            c="k",
        )
        
        # Plot cells
        lsig.plot_hex_sheet(
            ax=axins, 
            X=X_rho[i], 
            var=var, 
            rho=rhos[i], 
            sender_idx=center, 
            xlim=xlim, 
            ylim=ylim,
            ec="w",
            lw=0.2,
            # axis_off=False,
        )
    
    # Add titles/labels
    ax.set_title("Effect of density on signaling", fontsize=18)
    ax.text(
        rho_space.max(), 
        beta_space.max(), 
        r"$\beta(\rho, m)=e^{-m(\rho-1)}$", 
        ha="right", 
        va="top", 
        fontsize=14,
    )
    ax.text(
        rho_space.max(), 
        beta_space.max() - 0.125, 
        f"$m={m:.2f}$", 
        ha="right", 
        va="top", 
        fontsize=14,
    )
    
    # Set other options
    ax.set_xlim(plot_xlim)
    ax.set_ylim(plot_ylim)
    ax.set_xlabel(r"$\rho$", fontsize=16)
    ax.set_ylabel("Signaling coefficient", fontsize=16)
    plt.tick_params(labelsize=12)
    # ax.set_aspect(4)
    
    plt.tight_layout()

    if save:
        fpath = os.path.join(save_dir, fname + "." + fmt)
        print(f"Writing to: {fpath}")
        plt.savefig(fpath, dpi=dpi, format=fmt)


main(
    save=True,
)

