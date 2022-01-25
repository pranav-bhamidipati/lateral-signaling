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
fname    = "weighted_adjacency_"

def main(
    rows=12,
    boxsize=8,
    imsize=101,
    cmap="viridis",
    figsize=(6.2, 6.5),
    save=False,
    fmt="png",
    dpi=300,
):
    
    nrho = 101
    rho_space = np.linspace(1, 7, nrho)
    beta_space = lsig.beta_rho_exp(rho_space, 1.)


    # In[ ]:


    rhos = np.array([1., 2., 4.])
    betas = lsig.beta_rho_exp(rhos, 1.)
    X_rho = [
        X / np.sqrt(rho)
        for rho in rhos
    ]

    _extent = np.min([-X_rho[-1].min(axis=0).max(), X_rho[-1].max(axis=0).min()])
    xlim = -_extent, _extent
    ylim = -_extent, _extent

    plot_xlim = 0.5, 7.5
    plot_ylim = -0.05, 1.05

    var = np.zeros(X.shape[0], dtype=np.float32)
    var[center] = 1.


    # In[ ]:


    fig, ax = plt.subplots(figsize=(6, 4))

    # # Title of plot
    # plt.suptitle(r"Coefficient $\beta$ encodes density-dependence")

    # Plot curve
    ax.plot(rho_space, beta_space)

    # Plot lattices as inset plots
    inset_locs = np.array([
        [1.34, 0.72 ],
        [2.59, 0.335],
        [4.85, 0.195],
    ])
    ins_width, ins_height = 1.75, 0.275

    for i in range(3):
        x_ins, y_ins = inset_locs[i]
        endpoint_x = x_ins + ins_width/6
        endpoint_y = y_ins + ins_height/2
        
        axins = ax.inset_axes(
            [x_ins, y_ins, ins_width, ins_height],
            transform=ax.transData,
        )
        
        ax.plot(
            (endpoint_x, rhos[i]), 
            (endpoint_y, betas[i]), 
            c="k",
        )
        
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
        r"$m=1.00$", 
        ha="right", 
        va="top", 
        fontsize=14,
    )
    ax.set_xlim(plot_xlim)
    ax.set_ylim(plot_ylim)
    ax.set_xlabel(r"$\rho$", fontsize=16)
    ax.set_ylabel("Signaling coefficient", fontsize=16)
    plt.tick_params(labelsize=12)
    # ax.set_aspect(4)

    if save:
        fig_fname = "signaling_dampening_w_density"
        fig_path = os.path.join(save_dir, fig_fname + "." + fmt)
        plt.savefig(fig_path, dpi=dpi, format=fmt)


main(
    save=False,
)

