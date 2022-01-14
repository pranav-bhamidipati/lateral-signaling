import os
from glob import glob
import json
from copy import deepcopy
import h5py

import numpy as np
import pandas as pd
from tqdm import tqdm

import colorcet as cc
import matplotlib.pyplot as plt
import matplotlib as mpl
import holoviews as hv
hv.extension("matplotlib")

import lateral_signaling as lsig

data_dir     = os.path.abspath("../data/simulations/20220111_constantdensity/sacred")
save_dir     = os.path.abspath("../plots")
layout_fpath = os.path.join(save_dir, "constant_density_imlayout_")
curves_fpath = os.path.join(save_dir, "constant_density_sqrtarea_")
fmt          = "png"
dpi          = 300

def main(
    layout_fpath=layout_fpath,
    curves_fpath=curves_fpath,
    delays_to_plot=[],
    curves_tmax=None,
    pad=0.05,
    save_layout=False,
    save_curves=False,
    fmt=fmt,
    dpi=dpi,
):

    # Read in data from experiments
    run_dirs = glob(os.path.join(data_dir, "[0-9]*"))

    # Define data to read
    rhos = []
    S_ts = []
    R_ts = []

    for i, rd in enumerate(run_dirs):
        
        # Read data from files
        config_file  = os.path.join(rd, "config.json")
        results_file = os.path.join(rd, "results.hdf5")
        
        if i == 0:
            with open(config_file, "r") as c:
                config = json.load(c)

                # Dimensions of cell sheet
                rows = config["rows"]
                cols = config["cols"]

                # Delay parameter
                delay = config["delay"]

                # Threshold parameter (for area calculation)
                k = config["k"]
            
        with h5py.File(results_file, "r") as f:

            if i == 0:

                # Time-points
                t = np.asarray(f["t"])
                dt = t[1] - t[0]

                # Index of sender cell
                sender_idx = np.asarray(f["sender_idx"])

            # Density (constant)
            rho = np.asarray(f["rho_t"])[0]

            # Signal and reporter expression vs. time
            S_t = np.asarray(f["S_t"])
            R_t = np.asarray(f["R_t"])

        # Store data
        rhos.append(rho)
        S_ts.append(S_t)
        R_ts.append(R_t)

    sort_rhos   = np.argsort(rhos)
    rhos        = np.asarray(rhos)[sort_rhos]
    S_ts        = np.asarray(S_ts)[sort_rhos]
    R_ts        = np.asarray(R_ts)[sort_rhos]
    
    # Convert selected time-points from units of delay
    plot_times = np.array([delay * p for p in delays_to_plot])

    # Get indices of the closest time-points
    plot_frames = np.argmin(np.subtract.outer(t, plot_times) ** 2, axis=0)

    # Make a lattice centered on the sender
    X = lsig.hex_grid(rows, cols)
    X = X - X[sender_idx]

    # Get mask of non-sender cells (transceivers)
    n = X.shape[0]
    ns_mask = np.ones(n, dtype=bool)
    ns_mask[sender_idx] = False
    
    # Get cell positions based on density
    Xs = np.multiply.outer(1 / np.sqrt(rhos), X)

    ## Some manual plotting options
    # Font sizes
    SMALL_SIZE  = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    
    # Zoom in to a factor of `zoom` (to emphasize ROI)
    zoom = 0.7

    if save_layout:
        
        # Set font sizes
        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
        
        # Get default kwargs for plotting
        plot_kwargs = deepcopy(lsig.plot_kwargs)
        plot_kwargs["sender_idx"] = sender_idx
        
        # Turn on scalebar
        plot_kwargs["scalebar"] = True

        # Axis title
        plot_kwargs["title"] = ""
        
        # axis limits
        densest_lattice = np.argmax(rhos)
        _xmax = np.abs(Xs[densest_lattice, :, 0]).max()
        _ymax = np.abs(Xs[densest_lattice, :, 1]).max()
        plot_kwargs["xlim"] = -_xmax * zoom, _xmax * zoom
        plot_kwargs["ylim"] = -_ymax * zoom, _ymax * zoom

        # colorscale limits
        plot_kwargs["vmin"] = 0
        plot_kwargs["vmax"] = S_ts[:, :(plot_frames.max() + 1), ns_mask].max()

        # some args for colorscale
        plot_kwargs["cmap"] = lsig.kgy
        plot_kwargs["cbar_aspect"] = 8
        plot_kwargs["colorbar"] = False    
        
        # Make figure
        prows = 3
        pcols = 3
        fig, axs = plt.subplots(
            nrows=prows,
            ncols=pcols,
            figsize=(6.2, 5.0),
            gridspec_kw=dict(width_ratios=[1] * (pcols - 1) + [1.2]),
        )
         
        # Plot sheets
        for i, ax in enumerate(axs.flat):
            
            row = i // pcols
            col = i %  pcols

            # Hide scalebar text except first image
            font_size = (i == 0) * 10
            plot_kwargs["sbar_kwargs"]["font_properties"] = dict(
                weight = 1000,
                size   = font_size,
            )
            
            # Plot cell sheet
            lsig.plot_hex_sheet(
                ax=ax,
                X=Xs[row],
                var=S_ts[row, plot_frames[col]],
                rho=rhos[row],
                **plot_kwargs,
            )

            # Make colorbars (empty except in first row)
            if col == pcols - 1:
                cbar = plt.colorbar(
                    plt.cm.ScalarMappable(
                        norm=mpl.colors.Normalize(
                            plot_kwargs["vmin"], plot_kwargs["vmax"]
                        ), 
                        cmap=plot_kwargs["cmap"]), 
                    ax=ax,
                    aspect=plot_kwargs["cbar_aspect"],
                    extend=plot_kwargs["extend"],
                    shrink = (1e-5, 1.)[row == 0],
                    label="",
                    ticks=[],
    #                format="%.2f",
                )
        
        plt.tight_layout()
        
        # Get path and print to output
        _fpath = str(layout_fpath)
        if not _fpath.endswith(fmt):
            _fpath += "." + fmt
        print("Writing to:", _fpath)

        # Save
        plt.savefig(_fpath, dpi=dpi)

    if save_curves:

        ## Plot timeseries of expression
        # Get early time-range 
        if curves_tmax is not None:
            tmax_idx = np.searchsorted(t, delay * curves_tmax)
            tslice   = slice(tmax_idx)
        else:
            tslice   = slice(None)
        
        nt = t[tslice].size

        # Calculate the number of activated transceivers
        n_act_ts = (S_ts[:, tslice] > k).sum(axis=2) - 1

#        # Percent of activated transceivers
#        pct_act_t = n_act_t / n * 100
#
#        # Optionally normalize percentage
#        pct_act_t = lsig.normalize(pct_act_t, 0, pct_act_t.max()) * 100

        # Area and sqrt(Area) of activation
        A_ts = np.array([lsig.ncells_to_area(n, rho) for n, rho in zip(n_act_ts, rhos)])
        sqrtA_ts = np.sqrt(A_ts)

        # Normalize area and sqrt(area)
        A_ts_norm = lsig.normalize(A_ts, 0, A_ts.max())
        sqrtA_ts_norm = lsig.normalize(sqrtA_ts, 0, sqrtA_ts.max())

        # Axis limits with padding
        xmin = 0.0
        xmax = t[tslice][-1]
        ymin = 0.0
        ymax = 0.45 
        xlim = xmin - pad * (xmax - xmin), xmax + pad * (xmax - xmin)
        ylim = ymin - pad * (ymax - ymin), ymax + pad * (ymax - ymin)

        # Make color/linestyle cycles
        ccycle = lsig.sample_cycle(cc.gray[:150], 3)
#        ccycle = hv.Cycle(lsig.cols_blue)
        lcycle = hv.Cycle(["solid", "dashed", "dotted"])

        # Make data
        curve_data = {
            "t"            : np.tile(t[tslice], len(rhos)),
            "A_t"          : A_ts.ravel(),
            "sqrtA_t"      : sqrtA_ts.ravel(), 
            "sqrtA_t_norm" : sqrtA_ts_norm.ravel(),
            "A_t_norm"     : A_ts_norm.ravel(),
            "density"      : np.repeat([fr"$\rho =$ {int(r)}" for r in rhos], nt),
        }
        
        # Tick labels
        xticks = [
            (0 * delay, "0"), 
            (1 * delay, "τ"), 
            (2 * delay, "2τ"), 
            (3 * delay, "3τ"), 
            (4 * delay, "4τ"), 
        ]

        # Plot curves
        curve_plot = hv.Curve(
            curve_data,
            kdims=["t"],
            vdims=["sqrtA_t", "density"],
        ).groupby(
            "density",
        ).opts(
            xlabel="Simulation time",
            xlim=xlim,
            xticks=xticks,
            ylabel=r"$\sqrt{Area}$ ($mm$)",
#            ylabel=r"$\sqrt{Area}$ (norm.)",
            ylim=ylim,
            yticks=[0.0, 0.1, 0.2, 0.3, 0.4],
            linewidth=2,
            linestyle=lcycle,
            color=ccycle,
#            color="k",
            aspect=1,
        ).overlay(
            "density"
        ).opts(
#            show_legend=False,
            legend_position="right",
            fontscale=1.3,
        )

        # Save
        _fpath = curves_fpath + "." + fmt
        print("Writing to:", _fpath)
        hv.save(curve_plot, curves_fpath, fmt=fmt, dpi=dpi)


main(
    save_layout=True,
    save_curves=True,
#    plot_frames=(100, 200, 300),
    delays_to_plot=[2, 4, 6],
    curves_tmax=4,
)
