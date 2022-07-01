import os
from functools import partial
import multiprocessing as mp
import psutil

import pandas as pd
import numpy as np
#from tqdm import tqdm

#import colorcet as cc
import holoviews as hv
hv.extension("matplotlib")

import matplotlib.pyplot as plt

import lateral_signaling as lsig
lsig.default_rcParams()

# File I/O
data_dir           = os.path.abspath("../data/FACS")
metadata_res_fname = os.path.join(data_dir, "metadata_with_LLR.csv")
save_dir           = os.path.abspath("../plots/tmp")

spikeplot_fname    = "FACS_likelihood_spikeplot_"
LLR_vs_nbins_fname = "FACS_llr_vs_num_bins_"


def main(
    metadata_res,
    idx_to_plot,
    llrs_v_nbins,
    spikeplot_fname=spikeplot_fname, 
    LLR_vs_nbins_fname=LLR_vs_nbins_fname,
    transparent=True,
    **kw
):
    """"""
    plot_spikeplot(
        metadata_res, idx_to_plot, spikeplot_fname, **kw 
    )
    plot_LLR_v_nbins(
        metadata_res, llrs_v_nbins, LLR_vs_nbins_fname, transparent, **kw
    )


def plot_spikeplot(
    metadata_res,
    idx_to_plot,
    fname=spikeplot_fname,
    plot_errors=True,
    save_dir=save_dir,
    save=False,
    fmt="png",
    dpi=300,
    **kw
):
    nrows = metadata_res.shape[0]
    nplot = len(idx_to_plot)
    llr_data = metadata_res.iloc[idx_to_plot].copy()
    llr_data = llr_data.reset_index(drop=True)
    llr_data.index.name = "id_num"
    llr_data = llr_data.reset_index()

    # Starting y-value of each spike in the spike plot
    llr_data["y0"] = 0.

    # Axis limits
    xlim = (-1, llr_data.shape[0])
    ylim = (-4500, 4500)

    # Labels for each group of samples
    group_labels = ("Ref.", "Cell density", "Cytoskel.\ntension", "ECM")

    # Make an overline whose color will denote experimental grouping
    overline_xbuffer = 0.35
    overline_ybuffer = -0.00
    overline_text_buffer = 0.015

    # Initialize endpoints and coloring of overline 
    overline_ypos = (1 + overline_ybuffer) * (ylim[1] - ylim[0]) + ylim[0]
    overline_endpoints = overline_ypos * np.ones((4, 4))
    overline_colors = metadata_res.Color_dark.values[:4]

    for i, char in enumerate("Racb"):
        
        # Get reference samples (ON and OFF)
        if char == "R":
            grp = ~llr_data.State.str.contains("-")
            
        # Get indices of samples in this group, excluding OFF and ON samples
        else:
            grp = llr_data.subplot.str.contains(char) & llr_data.State.str.contains("-")
        
        # Get x-values of samples in group
        grp_idx = grp.values.nonzero()[0]
        grp_xvals = llr_data.id_num.values[grp_idx]
        
        # Get color
        overline_colors[i] = llr_data.Color_dark.values[grp_idx[0]]
        
        # Store lowest and highest as endpoints, with buffer
        overline_endpoints[i, 0] = grp_xvals.min() - overline_xbuffer
        overline_endpoints[i, 2] = grp_xvals.max() + overline_xbuffer

        
    ## Plot log-likelihood ratio as a spikeplot
    llr_points = hv.Points(
        llr_data,
        kdims=["id_num", "LLR_mean"],
    ).opts(
        # c=colors,
        c="k",
        s=25,
    )

    llr_segments = hv.Segments(
        llr_data,
        [
            "id_num",
            "y0",
            "id_num",
            "LLR_mean",
        ],
    ).opts(
        # color=colors,
        color="k",
    )

    llr_xaxis = hv.Segments(
        (xlim[0], 0, xlim[1], 0),
    ).opts(
        color="k",
        lw=1,
    )

    llr_overlines = hv.Segments(
        overline_endpoints,
    ).opts(
        # color="k",
        color=overline_colors,
        lw=4,
    )

    llr_group_text_list = []
    for i, _label in enumerate(group_labels):
        _xpos = np.mean([overline_endpoints[i, 0], overline_endpoints[i, 2]])
        _ypos = (1 + overline_ybuffer + overline_text_buffer) * (ylim[1] - ylim[0]) + ylim[0]
        _group_text = hv.Text(
            _xpos,
            _ypos,
            _label,
            fontsize=8,
            valign="bottom",
        )
        llr_group_text_list.append(_group_text)

    llr_group_text = hv.Overlay(llr_group_text_list)

    ylabel = r"Log-Likelihood$\left[\,\frac{\mathrm{signaling\; ON}}{\mathrm{signaling\; OFF}}\,\right]$"
    yticks = [-4000, -2000, 0, 2000, 4000]

    llr_spikeplot = (
        llr_segments * llr_points * llr_xaxis * llr_overlines * llr_group_text
    ).opts(
        hooks=[lsig.remove_RT_spines],
    ).opts(
        xlim=xlim,
        xaxis=False,
        ylabel=ylabel,
        yticks=yticks,
        ylim=ylim,
        aspect=2,
    )

    # Save
    if not plot_errors:
        if save:
            fpath = os.path.join(save_dir, fname + "." + fmt)
            print("Writing to:", fpath)
            hv.save(llr_spikeplot, fpath, dpi=dpi)
    else:

        ## Plot LLR as a scatterplot with confidence bounds
        # Get x-values for plot
        x_vals = -np.ones(nrows)
        x_vals[idx_to_plot] = np.arange(nplot)

        # Get LLR means and Confidence intervals
        llr_mean    = metadata_res.LLR_mean.values
        llr_95CI_lo = metadata_res.LLR_95CI_lo
        llr_95CI_hi = metadata_res.LLR_95CI_hi

        # Get data arrays in format that Holoviews expects
        pointsdata = np.array([x_vals, llr_mean]).T
        errdata    = np.array([
            x_vals, llr_mean, llr_95CI_lo - llr_mean, llr_95CI_hi - llr_mean
        ]).T

        # Select data to plot
        pointsdata = pointsdata[idx_to_plot]
        errdata    = errdata[idx_to_plot]

        # Options
        errplot_opts = [
            hv.opts.HLine(
                c="k", lw=1
            ),
            hv.opts.ErrorBars(
                capsize=4, color="k"
            ),
            hv.opts.Overlay(
                xaxis=False,
                xlim=xlim,
                # xlabel="Samples", 
                # xticks=[(i, "") for i in range(nrows)],
                ylabel = r"Log-Likelihood$\left[\,\frac{\mathrm{signaling\; ON}}{\mathrm{signaling\; OFF}}\,\right]$",
                yticks = [-4000, -2000, 0, 2000, 4000],
                ylim=ylim,
                aspect=2,
            ),
            hv.opts.Scatter(
                s=20, 
                # color=llr_data.Color_dark.values,
                color="k",
            ),
        ]

        # Plot with error bars
        errplot = hv.Overlay([
            hv.HLine(0),
            hv.ErrorBars(
                errdata
            ),
            hv.Scatter(
                pointsdata
            ).opts(
                marker="o",
            ),
            llr_overlines,
            llr_group_text,
        ]).opts(
            *errplot_opts, 
        )

        if save:
            fpath = os.path.join(save_dir, fname + "." + fmt)
            print("Writing to:", fpath)
            hv.save(errplot, fpath, fmt=fmt, dpi=dpi)


def plot_LLR_v_nbins(
    metadata_res, 
    llrs_v_nbins,
    fname, 
    transparent,
    save_dir=save_dir,
    save=False, 
    fmt="png", 
    dpi=300,
    **kw
):
    # Set plot options
    nbins_opts = [
        hv.opts.Overlay(
            aspect=2,
           legend_position="right",
            # legend_position="bottom",
            logx=True,
            xlabel="# histogram bins",
            ylabel=r"Log-Likelihood$\left[\,\frac{\mathrm{signaling\; ON}}{\mathrm{signaling\; OFF}}\,\right]$",

        ), 
        hv.opts.Curve(
            linewidth=1,
        ),
        hv.opts.HLine(
            c="k",
            linestyle="dotted",
            linewidth=1,
        ),
    ]

    # Plot LLR as a function of # bins used for each sample
    llr_nbins_plot = hv.Overlay(
        [
            hv.Curve((nbins_range, llr), label=label).opts(c=clr, linestyle=ls)
            for llr, label, clr, ls in zip(
                llrs_v_nbins.T, 
                metadata_res.Label,
                metadata_res.Color_shaded,
                metadata_res.Linestyle,
            )
        
        # Add a line showing log-ratio of zero (equal likelihood)
        ] + [hv.HLine(0)]
    ).opts(*nbins_opts)

    if save: 
        fpath = os.path.join(save_dir, fname + "." + fmt)
        print("Writing to:", fpath)
        hv.save(llr_nbins_plot, fpath, dpi=dpi, transparent=transparent)


#refdata_idx = np.array([OFF_idx, ON_idx])


def calculate_LLR(_nbins, data, OFF_idx, ON_idx):
    """
    """
    _data_hists = np.array([lsig.data_to_hist(d.values, _nbins)[0] for d in data])

    # Add 1 to each bin to avoid div by 0. (regularization) 
    _data_hists = _data_hists + 1
    _data_hists_pdf    = _data_hists / np.sum(_data_hists, axis=1, keepdims=True)
    _data_hists_logpdf = np.log10(_data_hists_pdf)

    # Compare to reference log-PDFs
    _OFF_hist_logpdf = _data_hists_logpdf[OFF_idx]
    _ON_hist_logpdf  = _data_hists_logpdf[ON_idx]
    log_like_OFF  = np.sum(_data_hists *  _OFF_hist_logpdf, axis=1)
    log_like_ON   = np.sum(_data_hists *   _ON_hist_logpdf, axis=1)
    
    return log_like_ON - log_like_OFF


# Parallelize calculation of LLR
if __name__ == '__main__':
    
    ## Max number of bins to try
    max_nbins = 1000000
    n_nbins = 500
#    n_nbins = 5

    # All # of bins to try
    nbins_range = np.unique(np.geomspace(1, max_nbins, n_nbins).astype(int))

    metadata_res = pd.read_csv(metadata_res_fname)
    data = [pd.read_csv(os.path.join(data_dir, f)).squeeze() for f in metadata_res.filename]
    OFF_idx = metadata_res.State.str.contains("OFF").values.nonzero()[0][0]
    ON_idx  = metadata_res.State.str.contains("ON").values.nonzero()[0][0]
    
    get_LLRs_given_nbins = partial(
        calculate_LLR, data=data, OFF_idx=OFF_idx, ON_idx=ON_idx
    )
    
    n_threads = psutil.cpu_count(logical=False)
    print(f"Assembling thread pool ({n_threads} workers)")
    pool = mp.Pool(n_threads)
    
    print("Computing LLR")
    results = pool.map(get_LLRs_given_nbins, nbins_range, chunksize=5)
    pool.close()
    pool.join()

    print("Complete")
    
    llrs_v_nbins = np.array(results)
#    llrs_v_nbins  = log_like_ON_v_nbins - log_like_OFF_v_nbins

    # Select and re-order samples for display
    idx_to_plot = metadata_res.subplot.str.contains("d").values.nonzero()[0]
    nrows = metadata_res.shape[0]
    nplot = idx_to_plot.size

    _permute = np.arange(nplot)
    _permute[1:] += 6
    _permute[-6:] = np.arange(1, 7)
    idx_to_plot = idx_to_plot[_permute]
    
#    idx_to_plot = np.arange(metadata_res.shape[0])

    main(
        metadata_res,
        idx_to_plot,
        llrs_v_nbins,
        save=True,
        save_dir=os.path.abspath("../plots/tmp")
    )


## Initialize output
#log_like_OFF_v_nbins  = np.zeros((len(nbins_range), len(data))) 
#log_like_ON_v_nbins   = np.zeros((len(nbins_range), len(data))) 
#
#for i, _nbins in enumerate(tqdm(nbins_range)):
#
#    # Get number of observations in each bin
#    _data_hists = np.array([lsig.data_to_hist(d.values, _nbins)[0] for d in data])
#
#    # Add 1 to each bin to avoid div by 0. (regularization) 
#    _data_hists = _data_hists + 1
#    
#    # Normalize to PDF and take the logarithm
#    _data_hists_pdf    = _data_hists / np.sum(_data_hists, axis=1, keepdims=True)
#    _data_hists_logpdf = np.log10(_data_hists_pdf)
#
#    # Get reference log-PDFs
#    _OFF_hist_logpdf, _ON_hist_logpdf = _data_hists_logpdf[refdata_idx]
#
#    # Calculate the log likelihood of data given empirical distributions
#    log_like_OFF_v_nbins[i]  = np.sum(_data_hists *  _OFF_hist_logpdf, axis=1)
#    log_like_ON_v_nbins[i]   = np.sum(_data_hists *   _ON_hist_logpdf, axis=1)


