import os

import pandas as pd
import numpy as np

import colorcet as cc
import holoviews as hv
hv.extension("matplotlib")

import matplotlib.pyplot as plt

import lateral_signaling as lsig


# Options/files/directories for reading
data_dir           = os.path.abspath("../data/FACS_data")
metadata_res_fname = os.path.join(data_dir, "metadata_with_LLR.csv")

# Options/files/directories for writing
save_dir  = os.path.abspath("../plots")
save_figs = True
fmt       = "png"
dpi       = 300

# Read in metadata
metadata_res = pd.read_csv(metadata_res_fname)

# Get which samples to include in the plots
idx_to_plot = metadata_res.subplot.str.contains("d").values.nonzero()[0]
nrows = metadata_res.shape[0]
nplot = idx_to_plot.size

# Re-order samples for display
_permute = np.arange(nplot)
_permute[1:] += 6
_permute[-6:] = np.arange(1, 7)
idx_to_plot = idx_to_plot[_permute]

# Get data 
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
if save_figs:
    fname = "FACS_likelihood_spikeplot"
    fpath = os.path.join(save_dir, fname + "." + fmt)
    print("Saving:", fpath)
    hv.save(llr_spikeplot, fpath, dpi=dpi)


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

if save_figs:
    fname = "FACS_likelihood_with_errorbars"
    fpath = os.path.join(save_dir, fname + "." + fmt)
    print("Saving:", fpath)
    hv.save(errplot, fpath, fmt=fmt, dpi=dpi)