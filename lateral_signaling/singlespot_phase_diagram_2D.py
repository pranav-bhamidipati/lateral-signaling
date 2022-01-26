import os
from glob import glob
import json
import h5py

import numpy as np
import pandas as pd
from tqdm import tqdm

import colorcet as cc
import cmocean.cm as cmo

import holoviews as hv
hv.extension("matplotlib")

import matplotlib.pyplot as plt

import lateral_signaling as lsig


# Reading
data_dir       = os.path.abspath("../data/simulations/")
sacred_dir     = os.path.join(data_dir, "20211209_phase_2D/sacred")
thresh_fpath   = os.path.join(data_dir, "phase_threshold.json")
examples_fpath = os.path.join(data_dir, "phase_examples.json")

# Writing
save_dir = os.path.abspath("../plots")
fpath_pfx = os.path.join(save_dir, "phase_diagram_2D_")

def main(
    pad=0.05,
    save=False,
    prefix=fpath_pfx,
    suffix="_",
    fmt="png",
    dpi=300,
):

    ## Read in and assemble data
    # Get phase example values
    with open(examples_fpath, "r") as f:
        ex_dict  = json.load(f)
        ex_name   = ex_dict["name"]
        ex_xval   = ex_dict["g"]
        ex_yval   = ex_dict["rho_0"]
        ex_xlabel = ex_dict["label_x"]
        ex_ylabel = ex_dict["label_y"]

    # Get threshold for v_init
    with open(thresh_fpath, "r") as f:
        threshs = json.load(f)
        v_init_thresh = float(threshs["v_init_thresh"])

    # Read in phase metric data
    run_dirs = glob(os.path.join(sacred_dir, "[0-9]*"))

    # Store each run's data in a DataFrame
    dfs = []
    for rd_idx, rd in enumerate(tqdm(run_dirs)):    
        
        _config_file = os.path.join(rd, "config.json")
        _results_file = os.path.join(rd, "results.hdf5")
        
        if (not os.path.exists(_config_file)) or (not os.path.exists(_results_file)):
            continue
        
        # Get some info from the run configuration
        with open(_config_file, "r") as c:
            config = json.load(c)
            
            # Initial density, carrying capacity
            rho_0   = config["rho_0"]
            rho_max = config["rho_max"]
            
        # Get remaining info from run's data dump
        with h5py.File(_results_file, "r") as f:
            
            # Phase metrics
            v_init    = np.asarray(f["v_init_g"])
            n_act_fin = np.asarray(f["n_act_fin_g"])
            
            # Proliferation rates and time-points
            if rd_idx == 0:
                g_space = config["g_space"]
                t = np.asarray(f["t"])
        
        # Assemble dataframe
        _df = pd.DataFrame(dict(
            v_init=v_init,
            n_act_fin=n_act_fin,
            g=g_space,
            rho_0=rho_0,
            rho_max=rho_max,
        ))
        dfs.append(_df)

    # Concatenate into one dataset
    df = pd.concat(dfs).reset_index(drop=True)
    df["g_inv_days"] = lsig.g_to_units(df["g"].values)
    
    # Assign phases and sort by phase
    df["phase"] = (df.v_init > v_init_thresh).astype(int) * (1 + (df.n_act_fin > 0).astype(int))
    df = df.sort_values("phase")

    # Extract data ranges
    g_space     = np.unique(df["g"])
    g_range     = g_space[-1] - g_space[0]
    rho_0_space = np.unique(df["rho_0"])
    rho_0_range = rho_0_space[-1] - rho_0_space[0]

    # Find phase examples in data
    df["example"] = ""
    for x, y, l in zip(ex_xval, ex_yval, ex_name):
        df.loc[
            np.isclose(df["g"], x) & np.isclose(df["rho_0"], y), 
            "example"
        ] = l

    ## Plot basic phase diagram
    # Set axis limits
    xlim = tuple([
        lsig.g_to_units(g_space[0]  - pad * g_range),
        lsig.g_to_units(g_space[-1] + pad * g_range),
    ])
    ylim = tuple([
        rho_0_space[0]  - pad * rho_0_range,
        rho_0_space[-1] + pad * rho_0_range,
    ])

    # Restrict to cases where rho_0 <= rho_max (net growth)
    df = df.loc[df["rho_0"] <= df["rho_max"], :]

    # Colors for phase regions
    phase_colors = lsig.cols_blue[::-1]

    # Options for different plot types
    plot_kw = dict(
        xlim = xlim,
        ylim = ylim,
        xlabel = r"proliferation rate ($days^{-1}$)",
        xticks = (0.5, 1.0, 1.5),
        ylabel = r"init. density (x 100% confl.)",
        yticks = (0, 1, 2, 3, 4, 5, 6),
        hooks=[lsig.remove_RT_spines],
        fontscale=1.,
        show_legend=False,
#        aspect=1.,
    )
    bare_kw = dict(
        marker="s",
        edgecolor=None,
        s=60,
        color=hv.Cycle(phase_colors),
    )
    example_kw = dict(
        marker="s",
        s=60,
        edgecolor="k",
        linewidth=1.5,
        color=hv.Cycle(phase_colors),
    )
    text_init_kw = dict(
        fontsize=11, 
        halign="left",
    )
    text_kw = dict(
        c="w",
        weight="normal",
    )

    # Make bare phase diagram
    phasediagram_bare = hv.Scatter(
        data=df,
        kdims=["g_inv_days"],
        vdims=["rho_0", "phase"],
    ).groupby(
        "phase"
    ).opts(
        **bare_kw,
        **plot_kw,
    ).overlay()
    
    # Phase diagram with examples
    examples = hv.Scatter(
        data=df.loc[df["example"].str.len() > 0],
        kdims=["g_inv_days"],
        vdims=["rho_0", "phase"],
    ).groupby(
        "phase"
    ).opts(
        **example_kw
    ).overlay()
    
    phasediagram_ex = (phasediagram_bare * examples).opts(**plot_kw)

    # Phase labels
    text_labels = [
        hv.Text(x, y, l, **text_init_kw) 
        for x, y, l in zip(ex_xlabel, ex_ylabel, ex_name)
    ]

    # Labeled phase diagram
    phasediagram_labeled = hv.Overlay(
        [phasediagram_bare, examples, *text_labels]
    ).opts(
        hv.opts.Text(**text_kw),
        hv.opts.Overlay(**plot_kw),
    )

    if save:
        
        fpath = prefix + "bare" + suffix
        _fpath = fpath + "." + fmt
        print(f"Writing to: {_fpath}")
        hv.save(phasediagram_bare, fpath, fmt=fmt, dpi=dpi)

        fpath = prefix + "examples" + suffix
        _fpath = fpath + "." + fmt
        print(f"Writing to: {_fpath}")
        hv.save(phasediagram_ex, fpath, fmt=fmt, dpi=dpi)

        fpath = prefix + "labeled" + suffix
        _fpath = fpath + "." + fmt
        print(f"Writing to: {_fpath}")
        hv.save(phasediagram_labeled, fpath, fmt=fmt, dpi=dpi)


main(
    save=True,
    suffix="_",
)
0/0


# ## Plot time-series of examples

# <hr>

# In[70]:


# Make data
example_data_list = [
    dict(
        t=t_sample,
        example=[i] * t_sample.size,
        actnum=actnum_example_sample[i],
        actrad=actrad_example_sample[i],
        actarea=actarea_example_sample[i],
    )
    for i, _ in enumerate(examples_idx)
]


# In[34]:


# Plot
example_plots = [
    hv.Curve(
        example_data_list[i],
        kdims=["t"],
        vdims=["actarea"],
    ).opts(
        xlabel=("", "", "time")[i],
        xticks=0,
#         ylabel=("", "radius (cell diameters)", "")[i],
#         yticks=[0, 25],
#         ylim=(-1, 28),
        linewidth=10,
        fontscale=3,
        color=phase_colors[examples_idx][i],
    )
    for i, _ in enumerate(examples_idx)
]


# In[71]:


examples_overlay = hv.Layout(example_plots).opts(vspace=0.4, sublabel_size=0).cols(1)
examples_overlay


# In[62]:


nt_t = np.searchsorted(lsig.t_to_units(t), [1.])[0]


# In[67]:


examples_max_idx = S_actnum_mean[examples_idx].argmax(axis=1)
examples_max_idx[0] = int(nt_t * 1.5)

examples_max_idx


# In[68]:


lsig.t_to_units(t)[examples_max_idx], S_actnum_mean[examples_max_idx]


# In[36]:


example_max_points = hv.Scatter(
    (lsig.t_to_units(t)[examples_max_idx], S_actnum_mean[examples_max_idx]),
).opts(
)

example_plots_horizontal = [
    hv.Curve(
        example_data_list[i],
        kdims=["t"],
        vdims=["actrad"],
    ).opts(
        xlabel=("", "days", "")[i],
        xlim=(0, 8),
        xticks=(0, 8),
        ylabel=("radius (cell diam.)", "", "")[i],
        yticks=[0, 25],
        ylim=(-1, 28),
        linewidth=10,
        fontscale=3,
        color=phase_colors[examples_idx][i],
    )
    for i, _ in enumerate(examples_idx)
]

examples_overlay_horizontal = hv.Layout(example_plots_horizontal).opts(hspace=0.25, sublabel_size=0).cols(3)


# In[37]:


examples_overlay_horizontal


# ## Save

# In[38]:


fname = "phasediagram_examples_radius_timeseries_horizontal"

fpath = os.path.abspath(os.path.join(save_dir, fname + "." + fig_fmt))

if save_figs:
    hv.save(examples_overlay_horizontal, fpath, dpi=dpi)


# <hr>

# # Plot stimulated area

# In[134]:


area_ceiling = 1e5
S_A_max_trunc = np.minimum(S_A_max, area_ceiling)


# In[135]:


stim_mask = np.logical_and(S_A_max > 0.1 * area_ceiling, S_A_max < area_ceiling)

# stimulated_points_outline3 = hv.Scatter(
#     (
#         lsig.g_to_units(param_space_agg[stim_mask, 0]), 
#         param_space_agg[stim_mask, 1],
#     ),
# ).opts(
#     marker = "s",
#     edgecolor = "k",
#     linewidth=2,
#     s=70,
# )

# stimulated_plot = hv.Scatter(
#     (
#         lsig.g_to_units(param_space_agg[stim_mask, 0]), 
#         param_space_agg[stim_mask, 1],
#     ),
# ).opts(
#     xlim = xlim,
#     ylim = (ylim[0] * 18/25, ylim[1] * 18/25),
#     aspect = 25/18,
#     xlabel = r"proliferation rate ($days^{-1}$)",
#     xticks = (0.5, 1.0, 1.5),
#     ylabel = r"init. density (x 100% confl.)",
#     yticks = (0, 1, 2, 3, 4, 5),
#     marker = "s",
# #     edgecolor = "w",
#     s=65,
# #     c="w",
#     c="#e5e5e5",
# #     c=stimulated_pct[mask],
# #     cmap=cmo.algae,
#     colorbar=True,
# #     logx=True, 
#     fontscale=1.,
# )

stimulated_points_bg = hv.Scatter(
    (
        lsig.g_to_units(param_space_agg[mask, 0]), 
        param_space_agg[mask, 1],
    ),
).opts(
    marker = "s",
    edgecolor = (0,0,0,0),
    c=lsig.hexa2hex(phase_colors[mask], alpha=0.8),
    linewidth=0,
    s=65,
)

stimulated_var_points = hv.Scatter(
    (
        lsig.g_to_units(param_space_agg[mask, 0]), 
        param_space_agg[mask, 1],
    ),
).opts(
    marker=".",
#     s=130 * (R_actnum_mean.max(axis=1) > 0)[mask],
    s=130 * S_A_max_trunc[mask] / area_ceiling,
#     c="k",
    c="w",
#     c=phase_colors[mask], 
)

# stimulated_sat_points = hv.Scatter(
#     (
#         lsig.g_to_units(param_space_agg[phase==0, 0]), 
#         param_space_agg[phase==0, 1],
#     ),
# ).opts(
#     marker="s",
#     s=70,
#     c="k",
# )

points_scale_bg_xvals = np.array([
    [*(1.9 * np.ones(21)), *(1.95 * np.ones(21))],
    [*np.linspace(0.25, 5.0, 21), *np.linspace(0.25, 5.0, 21)]
])

var_points_scale_bg = hv.Scatter(
    points_scale_bg_xvals.T
).opts(
    marker = "s",
    edgecolor = (0,0,0,0),
    c=lsig.col_gray,
    linewidth=0,
    s=65,
)


var_points_scale = hv.Scatter(
    (1.925 * np.ones(11), np.linspace(0.375, 4.875, 11))
).opts(
    marker=".",
    s=130 * np.arange(1, -0.01, -0.1),
    c="w",
#     aspect=0.1
)

x_stretch = (2 - xlim[0]) / (xlim[1] - xlim[0])

stimulated_overlay = (
#     stimulated_points_outline * stimulated_plot * stimulated_var_points
    stimulated_points_bg * stimulated_var_points * var_points_scale_bg * var_points_scale
).opts(
#     xlim = (xlim[0], 2),
#     ylim = (ylim[0] * 18/25, ylim[1] * 18/25),
#     aspect = 25/18 * x_stretch,
    xlim = (xlim[0], 2),
    ylim = ylim,
    aspect = x_stretch,
#     xlim = xlim,
#     ylim = ylim,
    xlabel = r"proliferation rate ($days^{-1}$)",
    xticks = (0.5, 1.0, 1.5),
    ylabel = r"init. density (x 100% confl.)",
    yticks = (0, 1, 2, 3, 4, 5, 6),
)


# In[136]:


hv.output(stimulated_overlay, dpi=dpi//2)


# ## Save

# In[137]:


if save_figs:
    fname = "phasediagram_stimulated_area"
    
    fpath = os.path.abspath(os.path.join(save_dir, fname + "." + fig_fmt))
    hv.save(stimulated_overlay, fpath, dpi=dpi)


# <hr>

# In[76]:


np.array([
    [S_A_max_trunc[phase == 0].min(), S_A_max_trunc[phase == 0].max()], # OFF
    [S_A_max_trunc[phase == 1].min(), S_A_max_trunc[phase == 1].max()], # ON-OFF
    [S_A_max_trunc[phase == 2].min(), S_A_max_trunc[phase == 2].max()], # ON
]) / 1e4


# In[77]:


x_stretch = (2 - xlim[0]) / (xlim[1] - xlim[0])

stimulated_plot2 = hv.Scatter(
    (
        lsig.g_to_units(param_space_agg[phase == 1, 0]), 
        param_space_agg[phase == 1, 1],
    ),
).opts(
    xlim = (xlim[0], 2),
    ylim = (ylim[0] * 18/25, ylim[1] * 18/25),
    aspect = 25/18 * x_stretch,
    xlabel = r"proliferation rate ($days^{-1}$)",
    xticks = (0.5, 1.0, 1.5),
    ylabel = r"init. density (x 100% confl.)",
    yticks = (0, 1, 2, 3, 4, 5),
    marker = "s",
#     edgecolor = "w",
    s=70,
    c="w",
#     c=stimulated_pct[mask],
#     cmap=cmo.algae,
#     colorbar=True,
#     logx=True, 
    fontscale=1.,
)

stimulated_points_outline2 = hv.Scatter(
    (
        lsig.g_to_units(param_space_agg[phase == 1, 0]), 
        param_space_agg[phase == 1, 1],
    ),
).opts(
    marker = "s",
    edgecolor = "k",
    linewidth=2,
    s=70,
)

stimulated_var_points2 = hv.Scatter(
    (
        lsig.g_to_units(param_space_agg[phase == 1, 0]), 
        param_space_agg[phase == 1, 1],
    ),
).opts(
    marker=".",
    s=130 * S_A_max_trunc[phase == 1] / area_ceiling,
    c="k",
)

var_points_scale = hv.Scatter(
    (1.95 * np.ones(11), np.linspace(0.25, 4.25, 11))
).opts(
    marker=".",
    s=130 * np.arange(1, -0.01, -0.1),
    c="k",
#     aspect=0.1
)

stimulated_overlay2 = stimulated_points_outline2 * stimulated_plot2 * stimulated_var_points2 * var_points_scale


# In[78]:


hv.output(stimulated_overlay2, dpi=dpi//2)


# ## Save

# In[79]:


fname = "phasediagram_stimulated_area_v2"

fpath = os.path.abspath(os.path.join(save_dir, fname + "." + fig_fmt))

if save_figs:
    hv.save(stimulated_overlay2, fpath, dpi=dpi)


# __Make version just based on percent stimulation__

# In[80]:


x_stretch = (2 - xlim[0]) / (xlim[1] - xlim[0])

stim_mask = np.logical_and((S_A_max_trunc / area_ceiling) > 0.025, (S_A_max_trunc / area_ceiling) < 0.975)

stimulated_plot3 = hv.Scatter(
    (
        lsig.g_to_units(param_space_agg[stim_mask, 0]), 
        param_space_agg[stim_mask, 1],
    ),
).opts(
    xlim = (xlim[0], 2),
    ylim = (ylim[0] * 18/25, ylim[1] * 18/25),
    aspect = 25/18 * x_stretch,
    xlabel = r"proliferation rate ($days^{-1}$)",
    xticks = (0.5, 1.0, 1.5),
    ylabel = r"init. density (x 100% confl.)",
    yticks = (0, 1, 2, 3, 4, 5),
    marker = "s",
#     edgecolor = "w",
    s=70,
    c="w",
#     c=stimulated_pct[mask],
#     cmap=cmo.algae,
#     colorbar=True,
#     logx=True, 
    fontscale=1.,
)

stimulated_points_outline3 = hv.Scatter(
    (
        lsig.g_to_units(param_space_agg[stim_mask, 0]), 
        param_space_agg[stim_mask, 1],
    ),
).opts(
    marker = "s",
    edgecolor = "k",
    linewidth=2,
    s=70,
)

stimulated_var_points3 = hv.Scatter(
    (
        lsig.g_to_units(param_space_agg[stim_mask, 0]), 
        param_space_agg[stim_mask, 1],
    ),
).opts(
    marker=".",
    s=130 * (np.minimum(S_A_max, 2e5) / 2e5)[stim_mask],
    c="k",
)


stimulated_overlay3 = stimulated_points_outline3 * stimulated_plot3 * stimulated_var_points3 * var_points_scale


# In[81]:


hv.output(stimulated_overlay3, dpi=dpi//2)


# ## Save

# In[82]:


if save_figs:
    fname = "phasediagram_percent_stimulated_version3"
    
    fpath = os.path.abspath(os.path.join(save_dir, fname + "." + fig_fmt))
    hv.save(stimulated_overlay3, fpath, dpi=dpi)


# <hr>

# ## Add perturbed conditions to phase diagram

# In[83]:


phasediagram_bare = phasediagram.options(dict(
    Points=dict(edgecolor=None, s=65)
)).opts(
    xaxis=None,
    yaxis=None,
)


# In[84]:


hv.output(phasediagram_bare * hv.HLine(2.75), dpi=dpi//2)


# ## Save

# In[62]:


if save_figs:
    fname = "phasediagram_bare"
    
    fpath = os.path.abspath(os.path.join(save_dir, fname + "." + fig_fmt))
    hv.save(phasediagram_bare, fpath, dpi=dpi)


# In[57]:


dens_points = hv.Points(
    [
        (mle_params_df.g_inv_days.values[2], 1.),
        (mle_params_df.g_inv_days.values[2], 2.),
        (mle_params_df.g_inv_days.values[2], 4.),
#         (mle_params_df.g_inv_days.values[0], 1.),
#         (mle_params_df.g_inv_days.values[1], 1.),
    ]
).opts(
    marker="^",
    c=[
        *lsig.yob[1:], 
#         lsig.purple, 
#         lsig.greens[3]
    ],
    s=250,
    ec=lsig.col_black,
)

dens_bare_overlay = phasediagram_bare * dens_points


# In[58]:


hv.output(dens_bare_overlay, dpi=dpi//2)


# ## Save

# In[98]:


fname = "phasediagram_with_densities"

fpath = os.path.abspath(os.path.join(save_dir, fname + "." + fig_fmt))

if save_figs:
    hv.save(dens_bare_overlay, fpath, dpi=dpi)


# In[60]:


drug_points = hv.Points(
    [
        (mle_params_df.g_inv_days.values[2], 1.),
#         (mle_params_df.g_inv_days.values[2], 2.),
#         (mle_params_df.g_inv_days.values[2], 4.),
        (mle_params_df.g_inv_days.values[0], 1.),
        (mle_params_df.g_inv_days.values[1], 1.),
    ]
).opts(
    marker="^",
    c=[
        lsig.yob[1],
        lsig.purple, 
        lsig.greens[3]
    ],
    s=250,
    ec=lsig.col_black,
)

drug_bare_overlay = phasediagram_bare * drug_points


# In[61]:


hv.output(drug_bare_overlay, dpi=dpi//2)


# ## Save

# In[99]:


fname = "phasediagram_with_drugs"

fpath = os.path.abspath(os.path.join(save_dir, fname + "." + fig_fmt))

if save_figs:
    hv.save(drug_bare_overlay, fpath, dpi=dpi)


# In[63]:


dens_drug_points = hv.Points(
    [
        (mle_params_df.g_inv_days.values[2], 1.),
        (mle_params_df.g_inv_days.values[2], 2.),
        (mle_params_df.g_inv_days.values[2], 4.),
        (mle_params_df.g_inv_days.values[0], 1.),
        (mle_params_df.g_inv_days.values[1], 1.),
    ]
).opts(
    marker="^",
    c=[*lsig.yob[1:], lsig.purple, lsig.greens[3]],
    s=250,
    ec=lsig.col_black,
)

dens_drug_bare_overlay = phasediagram_bare * dens_drug_points


# In[64]:


hv.output(dens_drug_bare_overlay, dpi=dpi//2)


# ## Save

# In[100]:


fname = "phasediagram_with_densities_and_drugs"

fpath = os.path.abspath(os.path.join(save_dir, fname + "." + fig_fmt))

if save_figs:
    hv.save(dens_drug_bare_overlay, fpath, dpi=dpi)

