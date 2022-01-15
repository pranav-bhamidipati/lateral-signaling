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

sim_dir    = os.path.abspath("../data/simulations/20220114_phase_perturbations/sacred")
data_fname = os.path.abspath("../data/whole_wells/drug_conditions_propagation_and_cellcount.csv")

save_dir   = os.path.abspath("../plots")
save_pfx   = os.path.join(save_dir, "growthrate_perturbation_normarea_")
fmt        = "png"
dpi        = 300

def main(
    sim_dir=sim_dir,
    data_fname=data_fname,
    pad=0.05,
    sample_every=1,
    save=False,
    prefix=save_pfx,
    suffix="",
    fmt=fmt,
    dpi=dpi,
):
    
    ## Read invitro data from file
    data = pd.read_csv(data_fname).dropna()

    # Get last time point
    tmax_days = data.Day.max()

    # Get conditions
    conds = data.Condition.unique()

    # Get normalized area
    data["Area_norm"] = data["GFP_um2"] / data["GFP_um2"].max()

#    # Convert area units and take sqrt
#    data["Area_mm2"]  = data["GFP_um2"] / 1e6
#    del data["GFP_um2"]
#    data["sqrtA_mm"] = np.sqrt(data["Area_mm2"])

    # axis limits with padding
    xmin = 1.0
    xmax = tmax_days
    ymin = 0.0
    ymax = 1.0
    xlim = xmin - pad * (xmax - xmin), xmax + pad * (xmax - xmin)
    ylim = ymin - pad * (ymax - ymin), ymax + pad * (ymax - ymin)
    
    # axis ticks
    yticks = (0, 0.2, 0.4, 0.6, 0.8, 1.0)
    xticks = (1, 2, 3, 4, 5, 6, 7, 8)
    
    # Set colors/linestyles/markers
    ccycle = hv.Cycle([lsig.purple, lsig.greens[3], lsig.yob[1]])
    
    # Other options
    opts = dict(
        xlabel="Days",
        xlim=xlim,
        xticks=xticks,
        ylabel="Area (norm.)",
        ylim=ylim,
        yticks=yticks,
        aspect=1.3,
        fontscale=1.3,
#        show_legend=False,
        legend_position="right",
    )

    ## Plot in vitro data
    invitro_curves = hv.Curve(
        data,
        kdims=["Day"],
        vdims=["Area_norm", "Condition"],
    ).groupby(
        "Condition"
    ).opts(
        c=ccycle,
        linewidth=2,
    ).overlay()
    
    invitro_points = hv.Scatter(
        data,
        kdims=["Day"],
        vdims=["Area_norm", "Condition"],
    ).groupby(
        "Condition"
    ).opts(
        c=ccycle,
        s=40,
    ).overlay()

    invitro_plot = hv.Overlay(
        [invitro_curves, invitro_points]
    ).opts(**opts)
    
    if save:

        # Print update and save
        fpath =  prefix + "invitro_" + suffix
        _fpath = fpath + "." + fmt
        print(f"Writing to: {_fpath}")
        hv.save(invitro_plot, fpath, fmt=fmt, dpi=dpi)
    
    return
    0/0

    ## Read simulated data
    run_dirs    = glob(os.path.join(sim_dir, "[0-9]*"))
    
    # Initialize lists to store data
    rho_0s = []
    rho_ts = []
    S_ts   = []

    for i, run_dir in enumerate(run_dirs):
        config_file  = os.path.join(run_dir, "config.json")
        results_file = os.path.join(run_dir, "results.hdf5")
        
        with open(config_file, "r") as c:
            config = json.load(c)

            # Check if run matches conditions
            if config["drug_condition"] != "untreated":
                continue

            # Expression threshold
            k = config["k"]

            # Initial density
            rho_0 = config["rho_0"]
         
        with h5py.File(results_file, "r") as f:
            
            # Restrict time-range to match in vitro
            t      = np.asarray(f["t"])
            t_days = lsig.t_to_units(t)
            tmask  = t_days <= tmax_days
            t      = t[tmask]
            t_days = t_days[tmask]
            nt     = t.size

            # Index of sender cell
            sender_idx = np.asarray(f["sender_idx"])

            # Density vs. time
            rho_t = np.asarray(f["rho_t"])[tmask]

            # Signal and reporter expression vs. time
            S_t = np.asarray(f["S_t"])[tmask]
            
            # Store data
            rho_0s.append(rho_0)
            rho_ts.append(rho_t)
            S_ts.append(S_t)
    
    # Convert to arrays
    rho_ts = np.asarray(rho_ts)
    S_ts   = np.asarray(S_ts)
    
    # Get initial densities as strings
    n_runs = len(rho_0s)
    rho_0_strs = [f"{int(r)}x" for r in rho_0s]
    
    # Calculate the number of activated transceivers
    n_act_ts = (S_ts > k).sum(axis=-1) - 1

    # Area and sqrt(Area) of activation
    A_ts = np.array([lsig.ncells_to_area(nc, rt) for nc, rt in zip(n_act_ts, rho_ts)])
    sqrtA_ts = np.sqrt(A_ts)
    
    ## Make plot
    # Make dataframe for plotting
    insilico_data = {
        "density"  : np.repeat(rho_0_strs, t_days[::sample_every].size), 
        "days"     : np.tile(t_days[::sample_every], n_runs),
        "Area_mm"  : A_ts[:, ::sample_every].flatten(),
        "sqrtA_mm" : sqrtA_ts[:, ::sample_every].flatten(),
    } 

    # Plot curves
    insilico_plot = hv.Curve(
        insilico_data,
        kdims=["days"],
        vdims=["sqrtA_mm", "density"],
    ).groupby(
        "density",
    ).opts(
        linewidth=3,
        color=ccycle,
    ).overlay(
        "density"
    ).opts(**opts)
    
    if save:

        # Print update and save
        fpath = prefix + "insilico_" + suffix
        _fpath = fpath + "." + fmt
        print(f"Writing to: {_fpath}")
        hv.save(insilico_plot, fpath, fmt=fmt, dpi=dpi)
    

main(
    save=True,
    pad=0.05,
    sample_every=5,
)
