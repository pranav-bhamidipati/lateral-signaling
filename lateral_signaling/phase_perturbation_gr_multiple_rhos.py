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

#sim_dir    = os.path.abspath("../data/simulations/20220204_phase_perturbations_growthrate/sacred")
sim_dir    = os.path.abspath("../data/simulations/20220802_phase_perturbations/sacred")
data_fname = os.path.abspath("../data/whole_wells/drug_conditions_propagation_and_cellcount.csv")

save_dir   = os.path.abspath("../plots")
save_pfx   = os.path.join(save_dir, "phase_perturbations__")

def main(
    sim_dir=sim_dir,
    data_fname=data_fname,
    colors=lsig.growthrate_colors,
    pad=0.05,
    sample_every=1,
    save=False,
    prefix=save_pfx,
    suffix="",
    fmt="png",
    dpi=300,
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
    
    # axis ticks
    yticks = (0, 0.2, 0.4, 0.6, 0.8, 1.0)
    xticks = (1, 2, 3, 4, 5, 6, 7, 8)
    
    # Set colors/linestyles/markers
    ccycle = hv.Cycle(colors)
    
    # Other options
    opts = dict(
        xlabel="Days",
        xlim = (xmin - pad * (xmax - xmin), xmax + pad * (xmax - xmin)),
        xticks=xticks,
        ylabel="% of area",
        ylim = (ymin - pad * (ymax - ymin), ymax + pad * (ymax - ymin)),
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
    
    ## Read simulated data
    run_dirs    = glob(os.path.join(sim_dir, "[0-9]*"))
    
    # Initialize lists to store data
    gs              = []
    dconds          = []
    rho_0s          = []
    rho_ts          = []
    S_t_reps        = []
    sender_idx_reps = []

    for i, run_dir in enumerate(run_dirs):
        config_file  = os.path.join(run_dir, "config.json")
        results_file = os.path.join(run_dir, "results.hdf5")
        
        with open(config_file, "r") as c:
            config = json.load(c)
            
            # Drug condition
            dcond   = config["drug_condition"]

            k       = config["k"]
            g       = config["g"]
            rho_0   = config["rho_0"]
            rho_max = config["rho_max"]
         
        with h5py.File(results_file, "r") as f:
            
            # Restrict time-range to match in vitro
            t      = np.asarray(f["t"])
            t_days = lsig.t_to_units(t)
            tmask  = t_days <= tmax_days
            t      = t[tmask]
            t_days = t_days[tmask]
            nt     = t.size

            # Re-scale time axis
            opts["xlim"] = t_days[0], opts["xlim"][1]
            opts["xticks"] = tuple([0] + list(opts["xticks"]))

            # Index of sender cell
            n_senders = int(np.asarray(f["n_senders"]))

            # Density vs. time
            rho_t = np.asarray(f["rho_t"])[tmask]

            # Signal and reporter expression vs. time
            S_t_rep = np.asarray(f["S_t_rep"])[:, tmask]
            n = S_t_rep.shape[-1]
            
            # Get senders
            sender_idx_rep = np.asarray(f["sender_idx_rep"])

            # Store data
            gs.append(g)
            dconds.append(dcond)
            rho_0s.append(rho_0)
            rho_ts.append(rho_t)
            S_t_reps.append(S_t_rep)
            sender_idx_reps.append(sender_idx_rep) 
    
    # Convert to arrays
    rho_ts          = np.asarray(rho_ts)
    S_t_reps        = np.asarray(S_t_reps)
    sender_idx_reps = np.asarray(sender_idx_reps)
    
    # Get number of conditions
    n_runs = len(dconds)
    
    # Calculate the avg number of activated transceivers across replicates
    n_act_t_reps = (S_t_reps > k).sum(axis=-1) - n_senders
    n_act_ts = n_act_t_reps.mean(axis=1)

    # Percent activated transceivers
    pctArea_ts = n_act_ts / (n - n_senders)
    
    # (Normalized) Area and sqrt(Area) of activation
    A_ts = np.array([lsig.ncells_to_area(nc, rt) for nc, rt in zip(n_act_ts, rho_ts)])
    normA_ts = A_ts / A_ts.max()
    sqrtA_ts = np.sqrt(A_ts)
    
    ## Make plot
    # Make dataframe for plotting
    insilico_data = {
        "condition" : np.repeat(dconds, t_days[::sample_every].size), 
        r"$\rho_0$" : np.repeat(rho_0s, t_days[::sample_every].size), 
        "days"      : np.tile(t_days[::sample_every], n_runs),
        "pctArea"   : pctArea_ts[:, ::sample_every].flatten(),
        "Area_norm" : normA_ts[:, ::sample_every].flatten(),
        "Area_mm"   : A_ts[:, ::sample_every].flatten(),
        "sqrtA_mm"  : sqrtA_ts[:, ::sample_every].flatten(),
    } 

    # Plot curves
    insilico_plots = hv.Curve(
        insilico_data,
        kdims=["days"],
        vdims=["pctArea", "condition", r"$\rho_0$"],
    ).groupby(
        ["condition", r"$\rho_0$"]
    ).opts(
        linewidth=3,
        color=ccycle,
    ).overlay(
        "condition"
    ).opts(
        show_legend=False, 
        **opts
    ).layout(
        r"$\rho_0$"
    )
    
    if save:

        # Print update and save
        fpath = prefix + "insilico_" + suffix
        _fpath = fpath + "." + fmt
        print(f"Writing to: {_fpath}")
        hv.save(insilico_plots, fpath, fmt=fmt, dpi=dpi)
    

main(
#    save=True,
    pad=0.05,
    sample_every=5,
)
