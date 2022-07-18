import os
from glob import glob
import json
import h5py

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as rotation
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

import lateral_signaling as lsig
from itertools import islice


# Reading
data_dir     = os.path.abspath("../data/simulations")
sacred_dir   = os.path.join(data_dir, "20211201_singlespotphase/sacred")
thresh_fpath = os.path.join(data_dir, "phase_threshold.json")

# Writing
save_dir        = os.path.abspath("../plots/tmp")
v_init_fname    = os.path.join(save_dir, "v_init_histogram")
n_act_fin_fname = os.path.join(save_dir, "n_act_fin_histogram")
phase3d_fname   = os.path.join(save_dir, "phase_boundaries_3D")

def main(
    figsize=(4, 3),
    nbins=30,
    binmin=1,
    binmax=6400,
    save=False,
    fmt="png",
    dpi=300,
):


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
            rho_0  = config["rho_0"]
            rho_max  = config["rho_max"]
            
        # Get remaining info from run's data dump
        with h5py.File(_results_file, "r") as f:
            
            # Phase metrics
            v_init    = np.asarray(f["v_init_g"])
            n_act_fin = np.asarray(f["n_act_fin_g"])
            
            # Proliferation rates and time-points
            if rd_idx == 0:
                g = list(f["g_space"])
                t = np.asarray(f["t"])
        
        # Assemble dataframe
        _df = pd.DataFrame(dict(
            v_init=v_init,
            n_act_fin=n_act_fin,
            g=g,
            rho_0=rho_0,
            rho_max=rho_max,
        ))
        dfs.append(_df)

    # Concatenate into one dataset
    df = pd.concat(dfs).reset_index(drop=True)
    nrow = df.shape[0]

    # Assign phases and corresponding plot colors
    df["phase"] = (df.v_init > v_init_thresh).astype(int) * (1 + (df.n_act_fin > 0).astype(int))
    df["color"] = np.array(lsig.cols_blue)[df.phase]

    ## Plot histogram of v_init
    fig, ax = plt.subplots(figsize=figsize)
    plt.hist(df.v_init, bins=nbins, color="k");
#    plt.hist(df.v_init, bins=nbins, color="k", density=True);

    # Threshold used
    plt.vlines(v_init_thresh, *plt.gca().get_ylim(), color="k", linestyles="dashed")

    plt.xlabel(r"$v_{init}$", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.tick_params(labelsize=12)

    plt.tight_layout()

    if save:
        
        _fpath = str(v_init_fname)
        if not _fpath.endswith(fmt):
            _fpath += "." + fmt
        print("Writing to:", _fpath)
        plt.savefig(_fpath, dpi=dpi)


    # Get pct inactivated cells at end of simulation
    pct_off = (df.n_act_fin == 0).sum() / df.shape[0]

    ## Plot n_act distribution
    fig = plt.figure()
    gs  = fig.add_gridspec(1, 3)

    # Plot bar chart with percentage of runs with any active cells at the end
    ax1 = fig.add_subplot(gs[0, 0])
    plt.fill([-1, -1, 1, 1], [0,       1,       1, 0], lsig.col_gray)
    plt.fill([-1, -1, 1, 1], [0, pct_off, pct_off, 0],           "k")

    # Add labels and options
    plt.title(r"  $n_{\mathrm{act, fin}}$ = 0", loc="left",   fontsize=16, y=-0.075)
    plt.title(f"{pct_off:.1%}\n",               loc="center", fontsize=14, y=pct_off-0.05)
    plt.xlim(-2, 2)
    plt.axis("off")

    # Plot histogram on linear-linear axes 
    ax2 = fig.add_subplot(gs[:, 1:])
    bins = np.linspace(binmin, binmax, nbins + 1)
    plt.hist(df.n_act_fin, bins=bins, color="k");
    plt.ylabel("# simulations", fontsize=16)
   
#    # Plot histogram on log-log axes
#    ax2 = fig.add_subplot(gs[:, 1:])
#    bins = np.geomspace(binmin, binmax, nbins + 1)
#    plt.hist(df.n_act_fin, bins=bins, color="k", density=True, log=True);
#    ax2.semilogx()
#    plt.ylabel("Frequency", fontsize=16)

    # Add labels and options
    plt.xlabel(r"$n_{\mathrm{act, fin}}$", fontsize=16)
    plt.tick_params(labelsize=12)

    plt.tight_layout()

    if save:
        
        _fpath = str(n_act_fin_fname)
        if not _fpath.endswith(fmt):
            _fpath += "." + fmt
        print("Writing to:", _fpath)
        plt.savefig(_fpath, dpi=dpi)


main(
    save=True,
)


