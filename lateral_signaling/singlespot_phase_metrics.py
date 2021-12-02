import os
from glob import glob
import json
import h5py

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

import lateral_signaling as lsig


# Paths to important places
data_dir     = os.path.abspath("../data/sim_data/20211201_singlespotphase/sacred")
save_dir     = os.path.abspath("../plots")
thresh_fpath = os.path.abspath("../data/sim_data/phase_threshold.json")

# Figure settings
save_figs = True
fig_fmt   = "png"
dpi       = 300


# Get threshold for v_init
with open(thresh_fpath, "r") as f:
    threshs = json.load(f)
    v_init_thresh = float(threshs["v_init_thresh"])


## Read in phase metric data
# Get directory for each run
run_dirs = glob(os.path.join(data_dir, "[0-9]*"))

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

# Assign phases and corresponding plot colors
df["phase"] = (df.v_init > v_init_thresh).astype(int) * (1 + (df.n_act_fin > 0).astype(int))
df["color"] = np.array(lsig.cols_blue)[df.phase]


## Visualize v_init distribution and cutoff
fig = plt.figure()

# Histogram
plt.hist(df.v_init, bins=50, color="k", density=True);

# Include threshold used
plt.vlines(v_init_thresh, *plt.gca().get_ylim(), color="k", linestyles="dashed")
plt.xlabel(r"$v_{init}$", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.tick_params(labelsize=12)
plt.tight_layout()

# Save
if save_figs:
    fname = "v_init_histogram"
    fpath = os.path.join(save_dir, fname + "." + fig_fmt)
    plt.savefig(fpath, dpi=dpi)

## Visualize n_act_fin distribution
fig = plt.figure()
gs  = fig.add_gridspec(1, 3)

# Calculate percent of runs where n_act_fin = 0
ax1 = fig.add_subplot(gs[0, 0])
pct_off = (df.n_act_fin == 0).sum() / df.shape[0]

# Include a bar-chart showing this percentage
plt.fill([-1, -1, 1, 1], [0,       1,       1, 0], lsig.col_gray)
plt.fill([-1, -1, 1, 1], [0, pct_off, pct_off, 0], "k"          )
plt.title(r"  $n_{\mathrm{act, fin}}$ = 0", loc="left",   fontsize=16, y=-0.075)
plt.title(f"{pct_off:.1%}\n",               loc="center", fontsize=14, y=pct_off-0.025)
plt.xlim(-1.75, 1.75)
plt.axis("off")

# Get non-zero n_act_fin values
ax2 = fig.add_subplot(gs[:, 1:])
n_act_fin = df.n_act_fin[df.n_act_fin > 0].values

# Histogram (log-scaled frequency)
nbins = 50
bins = np.geomspace(1, 6400, nbins + 1)
plt.hist(df.n_act_fin, bins=bins, color="k", density=True, log=True);
plt.xlabel(r"$n_{\mathrm{act, fin}}$", fontsize=16)
ax2.semilogx()
plt.ylabel("Frequency", fontsize=16)
plt.tick_params(labelsize=12)
plt.tight_layout()

if save_figs:
    fname = "n_act_fin_histogram"
    fpath = os.path.join(save_dir, fname + "." + fig_fmt)
    plt.savefig(fpath, dpi=dpi)
