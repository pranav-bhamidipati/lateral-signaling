#!/usr/bin/env python
# coding: utf-8


####################################
######## Set up environment ########
####################################

import lateral_signaling as lsig
import numpy as np
import pandas as pd
from tqdm import tqdm
import numba

import scipy.stats as st
from scipy.sparse import csr_matrix, diags

import os
from glob import glob
import datetime


####################################
######## Set global variables ######
####################################

# File I/O
experiment_name = "sweep_singlespotphase_dense"

# Seed for RNG
seed = 2021


####################################
######## Set simulation params #####
####################################

## Parameter values to sample
## Dense sampling
rep_space     = np.arange(5)
g_space       = np.linspace(0, 2.4, 25)[1:]
rho_0_space   = np.linspace(0, 6, 25)[1:]
# rho_max_space = np.linspace(0, 6, 25)[1:]

## Sparse sampling
# rep_space     = np.arange(2)
# g_space       = np.linspace(0, 2.4, 5)[1:]
# rho_0_space   = np.linspace(0, 6, 5)[1:]
# # rho_max_space = np.linspace(0, 6, 5)[1:]

# Select parameters to scan
free_params = (
    rep_space, 
    g_space, 
    rho_0_space, 
#     rho_max_space,
)

free_param_names = (
    "rep", 
    "g", 
    "rho_0", 
#     "rho_max",
)


####################################
######## Get/make directories ######
####################################

# Paths to data files
data_dir = os.path.abspath("../data/simulations")
mle_params_fname = "growth_parameters_MLE.csv"
mle_params_path = os.path.join(data_dir, mle_params_fname)

# Make results folder for experiment
experiment_dir = "_".join([
    str(datetime.date.today()), experiment_name
])
experiment_dir_path = os.path.join(data_dir, experiment_dir) 
os.makedirs(experiment_dir_path, exist_ok=False)


#######################################
######## Set up parameter sweep #######
#######################################

# Make array with all combinations of params
param_space = np.meshgrid(*free_params)
param_space = np.array(param_space).T.reshape(-1, len(free_params))

# Get metadata on parameter sampling space
n_runs = param_space.shape[0]         # total # parameter sets to run
n_reps = rep_space.size               # number of replicates
n_param_sets = int(n_runs / n_reps)   # number of unique param sets

# Get parameters for untreated condition (in dimensionless units)
mle_params_df = pd.read_csv(mle_params_file, index_col=0)
g, rho_max = mle_params_df.loc[
    mle_params_df.condition == "untreated", ["g_ratio", "rho_max_ratio"]
].values.ravel()


#######################################
######## Set up DDE params ############
#######################################

# Read trial parameters from CSV
params_df = pd.read_csv(os.path.join(data_dir, "sim_parameters.csv"))

# Get arguments for dampening function (beta), delay, and interaction radius
beta_args = params_df.loc[params_df.parameter.str.startswith("beta_")].value.values
delay     = params_df.loc[params_df.parameter.str.startswith("delay")].value.values[0]
r_int     = params_df.loc[params_df.parameter.str.startswith("r_int")].value.values[0]

# Package all other parameters 
param_names = params_df.loc[params_df.rhs, "parameter"].values.astype(str)
param_vals  = params_df.loc[params_df.rhs, "value"].values

# Make tuple to pass to lsig.integrate_DDE
dde_args = tuple(param_vals)

# Get `k`
where_k = next(i for i, pn in enumerate(param_names) if "k" == pn)
k = param_vals[where_k]
thresh = k

# Get `g`
where_g = next(i for i, pn in enumerate(param_names) if "g" == pn)
g = param_vals[where_g]

# Get `lambda` (basal promoter activity)
lambda_ = dde_args[4]


#######################################
######## Set up simulation ############
#######################################

# Set time parameters
tmax_days = 8
tmax = tmax_days / lsig.t_to_units(1)
nt_t = 100

# Get time points
nt = int(nt_t * tmax) + 1
t = np.linspace(0, tmax, nt)

# Make lattice
rows = cols = 80
X = lsig.hex_grid(rows, cols)

# Get # cells
n = X.shape[0]

# Get sender cell and center lattice on it
sender_idx = lsig.get_center_cells(X)
X = X - X.mean(axis=0)

# Adjacency
Adj = lsig.gaussian_irad_Adj(X, r_int, sparse=True, row_stoch=True)

# Set random seed
rng = np.random.default_rng(seed)

# Draw initial expression from a Half-Normal distribution with mean `lambda` (basal expression)
S0_rep = np.abs(rng.normal(
    size=(n_reps, n),
    scale=lambda_ * np.sqrt(np.pi/2)
))

# Fix sender cell(s) to constant expression
S0_rep[:, sender_idx] = 1


# __Set reporter signaling parameters__

# Reporter kinetics ratio
gamma_R = 0.1

# Package into args for lsig.reporter_rhs
R_args = [S0_rep[0], gamma_R, sender_idx]

# Initial R expression 
R0 = np.zeros(n, dtype=np.float32)



# Initialize results
S_actnum_param = np.empty((n_runs, nt), dtype=int)
S_tcmean_param = np.empty((n_runs, nt), dtype=np.float32)
S_A_param      = np.empty((n_runs, nt), dtype=np.float32)
R_actnum_param = np.empty((n_runs, nt), dtype=int)
R_tcmean_param = np.empty((n_runs, nt), dtype=np.float32)
R_A_param      = np.empty((n_runs, nt), dtype=np.float32)

# Make iterator
iterator = range(n_runs)
if progress_bar:
    iterator = tqdm(iterator)

for i in iterator:
    
    # Unpack parameters
#     rep, g_, rho_0_, rho_max_ = param_space[i]
    rep, g_, rho_0_ = param_space[i]
    rho_max_ = rho_max
    rep = int(rep)
    
    # Initial expression
    S0 = S0_rep[rep]

    # Supply S0 as parameter for R
    R_args = [S0, gamma_R, sender_idx]

    # Make a mask for transceivers
    tc_mask = np.ones(n, dtype=bool)
    tc_mask[sender_idx] = False
    
    # Get RHS of DDE equation to pass to integrator
    rhs = lsig.get_DDE_rhs(lsig.signal_rhs, Adj, sender_idx, lsig.beta_rho_exp, beta_args,)
    
    # Calculate density
    rho_t = lsig.logistic(t, g_, rho_0_, rho_max_)
    
    # Simulate
    S_t = lsig.integrate_DDE_varargs(
        t,
        rhs,
        var_vals=[rho_t],
        where_vars=where_rho,
        dde_args=dde_args,
        E0=S0,
        delay=delay,
        varargs_type="list",
    )
    
    # Number of activated TCs
    S_actnum_t = (S_t > thresh).sum(axis=1) - sender_idx.size
    
    # Mean fluorescence
    S_tcmean_t = S_t[:, tc_mask].mean(axis=1)
    
    # Save signal results
    S_xaxis_param[i]  = S_t[:, x_axis_cells]
    S_actnum_param[i] = S_actnum_t
    S_tcmean_param[i] = S_tcmean_t
        
    # Simulate reporter expression
    R_t = lsig.integrate_DDE_varargs(
        t,
        lsig.reporter_rhs,
        var_vals=[S_t],
        where_vars=0,
        dde_args=R_args,
        E0=R0,
        delay=0,
        min_delay=0,
        varargs_type="list",
    )

    # Number of activated cells
    R_act_t = R_t > thresh
    R_actnum_t = R_act_t.sum(axis=1)
    
    # Mean fluorescence
    R_tcmean_t = R_t[:, tc_mask].mean(axis=1)
    
    # Save reporter results
    R_xaxis_param[i]  = R_t[:, x_axis_cells]
    R_actnum_param[i] = R_actnum_t
    R_tcmean_param[i] = R_tcmean_t
    
    # Calculate +S and +R areas
    S_A_param[i] = lsig.A_cells_um(S_actnum_t, rho_t)
    R_A_param[i] = lsig.A_cells_um(R_actnum_t, rho_t)


# <hr>

# ## Results handling

# __Package results__

# In[180]:


# Store results 
data_dict = dict(
    n                = n,
    t                = t,
    trial_name       = trial_name,
    param_names      = param_names,
    param_vals       = param_vals,
    beta_args        = beta_args,
    delay            = delay,
    irad             = r_int,
    r_int            = r_int,
    rho_max          = rho_max,
    random_seeds     = seed + rep_space,
    sender_idx       = sender_idx,
    x_axis_cells     = x_axis_cells,
    S0_rep           = S0_rep,
    free_param_names = free_param_names,
    param_space      = param_space,
    thresh           = thresh,
    gamma_R          = gamma_R,
    S_xaxis_param    = S_xaxis_param,
    S_actnum_param   = S_actnum_param,
    S_tcmean_param   = S_tcmean_param,
    S_A_param        = S_A_param,
    R_xaxis_param    = R_xaxis_param,
    R_actnum_param   = R_actnum_param,
    R_tcmean_param   = R_tcmean_param,
    R_A_param        = R_A_param,
)


# __Set up save directory__

# In[188]:


# Make results directory if it doesn't exist
data_dir = os.path.join(save_dir, run_name)
if not os.path.exists(data_dir):
    print("Creating directory:", data_dir)
    os.mkdir(data_dir)

# Make output filename
data_fname = os.path.join(
    save_dir, run_name, run_name + "_results"
)


# __Save__

# In[189]:


# Save compressed results
print("Saving to:", data_fname)
np.savez_compressed(data_fname, **data_dict)
print("Mission complete.")


# <hr>

# In[190]:


import holoviews as hv
hv.extension("matplotlib")

data = np.load(data_fname + ".npz")

hv.Image(np.sqrt(data["S_actnum_param"])).opts(cmap="blues") + hv.Image(np.sqrt(data["R_actnum_param"])).opts(cmap="kr")


# In[218]:


param_space_agg = data["param_space"][data["param_space"][:, 0] == 0., 1:]
param_rep_idx   = np.zeros((n_runs // n_reps, n_reps),   dtype=int)

for i, row in enumerate(param_space_agg):
    param_rep_idx[i] = (data["param_space"][:, 1:] == param_space_agg[i]).all(axis=1).nonzero()[0]


# In[191]:


tc1_max  = data["S_xaxis_param"][:, :, 0].max(axis=1)

step_delay = int(delay / (data["t"][1] - data["t"][0]))
tc1_dsdt = (data["S_xaxis_param"][:, (step_delay + 1), 0] - data["S_xaxis_param"][:, step_delay, 0]) / (data["t"][1] - data["t"][0])


# In[192]:


lsig.ecdf(tc1_max).opts(s=4) + lsig.ecdf(tc1_dsdt).opts(s=1, axiswise=True)


# In[282]:


tc1_dsdt_sort, tc1_dsdt_ecdf = lsig.ecdf_vals(tc1_dsdt)
win = 120
tc1_dsdt_smooth = np.convolve(tc1_dsdt_sort, np.ones(win), 'valid') / win
ecdf_smooth = tc1_dsdt_ecdf[(win//2):(-win//2 + 1)]


# In[291]:


tc1_dsdt_smooth[np.diff(np.diff(tc1_dsdt_smooth)).argmax() + 1]


# In[290]:


hv.Points(
    (ecdf_smooth, tc1_dsdt_smooth)
).opts(
    s=4,
) + hv.Points(
    (ecdf_smooth[0:-1] + np.diff(ecdf_smooth)/2, np.diff(tc1_dsdt_smooth))
).opts(
    s=4,
    axiswise=True
) + hv.Points(
    (ecdf_smooth[1:-1], np.diff(np.diff(tc1_dsdt_smooth)))
).opts(
    s=4,
    axiswise=True
)


# In[196]:


tc1_fin = data["S_xaxis_param"][:, -1, 0]
tc1_drop = tc1_max - tc1_fin


# In[197]:


lsig.ecdf(tc1_fin).opts(s=10) + lsig.ecdf(tc1_drop).opts(s=10, axiswise=True) + hv.Points((tc1_max, tc1_fin)).opts(s=10)


# In[198]:


xaxis_actnum_fin = (data["S_xaxis_param"][:, -1, :] > thresh).sum(axis=1)
xaxis_actnum_max = (data["S_xaxis_param"] > thresh).sum(axis=2).max(axis=1)


# In[199]:


lsig.ecdf(xaxis_actnum_fin).opts(s=10) + lsig.ecdf(xaxis_actnum_max - xaxis_actnum_fin).opts(s=10, axiswise=True) + hv.Points((xaxis_actnum_max, xaxis_actnum_fin)).opts(s=10)


# In[200]:


actnum_fin = data["S_actnum_param"][:, -1]
actnum_max = data["S_actnum_param"].max(axis=1)


# In[202]:


lsig.ecdf(actnum_fin).opts(s=4) + lsig.ecdf(actnum_max).opts(s=4, axiswise=True) + lsig.ecdf(actnum_max - actnum_fin).opts(s=4, axiswise=True) + hv.Points((actnum_max, actnum_fin)).opts(s=2)


# In[208]:


wt_runs = np.logical_and(data["param_space"][:, 1] == 0.7, data["param_space"][:, 2] == 1).nonzero()[0]


# In[209]:


data["S_actnum_param"][wt_runs]


# In[210]:


buffer = 0.05

xlim = tuple([
    lsig.g_to_units(g_space[0]  - buffer * (g_space[-1] - g_space[0])),
    lsig.g_to_units(g_space[-1] + buffer * (g_space[-1] - g_space[0])),
])

ylim = tuple([
    rho_0_space[0]  - buffer * (rho_0_space[-1] - rho_0_space[0]),
    rho_0_space[-1] + buffer * (rho_0_space[-1] - rho_0_space[0]),
])


# In[224]:


tc1_dsdt_mean   = tc1_dsdt[param_rep_idx].mean(axis=1)
actnum_fin_mean = actnum_fin[param_rep_idx].mean(axis=1)


# In[311]:


phase = (tc1_dsdt_mean > 0.6).astype(int) + 2 * (actnum_fin_mean > 36).astype(int)

phase_singlespot = hv.Scatter(
    (
        lsig.g_to_units(data["param_space"][param_rep_idx[:, 0], 1]),
        data["param_space"][param_rep_idx[:, 0], 2],
    ),
).opts(
    c=np.array(["k", "r", "b", "g"])[phase],
    s=40,
    xlim=xlim,
    ylim=ylim,
)

hv.output(phase_singlespot, dpi=140)


# In[ ]:




