#!/usr/bin/env python
# coding: utf-8

import lateral_signaling as lsig
import numpy as np
import pandas as pd
from tqdm import tqdm
import numba

import scipy.stats as st
from scipy.sparse import csr_matrix, diags

import os
from glob import glob


def signal_rhs(S, S_delay, Adj, sender_idx, beta_func, beta_args, alpha, k, p, delta, lambda_, rho):
    """
    Right-hand side of the transciever circuit delay 
    differential equation. Uses a matrix of cell-cell contact 
    lengths `L`.
    """

    # Get signaling as a function of density
    beta = beta_func(rho, *beta_args)
    
    # Get input signal across each interface
    S_bar = beta * (Adj @ S_delay)

    # Calculate dE/dt
    dS_dt = (
        lambda_
        + alpha
        * (S_bar ** p)
        / (
            k ** p 
            + (delta * S_delay) ** p 
            + S_bar ** p
        )
        - S
    )

    # Set sender cell to zero
    dS_dt[sender_idx] = 0

    return dS_dt


def reporter_rhs(R, R_delay, S, gamma_R, sender_idx):
    """Reporter dynamical equation"""
    
    dR_dt = (S - R) * gamma_R
    dR_dt[sender_idx] = 0
    
    return dR_dt


# Unique name of current run
run_name = "20210803_sweep_TCphase_dense2_HPC"

# Name of parameter set
trial_name = "lowcis_expbeta"

# Directory containing parameter set
trial_dir = "/home/pbhamidi/git/evomorph/lateral_signaling/"

# Saving directory
save_dir = "/home/pbhamidi/git/evomorph/lateral_signaling/HPC_data"

# Path to MLE growth parameters data
mle_params_file = "/home/pbhamidi/git/evomorph/data/growth_parameters_MLE.csv"

# Working directory
wd = "/home/pbhamidi/git/evomorph/lateral_signaling/"

# # Directory containing parameter set
# trial_dir = "C:/Users/Pranav/git/evomorph/lateral_signaling/"

# # Saving directory
# save_dir = "C:/Users/Pranav/git/evomorph/lateral_signaling/sim_data/"

# # Path to MLE growth parameters data
# mle_params_file = "C://Users/Pranav/git/evomorph/data/growth_parameters_MLE.csv"

# # Working directory
# wd = "C:/Users/Pranav/git/evomorph/lateral_signaling/"


# Set random seed
seed = 2021
np.random.seed(seed)

# Whether to display progress
progress_bar = True


os.chdir(wd)
print("Current directory:", os.getcwd())

# Directory to save results
save_dir = os.path.realpath(save_dir)
assert os.path.exists(save_dir), f"Directory does not exist: {save_dir}"
print("Will save to directory:", save_dir)


# __Get growth parameters__

assert os.path.exists(mle_params_file), "File does not exist"

mle_params_df = pd.read_csv(mle_params_file, index_col=0)
g, rho_max = mle_params_df.loc[
    mle_params_df.condition == "untreated", ["g_ratio", "rho_max_ratio"]
].values.ravel()


# __Set RHS of dynamical equation__

# Set the RHS function in long-form
rhs_long = signal_rhs

# Set beta(rho)
beta_func = lsig.beta_rho_exp


# __Load params__

# Search trial_dir for parameter set CSV
params_regexp = "*" + trial_name + "*.csv"
params_path = glob(os.path.join(trial_dir, params_regexp))

# Load parameter set
if len(params_path) == 0:
    raise FileNotFoundError(
        f"No file matches the regular expression `{params_regexp}` "
        + f"in the directory `{os.path.abspath(trial_dir)}` ."
    )
elif len(params_path) > 1:
    raise FileNotFoundError(
        f"More than one file matches the regular expression "
        + f"`{params_regexp}` in the directory "
        + f"`{os.path.abspath(trial_dir)}` "
    )
else:
    # Read trial parameters
    params_path = os.path.abspath(params_path[0])
    params_df = pd.read_csv(params_path)

# Get any arguments for beta function
is_beta_arg = [p.startswith("beta_") for p in params_df["parameter"].values]
beta_args   = params_df.value.values[is_beta_arg]

# Get the delay parameter
is_delay    = [p == "delay" for p in params_df["parameter"].values]
delay       = params_df.value.values[is_delay][0]

# Package all other parameters 
is_param    = [not (ba or d) for ba, d in zip(is_beta_arg, is_delay)]
param_names = params_df.parameter.values[is_param].astype(str)
param_vals  = params_df.value.values[is_param].astype(np.float32)

# Package arguments for lsig.integrate_DDE and 
#   lsig.integrate_DDE_varargs. Density param is 
#   initialized with an arbitrary value.
dde_args = *param_vals, 1.

# Get `g`
where_g = next(i for i, pn in enumerate(param_names) if "g" == pn)
g = param_vals[where_g]

# (Optional) Remove `g` from signaling parameters
dde_args = [*dde_args[:where_g], *dde_args[(where_g+1):]]

# Get index of `rho` (last argument)
where_rho = len(dde_args) - 1

# Get `k`
where_k = next(i for i, pn in enumerate(param_names) if "k" == pn)
k = param_vals[where_k]
thresh = k

# Get basal promoter activity (`lambda`)
lambda_ = dde_args[4]


# __Sender parameters__
# Percent senders in lattice
pct_s = 1


# __Time parameters__
# Set time parameters
tmax = 5
nt_t = 100

# Get time points
nt = int(nt_t * tmax) + 1
t = np.linspace(0, tmax, nt)

# __Construct lattice of cells__

# Make lattice
rows = cols = 100
X, Adj = lsig.hex_Adj(rows, cols, sparse=True, row_stoch=True)

# Get # cells
n = X.shape[0]

# Specify percent of population that is sender
n_s = int(n * (pct_s / 100)) + 1

# Define free parameters for parameter scan
# Dense sampling
rep_space     = np.arange(5)
g_space       = np.linspace(0.25, 2.25, 25)
rho_0_space   = np.linspace(0, 6, 25)[1:]
rho_max_space = np.linspace(0, 6, 25)[1:]

# # Sparse sampling
# rep_space     = np.arange(5)
# g_space       = np.linspace(0.25, 2.25, 5)
# rho_0_space   = np.linspace(0, rho_max, 5)[1:]
# rho_max_space = np.linspace(0, 8, 5)[1:]
 
free_params = (
    rep_space, 
    g_space, 
    rho_0_space, 
    rho_max_space,
)
free_param_names = (
    "rep", 
    "g", 
    "rho_0", 
    "rho_max",
)

# Make array with all combinations of params
param_space = np.meshgrid(*free_params)
param_space = np.array(param_space).T.reshape(-1, len(free_params))

# Get number of simulations
n_runs = param_space.shape[0]

# Get number of replicates
n_reps = rep_space.size

# Get number of unique parameter sets in sweep
n_param_sets = int(n_runs / n_reps)


# __Set senders and initial expression__
# Get sender indices for each replicate
sender_idx_rep = np.empty((n_reps, n_s), dtype=int)
S0_rep         = np.empty((n_reps, n), dtype=np.float32)

for rep in rep_space:
    
    # Change random seed for each replicate
    seed_ = seed + rep
    np.random.seed(seed_)
    
    # Assign senders randomly
    sender_idx_rep[rep] = np.random.choice(n, n_s, replace=False)
    
    # Random initial expression
    ## Drawn from a Half-Normal distribution with mean `lambda`
    S0_rep[rep] = st.halfnorm.rvs(
        size=n, 
        scale=lambda_ * np.sqrt(np.pi/2), 
        random_state=seed_,
    ).astype(np.float32)
    
    # Fix sender cell(s) to constant expression
    S0_rep[rep, sender_idx_rep[rep]] = 1
    
# Reset random seed
np.random.seed(seed)


# __Set reporter parameters__

# Reporter kinetics ratio
gamma_R = 0.1

# Package into args for reporter_rhs
R_args = [S0_rep[0], gamma_R, sender_idx_rep[0]]

# Initial R expression 
R0 = np.zeros(n, dtype=np.float32)



#### Run simulations

# Make mutable copy of dde args
args = list(dde_args)

# Initialize results vectors
S_actnum_param = np.empty((n_runs, nt), dtype=int)
S_tcmean_param = np.empty((n_runs, nt), dtype=np.float32)
R_actnum_param = np.empty((n_runs, nt), dtype=int)
R_tcmean_param = np.empty((n_runs, nt), dtype=np.float32)

# Make iterator
iterator = range(n_runs)

# # Make test iterator
# iterator = range(24)

if progress_bar:
    iterator = tqdm(iterator)

for i in iterator:    
    
    # Get parameters
    rep, g_, rho_0_, rho_max_ = param_space[i]
#     rep, g_, rho_0_ = param_space[i]
#     rho_max_ = rho_max
    rep = int(rep)
    
    # Get senders
    sender_idx = sender_idx_rep[rep]
    S0 = S0_rep[rep]

    # Supply S initial state as parameter for R
    R_args = [S0, gamma_R, sender_idx]

    # Make a mask for transceivers
    tc_mask = np.ones(n, dtype=bool)
    tc_mask[sender_idx] = False
    
    # Get RHS of DDE equation to pass to integrator
    rhs = lsig.get_DDE_rhs(rhs_long, Adj, sender_idx, beta_func, beta_args,)
    
    # Calculate density
    rho_t = lsig.logistic(t, g_, rho_0_, rho_max_)
    
    # Simulate
    S_t = lsig.integrate_DDE_varargs(
        t,
        rhs,
        var_vals=[rho_t],
        where_vars=where_rho,
        dde_args=args,
        E0=S0,
        delay=delay,
        varargs_type="list",
    )
    
    # Number of activated cells
    S_act_t = S_t > thresh
    S_actnum_t = S_act_t.sum(axis=1)
    
    # Mean fluorescence
    S_tcmean_t = S_t[:, tc_mask].mean(axis=1)
    
    # Save results
    S_actnum_param[i] = S_actnum_t
    S_tcmean_param[i] = S_tcmean_t
    
    # Simulate reporter expression
    R_t = lsig.integrate_DDE_varargs(
        t,
        reporter_rhs,
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
    
    # Save results
    R_actnum_param[i] = R_actnum_t
    R_tcmean_param[i] = R_tcmean_t
    
    # Save results for reporter
    R_actnum_param[i] = R_actnum_t
    R_tcmean_param[i] = R_tcmean_t


#### Results handling

# Store results 
data_dict = dict(
    n                = n,
    t                = t,
    trial_name       = trial_name,
    param_names      = param_names,
    param_vals       = param_vals,
    beta_args        = beta_args,
    delay            = delay,
    irad             = 1.,
#     rho_max          = rho_max,
    random_seeds     = seed+rep_space,
    sender_idx_rep   = sender_idx_rep,
    S0_rep           = S0_rep,
    free_param_names = free_param_names,
    param_space      = param_space,
    S_actnum_param   = S_actnum_param,
    S_tcmean_param   = S_tcmean_param,
    gamma_R          = gamma_R,
    R_actnum_param   = R_actnum_param,
    R_tcmean_param   = R_tcmean_param,
)

# Make results directory if it doesn't exist
data_dir = os.path.join(save_dir, run_name)
if not os.path.exists(data_dir):
    print("Creating directory", data_dir)
    os.mkdir(data_dir)

# Make output filename
data_fname = os.path.join(
    save_dir, run_name, run_name + "_results"
)

# Save compressed
np.savez_compressed(data_fname, **data_dict)
print("Mission complete.")
