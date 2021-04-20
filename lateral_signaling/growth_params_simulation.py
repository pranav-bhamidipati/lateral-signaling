#!/usr/bin/env python
# coding: utf-8

# ### Set up environment

# In[1]:


import lateral_signaling as lsig
import numpy as np
import pandas as pd
from tqdm import tqdm
import numba

import scipy.stats as st
from scipy.sparse import csr_matrix

import os
from glob import glob


# __Define RHS function(s)__

# In[2]:


def tc_rhs_beta_g_normA(S, S_delay, Adj, sender_idx, beta_func, beta_args, alpha, k, p, delta, lambda_, g, rho):
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
            ((g ** 2) * k) ** p 
            + (delta * S_delay) ** p 
            + S_bar ** p
        )
        - g * S
    )

    # Set sender cell to zero
    dS_dt[sender_idx] = 0

    return dS_dt


# In[3]:


from scipy.sparse import csr_matrix, diags, identity
import scipy.stats

def k_step_Adj(k, rows, cols=0, dtype=np.float32, row_stoch=False, **kwargs):
    """
    """
    
    if not cols:
        cols = rows
        
    # Construct adjacency matrix
    a = lsig.make_Adj_sparse(rows, cols, dtype=dtype, **kwargs)
    
    # Add self-edges
    n = rows * cols
    eye = identity(n).astype(dtype)
    A = (a + eye)
    
    # Compute number of paths of length k between nodes
    A = A ** k
    
    # Store as 0. or 1.
    A = (A > 0).astype(dtype)
    
    # Remove self-edges
    A = A - diags(A.diagonal())
    
    if row_stoch:
        rowsum = np.sum(A, axis=1)
        A = csr_matrix(A / rowsum)
    
    return A


# <hr>

# In[27]:


# Set directory
os.chdir("/home/pbhamidi/git/evomorph/lateral_signaling/")

# Print directory
print("Current directory: " + os.getcwd())

# Unique name of current run
run_name = "20210415_sweep_TCphase_dense_ling"

# Directory to save results
res_dir = "/home/pbhamidi/git/evomorph/lateral_signaling/data"

# Directory of parameter set
trial_dir = "."

# Name of parameter set
trial_name = "lowcis_expbeta"

# Set random seed
seed = 2021
np.random.seed(seed)

# Whether to display progress
progress_bar = True


# __Set growth min and max__

# In[5]:


# Set min and max density
rho_min, rho_max = 1, 5.63040245


# __Set RHS of dynamical equation__

# In[6]:


# Set the RHS function in long-form
rhs_long = tc_rhs_beta_g_normA

# Set beta(rho)
beta_func = lsig.beta_rho_exp


# __Set distance scale__

# In[7]:


# cell-cell distance (dimensionless)
r = 1.   


# __Convert simulation time (dimensionless) to days__

# In[8]:


t_to_days = lambda t: t / 7.28398176e-01   # time in days


# <hr>

# ## Fetch parameters

# In[9]:


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


# In[10]:


# Get any arguments for beta function
is_beta_arg = [p.startswith("beta_") for p in params_df["parameter"].values]
beta_args   = params_df.value.values[is_beta_arg]

# Get the delay parameter
is_delay    = [p == "delay" for p in params_df["parameter"].values]
delay       = params_df.value.values[is_delay][0]

# Package all other parameters 
is_param    = [not (ba or d) for ba, d in zip(is_beta_arg, is_delay)]
param_names = params_df.parameter.values[is_param]
param_vals  = params_df.value.values[is_param]

# Package arguments for lsig.integrate_DDE and 
#   lsig.integrate_DDE_varargs. Density param is 
#   initialized with rho_min.
dde_args = *param_vals, rho_min


# In[11]:


# Get index of `rho` (last argument)
where_rho = len(dde_args) - 1

# Get `k`
where_k = next(i for i, pn in enumerate(param_names) if "k" == pn)
k = param_vals[where_k]
thresh = k

# Get `g`
where_g = next(i for i, pn in enumerate(param_names) if "g" == pn)
g = param_vals[where_g]


# <hr>

# ## Figure parameters

# In[12]:


pct_s = 1


# __Set time parameters__

# In[13]:


# Set time parameters
tmax = 5
nt_t = 100

# Get time points
nt = int(nt_t * tmax) + 1
t = np.linspace(0, tmax, nt)


# __Construct lattice of cells__

# In[14]:


# Make lattice
rows = cols = 100
X = lsig.hex_grid(rows, cols, r=r)

# Get # cells
n = X.shape[0]

# Calculate cell adjacency
kAdj_1 = k_step_Adj(1, rows, cols, row_stoch=True)

# Specify percent of population that is sender
n_s = int(n * (pct_s / 100)) + 1

# Define free parameters for parameter scan
rep_space     = np.arange(5)
g_space       = np.linspace(0.25, 2.25, 25)
rho_0_space   = np.linspace(0, 8, 25)[1:]
rho_max_space = np.linspace(0, 8, 25)[1:]
free_params   = (rep_space, g_space, rho_0_space, rho_max_space)
param_names   = ("rep", "g", "rho_0", "rho_max")

# Make array with all combinations of params
param_space = np.meshgrid(*free_params)
param_space = np.array(param_space).T.reshape(-1, len(free_params))

# Get sender indices for each replicate
sender_idx_rep = np.empty((rep_space.size, n_s), dtype=int)
for rep in rep_space:    
    # Set random seed
    np.random.seed(seed + rep)
    # Assign senders randomly
    sender_idx = np.random.choice(n, n_s, replace=False)
    sender_idx_rep[rep] = sender_idx

# Get number of simulations
n_runs = param_space.shape[0]

# Make mutable copy of dde args
args = list(dde_args)

# Initialize results vectors
S_actnum_param = np.empty((n_runs, nt), dtype=int)
S_tcmean_param = np.empty((n_runs, nt), dtype=np.float32)

# Make iterator
iterator = range(n_runs)

# # Test iterator
# iterator = range(24)

if progress_bar:
    iterator = tqdm(iterator)

for i in iterator:    
    
    # Get parameters
    rep, g_, rho_0_, rho_max_ = param_space[i]
    args[where_g] = g_
  
    # Get senders
    sender_idx = sender_idx_rep[int(rep)]
    
    # Make a mask for transceivers
    tc_mask = np.ones(n, dtype=bool)
    tc_mask[sender_idx] = False
    
    # Get RHS of DDE equation to pass to integrator
    rhs_1 = lsig.get_DDE_rhs(rhs_long, kAdj_1, sender_idx, beta_func, beta_args,)
    
    # Get initial conditions
    S0 = np.zeros(n)
    S0[sender_idx] = 1

    # Calculate density
    rho_t = lsig.logistic(t, g_, rho_0_, rho_max_)

    # Simulate
    S_t = lsig.integrate_DDE_varargs(
        t,
        rhs_1,
        var_vals=rho_t,
        where_vars=where_rho,
        dde_args=args,
        E0=S0,
        delay=delay
    )
    
    # Number of activated cells
    S_act_t = S_t > thresh
    S_actnum_t = S_act_t.sum(axis=1)

    # Mean fluorescence
    S_tcmean_t = S_t[:, tc_mask].mean(axis=1)
    
    # Save results
    S_actnum_param[i] = S_actnum_t
    S_tcmean_param[i] = S_tcmean_t


# Store results 
data_dict = dict(
    n=n,
    t=t,
    random_seeds=seed+rep_space,
    sender_idx_rep=sender_idx_rep,
    param_names=param_names,
    param_space=param_space,
    S_actnum_param=S_actnum_param,
    S_tcmean_param=S_tcmean_param,
)

# Make results directory if it doesn't exist
data_dir = os.path.join(res_dir, run_name)
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

# In[34]:

# Save data 
data_file = os.path.join(
    res_dir, run_name, run_name + "_results"
)
np.savez_compressed(data_file, **data_dict)


# In[ ]:


print("Mission complete.")

