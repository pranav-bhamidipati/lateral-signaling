#!/usr/bin/env python
# coding: utf-8

#########################
## IMPORT DEPENDENCIES ##
#########################

import numpy as np
import pandas as pd
import numba
from math import ceil

from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay, ConvexHull
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import tqdm

import lattice_signaling as lsig
from scipy.sparse import csr_matrix

###########################
## DEFINE USER FUNCTIONS ##
###########################

@numba.njit
def act_radius_hex(t, X, E_save, thresh, L):
    """
    """
    
    # Get time-step
    dt = t[1] - t[0]
    
    # Get # activated cells at each time
    Etsum = (E_save > thresh).sum(axis=1)
    
    # Calculate area and estimated radius
    # Note: Cells are identical hexagons of side length 1/sqrt(3)
    a = Etsum * np.sqrt(3)/2
    r  = np.sqrt(a / np.pi).astype(np.float32)
    
    # Truncate if the radius reaches the periodic boundary
    bound = (L-1)*np.sqrt(3)/4
    crossed = np.argmax(r > bound)
    if crossed > 0:
        r = r[:crossed]
        tr = t[:crossed]
    else:
        tr = t.copy()
        
    # Isolate time-points where area changes
    jumps = np.diff(r[:-1]).nonzero()[0]
    
    # Construct output
    t_out = np.empty(jumps.size + 2)
    r_out = np.empty(jumps.size + 2)
    t_out[0], t_out[1:-1], t_out[-1] = tr[0], tr[jumps], tr[-1]
    r_out[0], r_out[1:-1], r_out[-1] = r[0], r[jumps], r[-1]
    
    return t_out, r_out


def hex_Adj(s, r=1, dtype=np.float32, csr=True):
    """
    """
    # Get # cells
    n = s ** 2
    
    # Make hexagonal grid
    X = lsig.hex_grid_square(n, r=r)
    
    # Construct adjacency matrix
    Adj = np.zeros((n,n), dtype=dtype)
    for i in range(s):
        for j in range(s):
            
            # Get neighbors of cell at location i, j
            nb = np.array(
                [
                    (i    , j + 1),
                    (i    , j - 1),
                    (i - 1, j    ),
                    (i + 1, j    ),
                    (i - 1 + 2*(j%2), j - 1),
                    (i - 1 + 2*(j%2), j + 1),
                ]
            ) % s
            
            # Populate Adj
            nbidx = np.array([ni*s + nj for ni, nj in nb])
            Adj[i*s + j, nbidx] = 1
    
    if csr:
        Adj = csr_matrix(Adj)
    
    return X, Adj


def integrate_DDE(
    t_span,
    rhs,
    dde_args,
    E0,
    delay,
    progress_bar=False,
    min_delay=5,
):
    # Get # time-points, dt, and # cells
    n_t = t_span.size
    dt = t_span[1] - t_span[0]
    n_c = E0.size
    
    # Get delay in steps
    step_delay = np.atleast_1d(delay) / dt
    assert (step_delay >= min_delay), (
        "Delay time is too short. Lower dt or lengthen delay."
    )
    step_delay = ceil(step_delay)
    
    # Initialize expression vector
    E_save = np.empty((n_t, n_c), dtype=np.float32)
    E_save[0] = E = E0
    
    # Construct time iterator
    iterator = np.arange(1, n_t)
    if progress_bar:
        iterator = tqdm.tqdm(iterator)

    for step in iterator:
        # Get past E
        past_step = max(0, step - step_delay)
        E_delay = E_save[past_step]
        
        # Integrate
        dE_dt = rhs(E, E_delay, *dde_args)
        E = np.maximum(0, E + dE_dt * dt) 
        E_save[step] = E
    
    return E_save


#############################
## SIMULATION AND ANALYSIS ##
#############################

# Set lattice parameters
L = 21
n = L ** 2
X, Adj = hex_Adj(L, csr=False)
sender_idx = lsig.get_center_cells(X)[0]


# Define RHS
@numba.njit
def hex_tc_rhs(E, E_delay, alpha, k, p, delta, lambda_, eps, sender_idx):
    """
    Returns RHS of transceiver DDE.
    """
    # Get signaling across each interface
    E_bar = eps * (Adj @ E_delay)
    
    # Calculate dE/dt
    dE_dt = (
        lambda_
        + alpha * (E_bar ** p) / (k ** p + (delta * E_delay) ** p + E_bar ** p)
        - E
    )
    
    # Set sender cell to zero
    dE_dt[sender_idx] = 0
    
    return dE_dt

# Set DDE parameters
p_s = 2
lambda_ = 1e-5

# Set time span
dt = 0.005
tmax = 5
t = np.linspace(0, tmax, int(tmax / dt) + 1)

# Set initial condition
E0 = np.zeros(n, dtype=np.float32)
E0[sender_idx] = 1

# Define free parameters for parameter scan
alpha_space = np.linspace(0.5, 3, 6)
k_s_space = np.logspace(-2, -0.5, 11)
delta_space = np.linspace(0, 4, 3)
eps_space = np.linspace(0.1, 1, 10)
tau_space = np.linspace(0.1, 0.7, 7)
free_params = (alpha_space, k_s_space, delta_space, eps_space, tau_space)

# Generate parameter-sets
param_space = np.meshgrid(*free_params)
param_space = np.array(param_space).T.reshape(-1, len(free_params))
n_runs = param_space.shape[0]

# Make iterator
iterator = range(n_runs)
iterator = tqdm.tqdm(iterator)

# Simulate each parameter set
print(f"Simulating {n_runs} parameter sets.")
E_save_arr = np.empty((n_runs, t.size, n))
for i in iterator:

    # Unpack parameters
    alpha, k_s, delta, eps, tau = param_space[i]
    args = (alpha, k_s, p_s, delta, lambda_, eps, sender_idx)
    
    # Simulate
    E_save = integrate_DDE(
        t_span=t,
        rhs=hex_tc_rhs,
        dde_args=args,
        E0=E0,
        delay=tau,
        progress_bar=False,
    )
    E_save_arr[i] = E_save

# Re-make iterator
iterator = range(n_runs)
iterator = tqdm.tqdm(iterator)

# Calculate mean wavefront velocity 
vmeans = np.empty(n_runs, dtype=np.float32)
for i in iterator:
    E = E_save_arr[i]
    tr, r = act_radius_hex(t, X, E, 0.1, L)
    vmean = (r[-1] - r[0]) / (tr[-1] - t[0])
    vmeans[i] = vmean


##########################
## WRITING DATA TO FILE ##
##########################

import datetime
import os
import sys

to_dir = "/home/ubuntu/git/evomorph/data/" + str(datetime.date.today()) + "_hexGRN_sweep"
if not os.path.exists(to_dir):
    os.mkdir(to_dir)

# Save metadata as csv
mdata_path = os.path.join(to_dir, "metadata.csv")
metadata = pd.DataFrame(dict(
    dt = dt,
    tmax = tmax,
    sender_idx = sender_idx,
    E0_tc = 0,
    E0_sender = 1,
)).to_csv(mdata_path)

# Save expression data and parameter space in compressed format (.npz)
E_path = os.path.join(to_dir, "Esave_paramspace.npz")
np.savez_compressed(E_path, {"E_save_arr": E_save_arr, "param_space": param_space})

# Save velocity data to file
v_path = os.path.join(to_dir, "mean_wave_velocity.csv")
vmean_data = pd.DataFrame({
    "promoter stregth": param_space[:, 0],
    "signaling threshold": param_space[:, 1], 
    "feedback strength": param_space[:, 2],
    "% shared interface": param_space[:, 3] * 100, 
    "delay time": tau
    "Mean wavefront speed": vmeans,
}).to_csv(v_path)