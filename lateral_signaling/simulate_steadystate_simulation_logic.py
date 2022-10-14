from uuid import uuid4
import os
from pathlib import Path
from math import ceil
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import lateral_signaling as lsig
import matplotlib.pyplot as plt
import colorcet as cc
import h5py

# Unique directory name
uid = str(uuid4())

# Write to temporary (fast read/write) directory of choice
data_dir = Path(f"/tmp/{uid}")  # Use root temp dir (Linux/MacOS)
# data_dir = Path(f"/home/pbhamidi/scratch/lateral_signaling/tmp/{uid}")  # Use scratch dir on compute cluster

data_dir.mkdir(exist_ok=True)


def do_one_simulation(
    n_reps,
    tmax_days,
    nt_t,
    rows,
    cols,
    pct_senders,
    r_int,
    alpha,
    k,
    p,
    delta,
    lambda_,
    beta_args,
    gamma_R,
    rho_0,
    rho_max,
    delay,
    beta_function,
    g=0,            # Zero growth (constant density)
    ex=None,
    save=False,
    progress=False,
    **kwargs,
):
    """Run a lateral signaling simulation"""

    # Set time parameters
    nt = int(nt_t * tmax_days) + 1
    t_days = np.linspace(0, tmax_days, nt)

    # Convert to dimensionless units for simulation
    t = t_days / lsig.t_to_units(1)
    dt = t[1] - t[0]

    # Get delay in Euler steps
    step_delay = ceil(delay / dt)

    # Make lattice
    X = lsig.hex_grid(rows, cols)
    center = lsig.get_center_cells(X)
    X = X - X[center]

    # Get # cells
    n = X.shape[0]

    # Get cell-cell Adjacency
    Adj = lsig.get_weighted_Adj(X, r_int, sparse=True, row_stoch=True)

    # Get number of senders
    n_senders = round(n * pct_senders / 100)

    # Calculate density
    rho_t = lsig.logistic(t, g, rho_0, rho_max)

    # Get function for density-based signaling attenuation
    beta_func = lsig.get_beta_func(beta_function)

    # Initialize storage for each replicate
    sender_idx_rep = np.zeros((n_reps, n_senders), dtype=int)
    tc_mask_rep    = np.zeros((n_reps, n), dtype=bool)
    R_final_rep    = np.zeros((n_reps, n), dtype=np.float32)
    S_final_rep    = np.zeros((n_reps, n), dtype=np.float32)

    iterator = range(n_reps)
    if progress:
        iterator = tqdm(iterator)
    for rep in iterator:

        # Randomly assign sender cells
        sender_idx = np.random.choice(n, n_senders, replace=False)

        # Make a mask for transceivers
        tc_mask = np.ones(n, dtype=bool)
        tc_mask[sender_idx] = False

        # Draw initial expression from a Half-Normal distribution with mean
        #   `lambda` (basal expression)
        S0 = np.abs(np.random.normal(size=n, scale=lambda_ * np.sqrt(np.pi / 2)))

        # Fix sender cell(s) to constant expression
        S0[sender_idx] = 1

        # Package into args for lsig.reporter_rhs
        R_args = [S0, gamma_R, sender_idx]

        # Initial R expression
        R0 = np.zeros(n, dtype=np.float32)

        # Package args for DDe integrator
        S_args = (
            Adj,
            sender_idx,
            beta_func,
            beta_args,
            alpha,
            k,
            p,
            delta,
            lambda_,
            g,
            rho_0,
        )
        where_rho = len(S_args) - 1

        # Simulate
        S_t = lsig.integrate_DDE_varargs(
            t,
            rhs=lsig.signal_rhs,
            var_vals=[rho_t],
            where_vars=where_rho,
            dde_args=S_args,
            E0=S0,
            delay=delay,
            varargs_type="list",
        )

        # Make version of S with delay
        S_t_delay = S_t[np.maximum(np.arange(nt) - step_delay, 0)]

        # Package args for DDe integrator
        R_args = (
            Adj,
            sender_idx,
            beta_func,
            beta_args,
            alpha,
            k,
            p,
            delta,
            lambda_,
            g,
            rho_0,
            S_t_delay[0],
            gamma_R,
        )
        where_S_delay = where_rho + 1

        # Simulate reporter expression
        R_t = lsig.integrate_DDE_varargs(
            t,
            lsig.reporter_rhs,
            var_vals=[rho_t, S_t_delay],
            where_vars=[where_rho, where_S_delay],
            dde_args=R_args,
            E0=R0,
            delay=delay,
            varargs_type="list",
        )

        # Store outcome of replicate
        sender_idx_rep[rep] = sender_idx
        tc_mask_rep[rep] = tc_mask
        S_final_rep[rep] = S_t[-1]
        R_final_rep[rep] = R_t[-1]

    if save:
        artifacts = []
        if ex is not None:

            # Dump data to file
            dump_file = data_dir.joinpath("results.hdf5")
            with h5py.File(dump_file, "w") as f:
                f.create_dataset("rho", data=rho_0)
                f.create_dataset("sender_idx_rep", data=sender_idx_rep)
                f.create_dataset("tc_mask_rep", data=tc_mask_rep)
                f.create_dataset("S_final_rep", data=S_final_rep)
                f.create_dataset("R_final_rep", data=R_final_rep)
            artifacts.append(dump_file)

            # Add all artifacts to Sacred
            for _a in artifacts:
                ex.add_artifact(_a)

            # Save any source code dependencies to Sacred
            source_files = [
                "lateral_signaling.py",
            ]
            for sf in source_files:
                ex.add_source_file(sf)
