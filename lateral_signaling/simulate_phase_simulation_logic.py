from uuid import uuid4
from tqdm import tqdm
import os
import h5py

import numpy as np
import matplotlib.pyplot as plt

import lateral_signaling as lsig


# Use a unique directory name for this run
uid = str(uuid4())

# Write to temporary (fast read/write) directory of choice
# data_dir = f"/tmp/{uid}"    # Use root temp dir (Linux/MacOS)
# data_dir = f"./tmp/{uid}"   # Use local temp
data_dir = f"/home/pbhamidi/scratch/lateral_signaling/tmp/{uid}"  # Use scratch dir on compute cluster
os.makedirs(data_dir, exist_ok=True)


def do_one_simulation(
    n_reps,
    g_space,
    tmax_days,
    nt_t,
    rows,
    cols,
    r_int,
    alpha,
    k,
    p,
    delta,
    lambda_,
    beta_args,
    gamma_R,
    g,
    rho_0,
    rho_max,
    delay,
    beta_function,
    progress=False,
    nt_t_save=100,
    ex=None,
    save=False,
    uid=uid,
    **kwargs,
):
    """Run a lateral signaling simulation and calculate phase metrics."""

    if progress:
        print(f"[ID {uid[:8]}]: Starting")

    # Set time span
    nt = int(nt_t * tmax_days) + 1
    t_days = np.linspace(0, tmax_days, nt)

    # Convert to dimensionless units for simulation
    t = t_days / lsig.t_to_units(1)
    dt = t[1] - t[0]

    # Get number of time-steps per delay
    step_delay = int(delay / dt)

    # Make lattice
    X = lsig.hex_grid(rows, cols)

    # Get # cells
    n = X.shape[0]

    # Get sender cell and center lattice on it
    sender_idx = lsig.get_center_cells(X)
    X = X - X[sender_idx]

    # Get index of TC1 (closest TC to sender)
    x_bias = np.array([-1e-6, 0.0])
    tc1_idx = np.argsort(np.linalg.norm(X + x_bias, axis=1))[1]

    # Get cell-cell Adjacency
    Adj = lsig.get_weighted_Adj(X, r_int, sparse=True, row_stoch=True)

    # Get function for density-based signaling attenuation
    beta_func = lsig.get_beta_func(beta_function)

    # Draw initial expression from a Half-Normal distribution with mean
    #   `lambda` (basal expression)
    S0_rep = np.abs(
        np.random.normal(size=n * n_reps, scale=lambda_ * np.sqrt(np.pi / 2))
    ).reshape(n_reps, n)

    # Fix sender cell(s) to constant expression
    S0_rep[:, sender_idx] = 1

    # Make a mask for transceivers
    tc_mask = np.ones(n, dtype=bool)
    tc_mask[sender_idx] = False

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

    # Package args for DDe integrator
    R_args = (
        Adj,
        sender_idx,
        lsig.beta_rho_exp,
        beta_args,
        alpha,
        k,
        p,
        delta,
        lambda_,
        g,
        rho_0,
        S0_rep[0],
        gamma_R,
    )
    where_S_delay = where_rho + 1

    # Initial R expression
    R0 = np.zeros(n, dtype=np.float32)

    # Calculate density time-courses
    rho_t_g = np.asarray([lsig.logistic(t, _g, rho_0, rho_max) for _g in g_space])

    # Skip time-points to reduce filesize
    save_skip = int(nt_t / nt_t_save)
    nt_save = np.arange(0, nt, save_skip).size

    # Set the threshold to consider a cell activated (equal to promoter threshold)
    thresh = k

    # Initialize output
    n_g = len(g_space)
    S_t_g_tcmean = np.empty((n_g, nt_save), dtype=np.float32)
    R_t_g_tcmean = np.empty_like(S_t_g_tcmean)
    S_t_g_actnum = np.empty((n_g, nt_save), dtype=np.float32)
    R_t_g_actnum = np.empty_like(S_t_g_actnum)
    v_init_g = np.empty((n_g), dtype=np.float32)
    n_act_fin_g = np.empty((n_g), dtype=np.float32)

    iterator = range(n_g)
    if progress:
        iterator = tqdm(iterator)
    for j in iterator:

        S_t_rep = np.empty((n_reps, nt, n), dtype=np.float32)
        R_t_rep = np.empty_like(S_t_rep)

        for i in range(n_reps):
            # Simulate
            S_t_rep[i] = lsig.integrate_DDE_varargs(
                t,
                rhs=lsig.signal_rhs,
                var_vals=[rho_t_g[j]],
                where_vars=where_rho,
                dde_args=S_args,
                E0=S0_rep[i],
                delay=delay,
                varargs_type="list",
            )

            # Make version of S after delay
            S_t_delay = S_t_rep[i, np.maximum(np.arange(nt) - step_delay, 0)]

            # Simulate reporter expression
            R_t_rep[i] = lsig.integrate_DDE_varargs(
                t,
                lsig.reporter_rhs,
                var_vals=[rho_t_g[j], S_t_delay],
                where_vars=[where_rho, where_S_delay],
                dde_args=R_args,
                E0=R0,
                delay=delay,
                varargs_type="list",
            )

        # Mean fluorescence
        S_t_g_tcmean[j] = S_t_rep[:, ::save_skip, tc_mask].mean(axis=(0, -1))
        R_t_g_tcmean[j] = R_t_rep[:, ::save_skip, tc_mask].mean(axis=(0, -1))

        # Number of TCs over threshold (activated)
        S_t_g_actnum[j] = (
            (S_t_rep[:, ::save_skip, tc_mask] > thresh).sum(axis=-1).mean(axis=0)
        )
        R_t_g_actnum = (
            (R_t_rep[:, ::save_skip, tc_mask] > thresh).sum(axis=-1).mean(axis=0)
        )

        # Intial velocity (dS/dt of the 1st TC at time t=delay)
        v_init_g[j] = (
            S_t_rep[:, (step_delay + 1), tc1_idx] - S_t_rep[:, step_delay, tc1_idx]
        ).mean() / dt

        # Number of TCs producing signal at end of simulation
        n_act_fin_g[j] = S_t_g_actnum[j, -1]

        # if progress:
        # print(f"[ID {uid[:8]}]: {j+1}/{n_g}")

    if save:

        if ex is not None:

            # Keep track of objects that Sacred should save
            artifacts = []

            # Dump data to an HDF5 file
            data_dump_fname = data_dir.joinpath("results.hdf5")
            with h5py.File(data_dump_fname, "w") as f:
                f.create_dataset("t", data=t[::save_skip])
                f.create_dataset("sender_idx", data=sender_idx)
                f.create_dataset("X", data=X)
                f.create_dataset("g_space", data=g_space)
                f.create_dataset("rho_t_g", data=rho_t_g[:, ::save_skip])
                f.create_dataset("S_t_g_tcmean", data=S_t_g_tcmean)
                f.create_dataset("S_t_g_actnum", data=S_t_g_actnum)
                f.create_dataset("R_t_g_tcmean", data=R_t_g_tcmean)
                f.create_dataset("R_t_g_actnum", data=R_t_g_actnum)
                f.create_dataset("v_init_g", data=v_init_g)
                f.create_dataset("n_act_fin_g", data=n_act_fin_g)
            artifacts.append(data_dump_fname)

            # Add objects to save to Sacred
            for _a in artifacts:
                ex.add_artifact(_a)

            # Add any source code dependencies to Sacred
            source_files = [
                "lateral_signaling.py",
            ]
            for _sf in source_files:
                ex.add_source_file(_sf)
