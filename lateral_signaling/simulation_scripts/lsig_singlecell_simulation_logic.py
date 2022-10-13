from uuid import uuid4
import os
from math import ceil
import numpy as np
import lateral_signaling as lsig
import h5py

# Use a unique directory name for this run
uid = str(uuid4())

# Write to temporary (fast read/write) directory of choice
data_dir = os.path.abspath(f"/tmp/{uid}")  # Use root temp dir (Linux/MacOS)
# data_dir = f"/home/pbhamidi/scratch/lateral_signaling/tmp/{uid}"  # Use scratch dir on compute cluster

os.makedirs(data_dir, exist_ok=True)


def do_one_simulation(
    tmax_days,
    nt_t,
    alpha,
    k,
    p,
    delta,
    lambda_,
    beta_args,
    gamma_R,
    g,
    delay,
    beta_function,
    rho_0,
    ex=None,
    save=False,
    save_skip=1,
    **kwargs,
):
    """Run a lateral signaling simulation"""

    # Set time parameters
    nt = int(nt_t * tmax_days) + 1
    t_days = np.linspace(0, tmax_days, nt)

    # Convert to dimensionless units for simulation
    t = t_days / lsig.t_to_units(1)
    dt = t[1] - t[0]

    # Set # cells
    n = 3

    # Set indices of Sender, Receiver, and Transceiver
    receiver_idx = np.array([0], dtype=int)
    sender_idx = np.array([1], dtype=int)
    transceiver_idx = np.array([2], dtype=int)

    # Get cell-cell Adjacency
    Adj = (
        np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=np.float32,
        )
        / 6
    )

    # Set initial expressions
    S0 = np.array([0, 1, 0], dtype=np.float32)

    # Get function for density sensitivity
    beta_func = lsig.get_beta_func(beta_function)

    # Package args for DDe integrator
    fixed_idx_tc = np.array([0, 1])
    tc_args = (
        Adj,
        fixed_idx_tc,
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

    # Simulate
    S_t_tc = lsig.integrate_DDE(
        t,
        rhs=lsig.signal_rhs,
        dde_args=tc_args,
        E0=S0,
        delay=delay,
    )

    # Get delayed expression
    step_delay = ceil(delay / dt)
    S_t_delay = S_t_tc[np.maximum(np.arange(nt) - step_delay, 0)]

    # Package args for DDe integrator
    fixed_idx_rc = np.array([1, 2])
    rc_args = (
        Adj,
        fixed_idx_rc,
        beta_func,
        beta_args,
        alpha,
        k,
        p,
        lambda_,
        g,
        rho_0,
        S_t_delay[0],
        gamma_R,
    )
    where_S_delay = len(rc_args) - 2

    # Simulate Receiver
    R_t_rc = lsig.integrate_DDE_varargs(
        t,
        rhs=lsig.receiver_rhs,
        var_vals=[S_t_delay],
        where_vars=[where_S_delay],
        dde_args=rc_args,
        E0=S0,
        delay=delay,
        varargs_type="list",
    )

    # Combine data
    E_t = np.zeros((nt, 3), dtype=np.float32)
    E_t[:, sender_idx] = S_t_tc[:, sender_idx]
    E_t[:, receiver_idx] = R_t_rc[:, receiver_idx]
    E_t[:, transceiver_idx] = S_t_tc[:, transceiver_idx]

    if save:

        artifacts = []

        if ex is not None:

            # Dump data to file
            data_dump_fname = os.path.join(data_dir, "results.hdf5")

            # Dump data to an HDF5 file
            with h5py.File(data_dump_fname, "w") as f:
                f.create_dataset("t", data=t[::save_skip])
                f.create_dataset("sender_idx", data=sender_idx)
                f.create_dataset("transceiver_idx", data=transceiver_idx)
                f.create_dataset("receiver_idx", data=receiver_idx)
                f.create_dataset("E_t", data=E_t[::save_skip])

            # Add data dump to Sacred
            artifacts.append(data_dump_fname)

            # Add all artifacts to Sacred
            for _a in artifacts:
                ex.add_artifact(_a)

            # Save any source code dependencies to Sacred
            source_files = [
                "lateral_signaling.py",
            ]
            for sf in source_files:
                ex.add_source_file(sf)
