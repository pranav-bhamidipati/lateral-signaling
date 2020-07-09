#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import scipy.interpolate as snt
import biocircuits
from math import ceil
from lattice_oop import *

def ddeint_2D(
    dde_rhs,
    E0,
    t_out,
    delays,
    I_t,
    lattice,
    dde_args=(),
    n_time_points_per_step=20,
    progress_bar=False,
):
    """Solve a delay differential equation on a growing lattice of cells."""
    
    assert all([delay > 0 for delay in delays]), "Non-positive delays are not permitted."

    t0 = t_out[0]
    t_last = t_out[-1]

    # Extract shortest and longest non-zero delay parameters
    min_tau = min(delays)

    # Get graph transition matrix 
    A = lattice.transition_mtx()
    
    # Make a shorthand for RHS function
    def rhs(E, t, E_past):
        return dde_rhs(
            E,
            t,
            E_past,
            I_t=I_t,
            A=A,
            delays=delays,
            params=dde_args,
        )

    # Define a piecewise function to fetch past values of E
    time_bins = [t0]
    E_past_funcs = [lambda t, *args: E0(t, I_t=I_t, n_cells=lattice.n_cells())]

    def E_past(t):
        """Define past expression as a piecewise function."""
        bin_idx = next((i for i, t_bin in enumerate(time_bins) if t < t_bin))
        return E_past_funcs[bin_idx](t)

    # Initialize expression.
    E = E0(t0, I_t=I_t, n_cells=lattice.n_cells())

    t_dense = []
    E_dense = []
    
    # Integrate in steps of size min_tau. Stops before the last step.
    t_step = np.linspace(t0, t0 + min_tau, n_time_points_per_step + 1)
    n_steps = ceil((t_out[-1] - t0) / min_tau)
    
    iterator = range(n_steps)
    if progress_bar:
        iterator = tqdm.tqdm(iterator)
        
    for j in iterator:

        # Start the next step
        E_step = [E]

        # Perform integration
        for i, t in enumerate(t_step[:-1]):
            dE_dt = rhs(E, t, E_past)
            dt = t_step[i + 1] - t
            E = np.maximum(E + dE_dt * dt, 0)
            E_step.append(E)
        
        t_dense = t_dense + list(t_step[:-1])
        E_dense = E_dense + E_step[:-1]
        
        # Make B-spline
        E_step = np.array(E_step)
        tck = [
            [snt.splrep(t_step, E_step[:, cell, i]) for i in range(E.shape[1])]
            for cell in range(lattice.n_cells())
        ]

        # Append spline interpolation to piecewise function
        time_bins.append(t_step[-1])
        interp = lambda t, k=j + 1: np.array(
            [
                [np.maximum(snt.splev(t, tck[cell][i]), 0) for i in range(E.shape[1])]
                for cell in range(lattice.n_cells())
            ]
        )
        E_past_funcs.append(interp)

        # Get time-points for next step
        t_step += min_tau
        
        # Make the last step end at t_last
        if t_step[-1] > t_last:
            t_step = np.concatenate((t_step[t_step < t_last], (t_last,),))

    # Add data for last time-point
    t_dense = t_dense + [t_last]
    E_dense = E_dense + [E]

    # Interpolate solution for returning
    t_dense = np.array(t_dense)
    E_dense = np.array(E_dense)
    
    E_return = np.empty((len(t_out), *E.shape))
    for cell in range(E.shape[0]):
        for i in range(E.shape[1]):
            tck = snt.splrep(t_dense, E_dense[:, cell, i])
            E_return[:, cell, i] = np.maximum(snt.splev(t_out, tck), 0)

    return t_dense, E_dense, E_return


def rhs_tc_delay_cis_leak_2D(E, t, E_past, I_t, A, delays, params):

    tau = delays[0]
    alpha, k_s, p_s, delta, lambda_ = params

    # Get the signal input E_bar to each cell after a delay tau
    E_tau = E_past(t - tau)
    E_bar = np.dot(A, E_tau)

    # Evaluate Hill term
    f = E_bar ** p_s / (k_s ** p_s + (delta * E_tau) ** p_s + E_bar ** p_s)

    # Calculate change in expression
    dE_dt = lambda_ + alpha * f - E
    dE_dt[0, :] = I_t(t) - E[0, :]

    return dE_dt


def E0(t, I_t, n_cells): 
    E = np.zeros((n_cells, 1), dtype=float)
    E[0, :] = I_t(t)
    return E


alpha_space = np.linspace(0.5, 5, 10)
k_s_space = np.logspace(-5, -0.5, 10)
p_s_space = 2
delta_space = np.linspace(0, 5, 6)
lambda_space = np.logspace(-6, -1.5, 10)


param_space = np.array(np.meshgrid(
    alpha_space, 
    k_s_space, 
    p_s_space,
    delta_space,
    lambda_space,
)).T.reshape(-1,5)

param_space = param_space[(param_space[:, 1] / np.sqrt(param_space[:, 4]) > 2).nonzero()[0], :]


lax2d = Regular2DLattice(R = 10)
t_results = np.linspace(0, 8, 201)

tau = 0.4
delays = (tau,)

axis_cells = np.argwhere((lax2d.X[:, 1] == 0) & (lax2d.X[:, 0] >= 0)).flatten()
n_axis_cells = axis_cells.size


def simulate(params):
    _, __, results = ddeint_2D(
        dde_rhs=rhs_tc_delay_cis_leak_2D,
        E0=E0,
        t_out=t_results,
        delays=delays,
        I_t=I_t,
        lattice=lax2d,
        dde_args=params,
        n_time_points_per_step=100,
        progress_bar=False,
    )
    return params, results[:, axis_cells, :]


from multiprocessing import Pool
if __name__ == '__main__':
    with Pool(16) as p:
        results_list = list(p.imap_unordered(simulate, param_space[:5]))


n_samples = len(results_list)
end_signals = np.array([result[-1, :, 0] for _, result in results_list])
sampled_param_space = np.array([p for p, _ in results_list])


dfs = []
param_names = ["alpha", "k_s", "p_s", "delta", "lambda"]
delay_names = ["tau"]
delays = (0.4,)
species_names = ["expression"]

lax_data = pd.DataFrame(
    {
        "cell": np.array(
            [
                "cell_" + str(i).zfill(3)
                for i in np.tile(axis_cells, t_results.size)
            ]
        ),
        "X_coord": np.tile(np.arange(n_axis_cells), t_results.size),
        "step": np.repeat(np.arange(t_results.size), n_axis_cells),
        "time": np.repeat(t_results, n_axis_cells),
    }
)


for params, result in results_list:
    param_data = lax_data.copy()
    result = result.reshape(-1, result.shape[-1]).T
    param_data.update({sp: ex for sp, ex in zip(species_names, result)})
    
    for p, v in zip(param_names, params):
        param_data[p] = v
    
    for d, v in zip(delay_names, delays):
        param_data[d] = v
    
    dfs.append(param_data)

df = pd.concat(dfs)



import os

directory = "2020-07-09_2D_delay_data"

if not os.path.exists(directory):
    os.makedirs(directory)

df.to_csv(os.join(directory, "sender_zs_cis_delay_leak.csv"))
