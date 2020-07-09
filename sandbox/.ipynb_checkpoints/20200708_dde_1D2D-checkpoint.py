#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import scipy.spatial as sp
import scipy.interpolate as snt
import biocircuits
import tqdm
from math import ceil

import colorcet as cc
import holoviews as hv
import bokeh.io
import bokeh.plotting
import bokeh_catplot

from datetime import date

hv.extension('matplotlib')
bokeh.io.output_notebook()

colors = cc.palette.glasbey_category10
ceiling = np.vectorize(ceil)
sample_cycle = lambda cycle, size: [cycle[i] for i in ceiling(np.linspace(0, len(cycle) - 1, size))]


# In[5]:


get_ipython().run_line_magic('run', 'lattice_signaling.py')
get_ipython().run_line_magic('run', 'lattice_oop.py')


# In[6]:


get_ipython().run_line_magic('load_ext', 'blackcellmagic')


# <hr>

# In[13]:


def ddeint_1D(
    dde_rhs,
    E0,
    t_out,
    delays,
    I_t,
    n_cells,
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

    # Calculate graph adjacency matrix
    A = np.diag((1,) * (n_cells - 1), -1) + np.diag((1,) * (n_cells - 1), 1)
    
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
    E_past_funcs = [lambda t, *args: E0(t, I_t=I_t, n_cells=n_cells)]

    def E_past(t):
        """Define past expression as a piecewise function."""
        bin_idx = next((i for i, t_bin in enumerate(time_bins) if t < t_bin))
        return E_past_funcs[bin_idx](t)

    # Initialize expression.
    E = E0(t0, I_t=I_t, n_cells=n_cells)

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
            for cell in range(n_cells)
        ]

        # Append spline interpolation to piecewise function
        time_bins.append(t_step[-1])
        interp = lambda t, k=j + 1: np.array(
            [
                [np.maximum(snt.splev(t, tck[cell][i]), 0) for i in range(E.shape[1])]
                for cell in range(n_cells)
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


# In[14]:


def rhs_tc_delay_cis_1D(E, t, E_past, I_t, A, delays, params):
    
    tau = delays[0]
    alpha, k_s, p_s, delta = params
    
    # Get the signal input E_bar to each cell after a delay tau
    E_tau = E_past(t - tau)
    E_bar = np.dot(A, E_tau)

    # Evaluate Hill term with cis-inhibition
    f = biocircuits.reg.act_hill(E_bar / (k_s + delta * E_tau), p_s)
    
    # Calculate change in expression
    dE_dt = alpha * f - E
    dE_dt[0, :] = I_t(t) - E[0, :]
        
    return dE_dt


# In[15]:


def E0(t, I_t, n_cells): 
    E = np.zeros((n_cells, 1))
    E[0, :] = I_t(t)
    return E


# In[16]:


alpha = 2
k_s = 0.001
p_s = 2
delta = 10
tau = 0.4

params = alpha, k_s, p_s, delta
delays = (tau,)


# In[17]:


t_results = np.linspace(0, 8, 200)
n_cells=10

t_dense, E_dense, results = ddeint_1D(
    dde_rhs=rhs_tc_delay_cis_1D,
    E0=E0,
    t_out=t_results,
    delays=delays,
    I_t=I_t,
    n_cells=n_cells,
    dde_args=params,
    n_time_points_per_step=200,
    progress_bar=False,
)


# In[18]:


data = pd.DataFrame(
    {
        "step": np.repeat(np.arange(t_results.size), n_cells),
        "time": np.repeat(t_results, n_cells),
        "cell": np.array(
            [
                "cell_" + str(i).zfill(3)
                for i in np.tile(np.arange(n_cells), t_results.size)
            ]
        ),
        "expression": results.flatten(),
    }
)


# In[19]:


color_cycle = hv.Cycle(['black'] + cc.fire[::25])

plt = hv.Curve(
    data=data,
    kdims=['time'],
    vdims=["expression", "cell"]
).groupby(
    "cell"
).opts(
    hv.opts.Curve(color=color_cycle),
).overlay().opts(legend_position="right", )

hv.output(plt, dpi=150)


# Do a quantification of this somehow? Or make more grid-plots.

# <hr>

# In[20]:


get_ipython().run_line_magic('run', 'lattice_oop.py')


# In[21]:


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


# In[22]:


lax2d = Regular2DLattice(R = 10)


# In[23]:


def rhs_tc_delay_cis_2D(E, t, E_past, I_t, A, delays, params):
    
    tau = delays[0]
    alpha, k_s, p_s, delta = params
    
    # Get the signal input E_bar to each cell after a delay tau
    E_tau = E_past(t - tau)
    E_bar = np.dot(A, E_tau)

    # Evaluate Hill term with cis-inhibition
    f = biocircuits.reg.act_hill(E_bar / (k_s + delta * E_tau), p_s)
    
    # Calculate change in expression
    dE_dt = alpha * f - E
    dE_dt[0, :] = I_t(t) - E[0, :]
        
    return dE_dt


# In[24]:


def E0(t, I_t, n_cells): 
    E = np.zeros((n_cells, 1))
    E[0, :] = I_t(t)
    return E


# In[25]:


alpha = 2
k_s = 0.001
p_s = 2
delta = 10
tau = 0.4

params = alpha, k_s, p_s, delta
delays = (tau,)


# In[26]:


t_results = np.linspace(0, 8, 200)

t_dense, E_dense, results = ddeint_2D(
    dde_rhs=rhs_tc_delay_cis_2D,
    E0=E0,
    t_out=t_results,
    delays=delays,
    I_t=I_t,
    lattice=lax2d,
    dde_args=params,
    n_time_points_per_step=200,
    progress_bar=True,
)


# In[27]:


axis_cells = np.argwhere((lax2d.X[:, 1] == 0) & (lax2d.X[:, 0] >= 0)).flatten()
axis_results = results[:, axis_cells, :]

data2d = pd.DataFrame(
    {
        "step": np.repeat(np.arange(t_results.size), axis_cells.size),
        "time": np.repeat(t_results, axis_cells.size),
        "cell": np.array(
            [
                "cell_" + str(i).zfill(3)
                for i in np.tile(axis_cells, t_results.size)
            ]
        ),
        "expression": axis_results.flatten(),
    }
)


# In[28]:


color_cycle = hv.Cycle(['black'] + cc.fire[::25])

plt = hv.Curve(
    data=data2d,
    kdims=['time'],
    vdims=["expression", "cell"]
).groupby(
    "cell"
).opts(
    hv.opts.Curve(color=color_cycle),
).overlay().opts(legend_position="right", )

hv.output(plt, dpi=150)


# Very. Cool.

# In[29]:


alpha = 2
k_s = 0.001
p_s = 2
delta = 2
tau = 0.4

params = alpha, k_s, p_s, delta
delays = (tau,)


# In[30]:


t_results = np.linspace(0, 8, 200)

t_dense, E_dense, results = ddeint_2D(
    dde_rhs=rhs_tc_delay_cis_2D,
    E0=E0,
    t_out=t_results,
    delays=delays,
    I_t=I_t,
    lattice=lax2d,
    dde_args=params,
    n_time_points_per_step=100,
    progress_bar=True,
)


# In[31]:


axis_cells = np.argwhere((lax2d.X[:, 1] == 0) & (lax2d.X[:, 0] >= 0)).flatten()
axis_results = results[:, axis_cells, :]

data2d = pd.DataFrame(
    {
        "step": np.repeat(np.arange(t_results.size), axis_cells.size),
        "time": np.repeat(t_results, axis_cells.size),
        "cell": np.array(
            [
                "cell_" + str(i).zfill(3)
                for i in np.tile(axis_cells, t_results.size)
            ]
        ),
        "expression": axis_results.flatten(),
    }
)


# In[32]:


color_cycle = hv.Cycle(['black'] + cc.fire[::25])

plt = hv.Curve(
    data=data2d,
    kdims=['time'],
    vdims=["expression", "cell"]
).groupby(
    "cell"
).opts(
    hv.opts.Curve(color=color_cycle),
).overlay().opts(legend_position="right", )

hv.output(plt, dpi=150)


# In[33]:


def rhs_tc_delay_cis_2D_nosender(E, t, E_past, I_t, A, delays, params):
    
    tau = delays[0]
    alpha, k_s, p_s, delta = params
    
    # Get the signal input E_bar to each cell after a delay tau
    E_tau = E_past(t - tau)
    E_bar = np.dot(A, E_tau)

    # Evaluate Hill term with cis-inhibition
    f = biocircuits.reg.act_hill(E_bar / (k_s + delta * E_tau), p_s)
    
    # Calculate change in expression
    dE_dt = alpha * f - E
#     dE_dt[0, :] = I_t(t) - E[0, :]
        
    return dE_dt


# In[34]:


alpha = 2
k_s = 0.05
p_s = 2
delta = 2
tau = 0.4

params = alpha, k_s, p_s, delta
delays = (tau,)


# In[35]:


def E0(t, I_t, n_cells): 
    E = np.zeros((n_cells, 1)) + 1e-4
#     E[0, :] = I_t(t)
    return E


# In[36]:


t_dense, E_dense, results = ddeint_2D(
    dde_rhs=rhs_tc_delay_cis_2D_nosender,
    E0=E0,
    t_out=t_results,
    delays=delays,
    I_t=lambda t: 0,
    lattice=lax2d,
    dde_args=params,
    n_time_points_per_step=200,
    progress_bar=True,
)


# In[37]:


axis_cells = np.argwhere((lax2d.X[:, 1] == 0) & (lax2d.X[:, 0] >= 0)).flatten()
axis_results = results[:, axis_cells, :]

data2d = pd.DataFrame(
    {
        "step": np.repeat(np.arange(t_results.size), axis_cells.size),
        "time": np.repeat(t_results, axis_cells.size),
        "cell": np.array(
            [
                "cell_" + str(i).zfill(3)
                for i in np.tile(axis_cells, t_results.size)
            ]
        ),
        "expression": axis_results.flatten(),
    }
)


# In[38]:


color_cycle = hv.Cycle(['black'] + cc.fire[::25])

plt = hv.Curve(
    data=data2d,
    kdims=['time'],
    vdims=["expression", "cell"]
).groupby(
    "cell"
).opts(
    hv.opts.Curve(color=color_cycle),
).overlay().opts(legend_position="right", )

hv.output(plt, dpi=150)


# What if we add leakiness? Leakiness can be parameterized by a parameter `lambda_`

# In[39]:


def rhs_tc_delay_cis_leak_2D_nosender(E, t, E_past, I_t, A, delays, params):

    tau = delays[0]
    alpha, k_s, p_s, delta, lambda_ = params

    # Get the signal input E_bar to each cell after a delay tau
    E_tau = E_past(t - tau)
    E_bar = np.dot(A, E_tau)

    # Evaluate Hill term
    f = E_bar ** p_s / (k_s ** p_s + (delta * E_tau) ** p_s + E_bar ** p_s)

    # Calculate change in expression
    dE_dt = lambda_ + alpha * f - E
    #     dE_dt[0, :] = I_t(t) - E[0, :]

    return dE_dt


# In[40]:


alpha = 2
k_s = 0.1
p_s = 5
delta = 10
lambda_ = 0.01
tau = 0.4

params = alpha, k_s, p_s, delta, lambda_
delays = (tau,)


# In[41]:


def E0(t, I_t, n_cells): 
    E = np.zeros((n_cells, 1)) + lambda_
#     E[0, :] = I_t(t)
    return E


# In[42]:


t_results = np.linspace(0, 8, 200)

t_dense, E_dense, results = ddeint_2D(
    dde_rhs=rhs_tc_delay_cis_leak_2D_nosender,
    E0=E0,
    t_out=t_results,
    delays=delays,
    I_t=I_t,
    lattice=lax2d,
    dde_args=params,
    n_time_points_per_step=200,
    progress_bar=True,
)


# In[43]:


axis_cells = np.argwhere((lax2d.X[:, 1] == 0) & (lax2d.X[:, 0] >= 0)).flatten()
n_axis_cells = axis_cells.size
axis_results = results[:, axis_cells, :]

data2d = pd.DataFrame(
    {
        "step": np.repeat(np.arange(t_results.size), axis_cells.size),
        "time": np.repeat(t_results, axis_cells.size),
        "cell": np.array(
            [
                "cell_" + str(i).zfill(3)
                for i in np.tile(axis_cells, t_results.size)
            ]
        ),
        "expression": axis_results.flatten(),
    }
)


# In[44]:


axis_results[8, :, 0]


# In[45]:


color_cycle = hv.Cycle(sample_cycle(cc.fire[:-5], n_axis_cells))

plt = hv.Curve(
    data=data2d,
    kdims=['time'],
    vdims=["expression", "cell"]
).groupby(
    "cell"
).opts(
    hv.opts.Curve(color=color_cycle, alpha=0.5),
).overlay().opts(legend_position="right", )

hv.output(plt, dpi=150)


# With the addition of leakiness, the initial condition now significantly affects whether the system will self-activate in the absence of a sender.

# In[46]:


def rhs_tc_delay_cis_leak_2D(E, t, E_past, I_t, A, delays, params):

    tau = delays[0]
    alpha, k_s, p_s, delta, lambda_ = params

    # Get the signal input E_bar to each cell after a delay tau
    E_tau = E_past(t - tau)
    E_bar = np.dot(A, E_tau)

    # Evaluate Hill term
    f = E_bar ** p_s / (k_s + (delta * E_tau) ** p_s + E_bar ** p_s)

    # Calculate change in expression
    dE_dt = lambda_ + alpha * f - E
    dE_dt[0, :] = I_t(t) - E[0, :]

    return dE_dt


# In[47]:


alpha = 2
k_s = 0.1
p_s = 5
delta = 100
lambda_ = 0.01
tau = 0.5

params = alpha, k_s, p_s, delta, lambda_
delays = (tau,)


# In[48]:


def E0(t, I_t, n_cells): 
    E = np.zeros((n_cells, 1)) + 1e-4
    E[0, :] = I_t(t)
    return E


# In[49]:


t_results = np.linspace(0, 8, 200)

t_dense, E_dense, results = ddeint_2D(
    dde_rhs=rhs_tc_delay_cis_leak_2D,
    E0=E0,
    t_out=t_results,
    delays=delays,
    I_t=I_t,
    lattice=lax2d,
    dde_args=params,
    n_time_points_per_step=100,
    progress_bar=True,
)


# In[50]:


axis_cells = np.argwhere((lax2d.X[:, 1] == 0) & (lax2d.X[:, 0] >= 0)).flatten()
n_axis_cells = axis_cells.size
axis_results = results[:, axis_cells, :]

data2d = pd.DataFrame(
    {
        "step": np.repeat(np.arange(t_results.size), axis_cells.size),
        "time": np.repeat(t_results, axis_cells.size),
        "cell": np.array(
            [
                "cell_" + str(i).zfill(3)
                for i in np.tile(axis_cells, t_results.size)
            ]
        ),
        "expression": axis_results.flatten(),
    }
)


# In[51]:


color_cycle = hv.Cycle(sample_cycle(cc.fire[:-5], n_axis_cells))

plt = hv.Curve(
    data=data2d,
    kdims=['time'],
    vdims=["expression", "cell"]
).groupby(
    "cell"
).opts(
    hv.opts.Curve(color=color_cycle),
).overlay().opts(legend_position="right", )

hv.output(plt, dpi=150)


# 

# What are my behavior criteria/categories?
# 
# 1) In the absence of sender, does the lattice self-activate? Is it "robust" to self-activation?
# 
# 2) In the presence of sender, how does it behave?
# - No activation (at SS, last cell is close to its no-sender steady state)
# - Activation with or without "boosted" dynamics. Increases to a high steady state. Could be monotonic, overshoot, overdamped, oscillatory, etc.
# - Imperfect activation (overshoot to middle steady-state)
# - Pulsed activation (overshoot to low steady-state)

# What are my plausible parameter ranges?
# 
# - $\alpha$ : 0.5 to 5 (lin-space?)
# - $k_s$ : 1e-6 to 2e-1 (log-space)
# - $p_s$ : 1 to 5 (but 2 is fine)
# - $\delta$ : 0 to 10 (technically, 0 to inf, but above 10, maybe diminishing returns)
# - $\lambda$ : 0 to 2e-2 (must be much lower than $k_s$ to ensure bistability)
# - $\tau$ : 0 to 0.7 (cell cycle time)
# 
# First, I want to see where the transceiver system is stable in the absence of senders. I'll fix the following parameters:
# 
# - $p_s$ : 2
# - $\delta$ : 0 (worst-case)
# - $\tau$ : 0.4
# 
# The free parameters will be:
# 
# - $\alpha$ : 0.5 to 5 (lin-space?)
# - $k_s$ : 1e-6 to 2e-1 (log-space)
# - $\lambda$ : 1e-7 to 2e-2 (log-space-ish, must be much less than $k_s$)
# 

# In[52]:


alpha_space = np.linspace(0.5, 5, 10)
k_s_space = np.logspace(-6.0, -0.6, 6)
p_s_space = 2
delta_space = 0
lambda_space = np.logspace(-7, -1.6, 6)
tau_space = 0.4


# In[53]:


param_space = np.array(np.meshgrid(
    alpha_space, 
    k_s_space, 
    p_s_space,
    delta_space,
    lambda_space,
)).T.reshape(-1,5)

param_space


# In[69]:


def rhs_tc_delay_cis_leak_2D_nosender(E, t, E_past, I_t, A, delays, params):

    tau = delays[0]
    alpha, k_s, p_s, delta, lambda_ = params

    # Get the signal input E_bar to each cell after a delay tau
    E_tau = E_past(t - tau)
    E_bar = np.dot(A, E_tau)

    # Evaluate Hill term
    f = E_bar ** p_s / (k_s ** p_s + (delta * E_tau) ** p_s + E_bar ** p_s)

    # Calculate change in expression
    dE_dt = lambda_ + alpha * f - E
    #     dE_dt[0, :] = I_t(t) - E[0, :]

    return dE_dt


# In[70]:


def E0(t, I_t, n_cells): 
    E = np.zeros((n_cells, 1), dtype=float)
#     E[0, :] = I_t(t)
    return E


# In[71]:


lax2d = Regular2DLattice(R = 10)
t_results = np.linspace(0, 8, 201)

tau = 0.4
delays = (tau,)

axis_cells = np.argwhere((lax2d.X[:, 1] == 0) & (lax2d.X[:, 0] >= 0)).flatten()
n_axis_cells = axis_cells.size


# In[85]:


results_list = []
iterator = tqdm.tqdm(param_space)

for params in iterator:
    if (params[4] * 10 > params[1]):
        continue
    
    t_dense, E_dense, results = ddeint_2D(
        dde_rhs=rhs_tc_delay_cis_leak_2D_nosender,
        E0=E0,
        t_out=t_results,
        delays=delays,
        I_t=I_t,
        lattice=lax2d,
        dde_args=params,
        n_time_points_per_step=100,
        progress_bar=False,
    )
    
    axis_results = results[:, axis_cells, :]
    
    results_list.append((params, axis_results))


# In[100]:


n_samples = len(results_list)
end_signals = np.array([result[-1, :, 0] for _, result in results_list])
sampled_param_space = np.array([p for p, _ in results_list])


# In[172]:


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


# In[178]:


for params, result in results_list:
    
    param_data = lax_data.copy()
    result = result.reshape(-1, result.shape[-1]).T
    param_data.update({sp: ex for sp, ex in zip(species_names, result)})
    
    for p, v in zip(param_names, params):
        param_data[p] = v
    
    for d, v in zip(delay_names, delays):
        param_data[d] = v
    
    dfs.append(param_data)


# In[179]:


df = pd.concat(dfs)


# In[183]:


# df.to_csv("2020-07-09_2D_delay_data/nosender_zs_cis_delay_leak.csv", )


# In[129]:


# end_df = {'cell' = np.tile(["cell_" + str(i).zfill(3)])}
end_data = pd.DataFrame(
    {
        "cell": np.array(
            [
                "cell_" + str(i).zfill(3)
                for i in np.tile(axis_cells, n_samples)
            ]
        ),
        "expression": end_signals.flatten(),
        "sample": np.repeat(np.arange(n_samples), n_axis_cells),
        "alpha": np.repeat(sampled_param_space.T[0], n_axis_cells),
        "k_s": np.repeat(sampled_param_space.T[1], n_axis_cells),
        "lambda": np.repeat(sampled_param_space.T[4], n_axis_cells),
    }
)

end_data["norm. expression"] = end_data["expression"] / end_data["alpha"]
end_data["k_s / lambda"] = end_data["k_s"] / end_data["lambda"]

end_data["k_s / sqrt(lambda)"] = end_data["k_s"] / np.sqrt(end_data["lambda"])


# In[120]:


plt = bokeh_catplot.ecdf(
    data=end_data,
    val="expression",
    cats=["cell"],
    width=600,
    height=600,
)
plt.legend.location = "bottom_right"

bokeh.io.show(plt)


# In[121]:


plt = bokeh_catplot.ecdf(
    data=end_data,
    val="expression",
    cats=["alpha"],
    width=600,
    height=600,
)
plt.legend.location = "bottom_right"

bokeh.io.show(plt)


# In[123]:


plt = bokeh_catplot.ecdf(
    data=end_data,
    val="norm. expression",
    cats=["alpha"],
    width=600,
    height=600,
)
plt.legend.location = "bottom_right"

bokeh.io.show(plt)


# In[188]:


plt = hv.Points(
#     data=end_data,
    data=end_data.loc[end_data["cell"] == "cell_"+str(axis_cells[0]).zfill(3), :],
    kdims=["k_s", "lambda"],
    vdims=["norm. expression", "alpha"]
).groupby(
    "alpha"
).opts(
    logx = True,
    logy = True,
    cmap = "viridis",
    color="norm. expression",
    colorbar = True,
).layout().cols(1)

hv.output(plt, dpi=150)


# In[190]:


plt = hv.Scatter(
    data=end_data,
    kdims=["k_s / sqrt(lambda)"],
    vdims=["norm. expression", "alpha", "cell"]
).groupby(
    "alpha"
).opts(
    logx = True,
    cmap = cc.palette.bjy[::25],
    color="cell",
    colorbar = True,
    alpha=0.25
).layout().cols(1)

hv.output(plt, dpi=150)


# There is a relationship here!! For a non-self-inducing lattice, we could enforce $\frac{k_s}{\sqrt\lambda} > 2$ and be right most of the time.
# 
# This is pretty restrictive - good, because it will help us restrict parameter search, but also not great because it presents design challenges. I would like to re-run this for different values of delta, since that will help us understand how cis-inhibition could act as a design principle that helps patterning.

# So now, let's return to our plausible parameter ranges
# 
# - $\alpha$ : 0.5 to 5 (lin-space?)
# - $k_s$ : 1e-6 to 2e-1 (log-space)
# - $p_s$ : 1 to 5 (but 2 is fine)
# - $\delta$ : 0 to 10 (technically, 0 to inf, but above 10, maybe diminishing returns)
# - $\lambda$ : 0 to 2e-2 ()
# - $\tau$ : 0 to 0.7 (cell cycle time)

# Now, I want to interrogate behavior in general. I'll fix the following parameters.
# 
# - $p_s$ : 2
# - $\tau$ : 0.4
# 
# The free parameters will be:
# 
# - $\alpha$ : 0.5 to 5 (lin-space)
# - $k_s$ : 1e-6 to 2e-1 (log-space)
# - $\delta$ : 0 to 10 (linear)
# - $\lambda$ : 1e-7 to 2e-2 (log-space-ish, must be much less than $k_s$)
# 
# I'll add the additional constraint: $\frac{k_s}{\sqrt\lambda} > 2$

# In[211]:


alpha_space = np.linspace(0.5, 5, 10)
k_s_space = np.logspace(-5, -0.5, 10)
p_s_space = 2
delta_space = np.linspace(0, 5, 6)
lambda_space = np.logspace(-6, -1.5, 10)


# In[212]:


param_space = np.array(np.meshgrid(
    alpha_space, 
    k_s_space, 
    p_s_space,
    delta_space,
    lambda_space,
)).T.reshape(-1,5)

param_space = param_space[(param_space[:, 1] / np.sqrt(param_space[:, 4]) > 2).nonzero()[0], :]


# In[213]:


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


# In[214]:


def E0(t, I_t, n_cells): 
    E = np.zeros((n_cells, 1), dtype=float)
    E[0, :] = I_t(t)
    return E


# In[215]:


lax2d = Regular2DLattice(R = 10)
t_results = np.linspace(0, 8, 201)

tau = 0.4
delays = (tau,)

axis_cells = np.argwhere((lax2d.X[:, 1] == 0) & (lax2d.X[:, 0] >= 0)).flatten()
n_axis_cells = axis_cells.size


# In[ ]:


results_list = []
iterator = tqdm.tqdm(param_space)

for params in iterator:
    _, __, results = ddeint_2D(
        dde_rhs=rhs_tc_delay_cis_leak_2D_nosender,
        E0=E0,
        t_out=t_results,
        delays=delays,
        I_t=I_t,
        lattice=lax2d,
        dde_args=params,
        n_time_points_per_step=100,
        progress_bar=False,
    )
    
    axis_results = results[:, axis_cells, :]
    
    results_list.append((params, axis_results))


# In[ ]:





# In[ ]:





# In[100]:


n_samples = len(results_list)
end_signals = np.array([result[-1, :, 0] for _, result in results_list])
sampled_param_space = np.array([p for p, _ in results_list])


# In[172]:


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


# In[178]:


for params, result in results_list:
    
    param_data = lax_data.copy()
    result = result.reshape(-1, result.shape[-1]).T
    param_data.update({sp: ex for sp, ex in zip(species_names, result)})
    
    for p, v in zip(param_names, params):
        param_data[p] = v
    
    for d, v in zip(delay_names, delays):
        param_data[d] = v
    
    dfs.append(param_data)


# In[179]:


df = pd.concat(dfs)


# In[183]:


# df.to_csv("2020-07-09_2D_delay_data/nosender_zs_cis_delay_leak.csv", )


# In[ ]:





# In[ ]:




