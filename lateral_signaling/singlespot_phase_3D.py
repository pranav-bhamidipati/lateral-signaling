#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from glob import glob
import json
import h5py

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as rotation
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

import lateral_signaling as lsig


# In[3]:


# %load_ext blackcellmagic
# %matplotlib inline


# <hr>

# In[4]:


data_dir     = os.path.abspath("../data/simulations/20211201_singlespotphase/sacred")
save_dir     = os.path.abspath("../plots")
thresh_fpath = os.path.abspath("../data/simulations/phase_threshold.json")

save_figs = True
fig_fmt   = "png"
dpi       = 300


# ---

# __Get threshold for $\mathit{v}_{\text{init}}$__

# In[5]:


with open(thresh_fpath, "r") as f:
    threshs = json.load(f)
    v_init_thresh = float(threshs["v_init_thresh"])


# __Read in phase metric data__

# In[6]:


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


# In[7]:


# Concatenate into one dataset
df = pd.concat(dfs).reset_index(drop=True)
nrow = df.shape[0]

# Assign phases and corresponding plot colors
df["phase"] = (df.v_init > v_init_thresh).astype(int) * (1 + (df.n_act_fin > 0).astype(int))
df["color"] = np.array(lsig.cols_blue)[df.phase]


# ---

# __View distributions of phase metrics__

# In[8]:


# hv.Histogram(df.v_init.values)
fig, ax = plt.subplots()

# Histogram
plt.hist(df.v_init, bins=50, color="k", density=True);

# Threshold used
plt.vlines(v_init_thresh, *plt.gca().get_ylim(), color="k", linestyles="dashed")

plt.xlabel(r"$v_{init}$", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.tick_params(labelsize=12)

plt.tight_layout()

if save_figs:
    fname = "v_init_histogram"
    fpath = os.path.join(save_dir, fname + "." + fig_fmt)
    plt.savefig(fpath, dpi=dpi)


# In[9]:


# hv.Histogram(df.v_init.values)
fig = plt.figure()
gs  = fig.add_gridspec(1, 3)

# ax0 = fig.add_subplot(gs[0, 0])
# plt.axis("off")

ax1 = fig.add_subplot(gs[0, 0])

pct_off = (df.n_act_fin == 0).sum() / df.shape[0]

plt.fill([-1, -1, 1, 1], [0,       1,       1, 0], lsig.col_gray)
plt.fill([-1, -1, 1, 1], [0, pct_off, pct_off, 0], "k"          )
plt.title(r"  $n_{\mathrm{act, fin}}$ = 0", loc="left",   fontsize=16, y=-0.075)
plt.title(f"{pct_off:.1%}\n",               loc="center", fontsize=14, y=pct_off-0.05)
plt.xlim(-2, 2)
plt.axis("off")

ax2 = fig.add_subplot(gs[:, 1:])

n_act_fin = df.n_act_fin[df.n_act_fin > 0].values

nbins = 50
bins = np.geomspace(1, 6400, nbins + 1)

# Histogram
# plt.hist(logn_act_fin, bins=500, color="k", density=True);
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


# ---

# __Plot phase boundaries in 3D__

# In[78]:


from itertools import islice

def window(seq, n):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def get_phase_boundary_idx(
    phase_grid, 
    phase1, 
    phase2, 
):
    """
    Given a `(n1 x n2 x n3 `Numpy array containing the phase at each point in a gridded 
    3D parameter space, find the points at which there's a boundary between `phase1` 
    and `phase2`. Uses a ray tracing-esque method that starts with the first axis, iterates 
    over each grid line parallel to that axis, finds the first boundary (`phase1`->`phase2` 
    or `phase2`->`phase1`), and repeats this for each spatial axis. 
    
    Returns a 3-tuple of lists `bound`. Each item in `bound[axis]` is the index of the 
    boundary found in `griddata`.
    """
    
    # Get number of sampled points along each axis
    shape = phase_grid.shape
    
    # Initialize output
    bounds = ([], [], [])
    
    # Iterate over axes
    for axis in range(3):
        
        # Get the i and j axes (indices on the plane
        #    perpendicular to the current axis)
        _i_axis = (axis + 1) % 3
        _j_axis = (axis + 2) % 3
        
        # Iterate over all rays parallel to the axis
        _slice = list(np.s_[:, :, :])
        for i in range(shape[_i_axis]):
            _slice[_i_axis] = i
            for j in range(shape[_j_axis]):
                _slice[_j_axis] = j
                _slice[axis] = slice(None, None, None)
                
                # Make a generator object to iterate along the ray
                gen = window(phase_grid[tuple(_slice)], 2)
                
                # Search for the first phase boundary
                try:
                    idx = next(
                        i for i, el in enumerate(gen) 
                        if (
                            el == (phase1, phase2)
                        ) or (
                            el == (phase2, phase1)
                        )
                    )
                
                # If not found, no worries
                except StopIteration:
                    continue
                
                # If found, add the indices to bounds in correct order
                else:
                    b = tuple(np.roll((idx, i, j), shift=axis))
                    bounds[axis].append(b)
        
    return bounds

def get_phase_boundary_pts(bounds, *grid_axes):
    
    # Make a new bounds object on the other side of the boundary
    bounds1 = bounds
    bounds2 = ([], [], [])
    for axis, (b1, b2) in enumerate(zip(bounds1, bounds2)):
        if b1:
            b2 = np.asarray(b1, dtype=int)
            b2[:, axis] += 1
            bounds2[axis][:] = [tuple(row) for row in b2]

    # Concatenate indices 
    b1_cat = np.concatenate([b for b in bounds1 if b]).T
    b2_cat = np.concatenate([b for b in bounds2 if b]).T

    # Get points on either side of boundary
    pts1 = np.array([grid_axes[i][b] for i, b in enumerate(b1_cat)]).T
    pts2 = np.array([grid_axes[i][b] for i, b in enumerate(b2_cat)]).T

    # Return midpoints as estimates of boundary
    return (pts1 + pts2) / 2


def plot_phase_boundaries_3D(
    ax3d,
    data,
    xyz,
    phase_pairs,
    phase_col="phase",
    alpha = 0.9,
    azim=15,
    elev=18,
    max_edge_length=0.9,
    zbias=1e-3,
    **kw
):
    
    # Unpack axis variables
    x, y, z = xyz
    
    # Get which coordinates were sampled along each axis
    grid_axes = (
        np.unique(df[x]),
        np.unique(df[y]),
        np.unique(df[z]),
    )

    # Get phase values in terms of x/y/z axes
    dgrid = data[
        [x, y, z, phase_col]
    ].pivot(
        index=[x, y], 
        columns=z, 
        values=phase_col
    )
    
    # Turn into (n_x x n_y x n_z) array containing the 
    #   values at each point in grid.
    dgrid = np.array(list(
        dgrid.groupby(x).apply(pd.DataFrame.to_numpy)
    ))
    
    # Set up plot
    ax3d.set(**kw)
    ax3d.azim = azim
    ax3d.elev = elev

    for i, (p1, p2) in enumerate(phase_pairs):
        
        # Get boundary indices from gridded data
        bounds    = get_phase_boundary_idx(dgrid, p1, p2)

        # Get boundary points
        bound_pts = get_phase_boundary_pts(bounds, *grid_axes)

        # Optionally, rotate boundary points before triangulation.
        # NOTE: 
        #   Under the hood, MPL does a projection down to 2D to make triangulations.
        #   This produces problems with manifolds perpendicular to XY plane. Hence,
        #   rotation before triangulation can improve the quality of the projection.
        #   I tried random rotations until one produced a good triangulation.
        if i in rot_idx:
            
            # Find which rotation vector to use
            j = np.nonzero(np.asarray(rot_idx) == i)[0][0]
            
            # Rotate points
            R = rotation.from_rotvec(rot_vecs[j])
            bp = R.apply(bound_pts)
            
        else:
            bp = bound_pts

        # Get triangulation
        _triangles     = mpl.tri.Triangulation(
            bp[:, 0], bp[:, 1]
        ).triangles

        # Remove triangles with long edges
        edges = bound_pts[_triangles] - bound_pts[np.roll(_triangles, 1, axis=1)]
        edge_lengths = np.linalg.norm(edges, axis=2)
        _mask = np.any(edge_lengths > max_edge_length, axis=1)

        # Use this triangulation and mask and supply points
        _triangulation = mpl.tri.Triangulation(
            bound_pts[:, 0],
            bound_pts[:, 1], 
            triangles=_triangles, 
            mask=_mask,
        )
        
        # Plot two surfaces, one for each side of the boundary, colored appropriately
        # Z must be provided because the triangulation object only preserves X and Y
        ax3d.plot_trisurf(
            _triangulation, 
            Z=bound_pts[:, 2],
            alpha=alpha,
            color=phase_colors[p1],
        )

        ax3d.plot_trisurf(
            _triangulation, 
            Z=bound_pts[:, 2] - zbias,
            alpha=alpha,
            color=phase_colors[p2],
        )


# In[117]:


# The XYZ axes of the phase diagram
xyz = ["g", "rho_max", "rho_0"]

# Phase pairs to plot
phase_pairs  = [
    (0, 1), 
    (1, 2), 
    (0, 2),
]

# Colors for phase regions
phase_colors = lsig.cols_blue[::-1]

# Rotation vectors - optionally used for better triangulation
#   when a part of the phase boundary is orthogonal to XY plane
rot_idx  = (1,) 
rot_vecs = ([ 1.3, -0.5,  0.2],)

# Maximum allowed edge length in a triangulation
max_edge_length = 0.9

# Additional axis options
axis_kw = dict(
    xlim3d = [0,  2.5],
    ylim3d = [0, 6.25],
    zlim3d = [0, 6.25],
    xlabel = r"$g$",
    ylabel = r"$\rho_{max}$",
    zlabel = r"$\rho_0$",
    xticks = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
    yticks = [0, 2, 4, 6],
    zticks = [0, 2, 4, 6],
)

text_kw = dict(
    transform=ax.transAxes, 
    ha="center",
    fontsize=18,
    zorder=1000,
)


# In[128]:

# %matplotlib inline
# %matplotlib widget

# Adjust font sizes
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12

# Set up plot
figsize = (8, 8)
fig, ax = plt.subplots(
    figsize=figsize,
    subplot_kw=dict(projection="3d")
)
ax.zaxis.set_rotate_label(False)

# Plot phase boundaries as surfaces
plot_phase_boundaries_3D(
    ax3d=ax,
    data=df,
    xyz=xyz,
    phase_pairs=phase_pairs,
    max_edge_length=max_edge_length,
    **axis_kw
)

ax.text2D(0.225 * figsize[0], 0.350 * figsize[1], "Attenuated", c="w", **text_kw)
ax.text2D(0.200 * figsize[0], 0.225 * figsize[1],  "Unlimited", c="k", **text_kw)
ax.text2D(0.340 * figsize[0], 0.250 * figsize[1],    "Limited", c="w", **text_kw)

if save_figs:
    fname = "phase_boundaries_3D"
    fpath = os.path.join(save_dir, fname + "." + fig_fmt)
    plt.savefig(fpath, dpi=dpi)


# In[ ]:




