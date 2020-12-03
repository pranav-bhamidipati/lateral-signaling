####### Load depenendencies
import os

import numpy as np
import pandas as pd
from math import ceil, floor, log10
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
import scipy.spatial as sp
import biocircuits

import numba
import tqdm
import time

from matplotlib import animation
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection, LineCollection
from shapely.geometry import Polygon, Point
from descartes import PolygonPatch
# import shapely.geometry as geom

import holoviews as hv
import colorcet as cc
import matplotlib.pyplot as plt
hv.extension('matplotlib')

ceiling = np.vectorize(ceil)
def sample_cycle(cycle, size): 
    return hv.Cycle(
        [cycle[i] for i in ceiling(np.linspace(0, len(cycle) - 1, size))]
    )

####### Delay diff eq integration

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


def integrate_DDE_varargs(
    t_span,
    rhs,
    var_vals,
    dde_args,
    E0,
    delay,
    progress_bar=False,
    min_delay=5,
    where_vars=np.array([5]),
):
    # Get # time-points, dt, and # cells
    n_t = t_span.size
    dt = t_span[1] - t_span[0]
    n_c = E0.shape[0]
    
    # Get delay in steps
    step_delay = np.atleast_1d(delay) / dt
    assert (step_delay >= min_delay), (
        "Delay time is too short. Lower dt or lengthen delay."
    )
    step_delay = ceil(step_delay)
    
    # Initialize expression vector
    E_save = np.empty((n_t, n_c), dtype=np.float32)
    E_save[0] = E = E0
    
    # Make dde_args mutable
    dde_args = list(dde_args)
    
    # Construct time iterator
    iterator = np.arange(1, n_t)
    if progress_bar:
        iterator = tqdm.tqdm(iterator)

    for step in iterator:
        # Get past E
        past_step = max(0, step - step_delay)
        E_delay = E_save[past_step]
        
        # Get past variable value(s)
        v = var_vals[past_step]
        dde_args[where_vars] = v
        
#         return rhs, E, E_delay, dde_args
        
        # Integrate
        dE_dt = rhs(E, E_delay, *dde_args)
        E = np.maximum(0, E + dE_dt * dt) 
        E_save[step] = E
    
#         print("Any nan in dE_dt?", np.isnan(dE_dt).any())
#         print("Any nan in E?", np.isnan(E).any())
    
    return E_save



####### Plot/animate expression and cell-cell contacts on a hexagonal grid

@numba.njit
def normalize(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)

@numba.njit
def beta_to_rad(beta, dist=1):
    return dist/(np.sqrt(4-beta**2))

@numba.njit
def rad_to_beta(rad, dist=1):
    return np.sqrt(4 - (dist**2 / rad**2))

def make_Adj(rows, cols=0, dtype=np.float32, **kwargs):
    """Construct adjacency matrix for a periodic hexagonal 
    lattice of dimensions rows x cols."""
    
    # Check if square
    if cols == 0:
        cols = rows
    
    # Initialize matrix
    n = rows * cols
    Adj = np.zeros((n,n), dtype=dtype)
    for i in range(cols):
        for j in range(rows):
            
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
            )
            
            nb[:, 0] = nb[:, 0] % cols
            nb[:, 1] = nb[:, 1] % rows
            
            # Populate Adj
            nbidx = np.array([ni*rows + nj for ni, nj in nb])
            Adj[i*rows + j, nbidx] = 1
    
    return Adj

def hex_Adj(rows, cols=0, dtype=np.float32, **kwargs):
    """
    """
    # Make hexagonal grid
    X = hex_grid(rows, cols, **kwargs)
    
    # Construct adjacency matrix
    Adj = make_Adj(rows, cols, dtype=dtype, **kwargs)
    
    return X, Adj


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def plot_var(
    ax1,
    i,
    X,
    cell_radii,
    var,
    vmin=None,
    vmax=None,
    cmap="CET_L8",
    ec=None,
    ifcc="",
    ppatch_kwargs=dict(edgecolor='gray'),
    lcoll_kwargs=dict(),
    title=None,
    axis_off=True,
    xlim=(),
    ylim=(),
    aspect=None,
    plot_ifc=True,
    **kwargs
):
    ax1.clear()
    if axis_off:
        ax1.axis("off")
    vor = Voronoi(X)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    if vmin is None:
        vmin = var.min()
    if vmax is None:
        vmax = var.max()

    cols = cc.cm[cmap](normalize(var[i], vmin, vmax))
    for j, region in enumerate(regions):
        poly = Polygon(vertices[region])
        circle = Point(X[j]).buffer(cell_radii[j])
        cell_poly = circle.intersection(poly)
        if cell_poly.area != 0:
            ax1.add_patch(PolygonPatch(cell_poly, fc=cols[j], **ppatch_kwargs))
    
    if plot_ifc:
        pts = [Point(*x).buffer(rad) for x, rad in zip(X, cell_radii)]
        Adj = make_Adj(int(np.sqrt(X.shape[0])))
        infcs = []
        for i, j in zip(*Adj.nonzero()):
            if (j - i > 0) & (pts[i].intersects(pts[j])):
                infc = pts[i].boundary.intersection(pts[j].boundary)
                infcs.append(infc)
        ax1.add_collection(LineCollection(infcs, color=ifcc, **lcoll_kwargs))
    
    if not xlim:
        xlim=[X[:, 0].min(), X[:, 0].max()]
    if not ylim:
        ylim=[X[:, 1].min(), X[:, 1].max()]
    if aspect is None:
        aspect=1
    ax1.set(
            xlim=xlim,
            ylim=ylim,
            aspect=aspect,
        )
    
    if title is not None:
        ax1.set_title(title)
    
def animate_var(
    X,
    cell_radii,    
    var,
    n_frames=100,
    file_name=None,
    dir_name="plots",
    xlim=None,
    ylim=None,
    fps=20,
    vmin=None,
    vmax=None,
    #     ec="red",
    cmap="CET_L8",
    ifcc="red",
    ppatch_kwargs=dict(edgecolor="gray"),
    lcoll_kwargs=dict(),
    title_fun=None,
    **kwargs
):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    skip = int((var.shape[0]) / n_frames)
    
    if title_fun is not None:
        tf=title_fun
    else:
        tf=lambda x: None
        
    cr = cell_radii.copy()
    if cr.ndim == 1:
        cr = np.repeat(cr[np.newaxis, :], var.shape[0])
    
    def anim(i):
        plot_var(
            ax1,
            skip * i,
            X,
            cell_radii[skip * i],
            var,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ifcc=ifcc,
            ppatch_kwargs=ppatch_kwargs,
            lcoll_kwargs=lcoll_kwargs,
            title=tf(skip * i),
            **kwargs,
        )

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if file_name is None:
        file_name = "animation_%d" % time.time()
    print("Writing to:", os.path.join(dir_name, file_name))

    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=fps, bitrate=1800)

    an = animation.FuncAnimation(fig, anim, frames=n_frames, interval=200)
    an.save("%s.mp4" % os.path.join(dir_name, file_name), writer=writer, dpi=264)


def inspect_out(
    X,
    cell_radii,
    var,
    idx=-1,
    vmin=None,
    vmax=None,
    cmap="CET_L8",
    #     ec="red",
    ifcc="red",
    ppatch_kwargs=dict(edgecolor="gray"),
    lcoll_kwargs=dict(),
    **kwargs
):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    
    if idx == -1:
        k = var.shape[0] - 1
    else:
        k = idx
        
    if len(X.shape) == 2:
        Xk = X.copy()
    else:
        Xk = X[k]
    
    if len(cell_radii.shape) == 1:
        crk = cell_radii.copy()
    else:
        crk = cell_radii[k]

#     if len(var.shape) == 1:
#         vark = var.copy()
#     else:
#         vark = var[k]

    plot_var(
        ax1,
        k,
        Xk,
        cell_radii[k],
        var,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        ifcc=ifcc,
        ppatch_kwargs=ppatch_kwargs,
        lcoll_kwargs=lcoll_kwargs,
        **kwargs
    )

    
def animate_var_lattice(
    X_arr,
    cell_radii,
    var,
    n_frames=100,
    file_name=None,
    dir_name="plots",
    xlim=None,
    ylim=None,
    fps=20,
    vmin=None,
    vmax=None,
    #     ec="red",
    cmap="CET_L8",
    ifcc="red",
    ppatch_kwargs=dict(edgecolor="gray"),
    lcoll_kwargs=dict(),
    title_fun=None,
    **kwargs
):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)

    skip = int((var.shape[0]) / n_frames)
    
    if title_fun is not None:
        tf=title_fun
    else:
        tf=lambda x: None
        
    cr = cell_radii.copy()
    if cr.ndim == 1:
        cr = np.repeat(cr[np.newaxis, :], var.shape[0], axis=0)
    
    if X_arr.ndim == 2:
        X = np.repeat(X_arr[np.newaxis, :], var.shape[0], axis=0)
    else:
        X = X_arr.copy()
    
    def anim(i):
        plot_var(
            ax1,
            skip * i,
            X[skip * i],
            cell_radii[skip * i],
            var,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ifcc=ifcc,
            ppatch_kwargs=ppatch_kwargs,
            lcoll_kwargs=lcoll_kwargs,
            title=tf(skip * i),
            **kwargs
        )

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if file_name is None:
        file_name = "animation_%d" % time.time()
    print("Writing to:", os.path.join(dir_name, file_name))

    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=fps, bitrate=1800)

    an = animation.FuncAnimation(fig, anim, frames=n_frames, interval=200)
    an.save("%s.mp4" % os.path.join(dir_name, file_name), writer=writer, dpi=264)

    
def plot_colormesh(
    ax,
    X,
    rows,
    cols,
    var,
    vmin=None,
    vmax=None,
    cmap="CET_L8",
    title=None,
    axis_off=False,
    xlim=(),
    ylim=(),
    aspect=None,
    pcolormesh_kwargs=dict(),
    **kwargs
):
    ax.clear()
    if axis_off:
        ax.axis("off")

#     cols = cc.cm[cmap](normalize(var, vmin, vmax))
    
    xx = X[:, 0].reshape(cols,rows)
    yy = X[:, 1].reshape(cols,rows)
    cols, rows = xx.shape
    
    xi, yi = np.meshgrid(np.arange(cols), np.arange(rows), indexing="ij")
    zz = var[rows*xi + yi]
    
    ax.pcolormesh(
        xx, 
        yy, 
        zz, 
        shading="gouraud", 
        cmap=cc.cm[cmap],
        vmin=vmin,
        vmax=vmax,
        **pcolormesh_kwargs
    )

    if not xlim:
        xlim=[X[:, 0].min(), X[:, 0].max()]
    if not ylim:
        ylim=[X[:, 1].min(), X[:, 1].max()]
    if aspect is None:
        aspect=1

    ax.set(
        xlim=xlim,
        ylim=ylim,
        aspect=aspect,
        **kwargs
    )

    if title is not None:
        ax.set_title(title)

def animate_colormesh(
    X_arr,
    rows,
    cols,
    var_t,
    vmin=None,
    vmax=None,
    n_frames=100,
    file_name=None,
    dir_name="plots",
    fps=20,
    cmap="CET_L8",
    title_fun=None,
    axis_off=False,
    xlim=(),
    ylim=(),
    aspect=None,
    pcolormesh_kwargs=dict(),
    **kwargs
):
    n_t = var_t.shape[0]
    
    if X_arr.ndim == 2:
        X_t = np.repeat(X_arr[np.newaxis, :], n_t, axis=0)
    else:
        X_t = X_arr.copy()
    
    if vmin is None:
        vmin = var_t.min()
    if vmax is None:
        vmax = var_t.max()
        
    if title_fun is not None:
        tf=title_fun
    else:
        tf=lambda x: None
    
    fig, ax = plt.subplots()
    skip = int(n_t / n_frames)
    def anim(i):
        plot_colormesh(
            ax,
            X_t[skip * i],
            rows,
            cols,
            var_t[skip * i],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            axis_off=axis_off,
            xlim=xlim,
            ylim=ylim,
            aspect=aspect,
            pcolormesh_kwargs=pcolormesh_kwargs,
            title=title_fun(skip * i),
            **kwargs
        )

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if file_name is None:
        file_name = "animation_%d" % time.time()
    print("Writing to:", os.path.join(dir_name, file_name))

    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=fps, bitrate=1800)

    an = animation.FuncAnimation(fig, anim, frames=n_frames, interval=200)
    an.save("%s.mp4" % os.path.join(dir_name, file_name), writer=writer, dpi=264)

def inspect_colormesh(
    X,
    rows,
    cols,
    var,
    idx=-1,
    ax=None,
    vmin=None,
    vmax=None,
    cmap="CET_L8",
    title=None,
    axis_off=False,
    xlim=(),
    ylim=(),
    aspect=None,
    pcolormesh_kwargs=dict(),
    **kwargs
):
    """
    """
    
    if X.ndim == 2:
        X_ = X.copy()
    else:
        X_ = X[idx]
    
    if var.ndim == 1:
        var_ = var.copy()
    else:
        var_ = var[idx]
    
    if vmin is None:
        vmin = var_.min()
    if vmax is None:
        vmax = var_.max()
    
    if ax is None:
        _, ax = plt.subplots()
    
    plot_colormesh(
        ax,
        X_,
        rows,
        cols,
        var_,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        axis_off=axis_off,
        xlim=xlim,
        ylim=ylim,
        aspect=aspect,
        title=title,
        pcolormesh_kwargs=pcolormesh_kwargs,
        **kwargs
    )
    
####### Hexagonal lattice



def act_vmean(t, X, E_save, thresh, chull=False):
    """
    Calculate mean velocity of activated cells on hexagonal lattice
    """
    
    # Get time difference
    dt = t[-1] - t[0]
    
    # Get cells at boundary
    if X.shape[0] < 3:
        X_where_bounds = np.array([i for i in range(X.shape[0])])
    else:
        X_where_bounds = ConvexHull(X).vertices
    
    crossed = np.argmax(E_save[:, X_where_bounds].sum(axis=1) > thresh)
    if crossed > 0:
        tr = t[:crossed].copy()
    else:
        tr = t.copy()

    # Get activated cells at first and last time
    Et0, Etlast = E_save[0] > thresh, E_save[-1] > thresh
    
    # Calculate area using convex hull volume or sum of cell areas
    if chull:
        
        # Exclude time-points with <3 points (throws error)
        if np.sum(Et0) < 3:
            a0 = 0
        else:
            a0 = ConvexHull(X[Et0]).volume
            
        if np.sum(Etlast) < 3:
            alast = 0
        else:
            alast = ConvexHull(X[Etlast]).volume
    else:
        a0 = E0.sum() * np.sqrt(3)/2
        alast = Etlast.sum() * np.sqrt(3)/2
    
    dr = np.diff(np.sqrt(np.array([a0, alast]) / np.pi))
    
    return dr/dt


def act_area_vor(X, E, thresh):
    """
    Calculate area of activated cells using Voronoi mesh
    """
    vor = Voronoi(X)
    areas = voronoi_areas(vor)
    
    return areas[E > thresh].sum()
    

def act_area_chull(X, E, thresh):
    """
    Calculate area of activated cells using convex hull
    """
    
    # Get cells at boundary
    if X.shape[0] < 3:
        X_where_bounds = np.array([i for i in range(X.shape[0])])
    else:
        X_where_bounds = ConvexHull(X).vertices

    # Get activated cells at first and last time
    Et = E > thresh
    
    # Exclude time-points with <3 points (throws error)
    if np.sum(Et) < 3:
        a = 0
    else:
        a = ConvexHull(X[Et]).volume
        
    return a
    
    
    
    
####### 1D chained linear induction simulation

def update_1D_ci(S, A, dt):
    """update function for 1D chained linear induction"""
    return np.maximum(S + np.dot(A, S) * dt, 0)

def ci_sim(n_tc, alpha, n, S_init, steps, dt):
    """Returns a data frame of simulated data for chained simple linear induction in 1D."""
    # Initialize expression
    S = np.zeros(n_tc + 2) + S_init
    df = pd.DataFrame(
        {
            "cell": ["I"] + ["S_" + str(i).zfill(n_tc // 10 + 1) for i in range(n_tc + 1)],
            "Signal expression": S,
            "step": 0,
        }
    )

    df.head()

    # Get system of equations as matrix
    block1 = np.array([[0,         0], 
                       [1,        -1]])
    block2 = np.zeros((2, n_tc))
    block3 = np.concatenate((np.array([[0, alpha / n],]), np.zeros((n_tc - 1, 2))))
    block4 = np.diag((alpha / n,) * (n_tc - 1), -1) + np.diag((-1,) * n_tc, 0) + np.diag((alpha / n,) * (n_tc - 1), 1)

    A = np.block([[block1, block2],
                  [block3, block4]])
    
    # Run simulation
    df_ls = [df]
    for step in np.arange(steps):
        S = update_1D_ci(S, A, dt)
        df_ls.append(
            pd.DataFrame(
                {
                    "cell": ["I"] + ["S_" + str(i).zfill(n_tc // 10 + 1) for i in range(n_tc + 1)],
                    "Signal expression": S,
                    "step": step + 1,
                }
            )
        )
    
    # Construct output dataframe
    df = pd.concat(df_ls)
    df["step"] = [int(x) for x in df["step"]]
    df["time"] = df["step"] * dt
    
    return df

####### Hill functions

# Activating Hill functions
def hill_a(x, k, p):
    """Activating Hill function. If x is 0, returns 0 and ignores divide by zero warning.
    NOTE: x must be a Numpy array, else x = 0 will throw a Zero Division error."""
    old_settings = np.seterr(divide='ignore')
    h = 1 / ((k / x) ** p + 1)
    np.seterr(**old_settings);
        
    return h

####### 1D chained non-linear induction simulation

def update_1D_ci_nl(S, A, dt, params):
    """update function for 1D chained non-linear induction"""
    alpha, n, k_s, p_s = params
    dS_dt = alpha * hill_a(np.dot(A, S)/n, k_s, p_s) - S
    dS_dt[0] = 0
    dS_dt[1] = S[0] - S[1]
    return np.maximum(S + dS_dt * dt, 0)

def ci_sim_nl(n_tc, params, steps, dt, S_init=None, I_0=None, update_fun=update_1D_ci_nl):
    """Returns a data frame of simulated data for chained non-linear induction in 1D."""
    
    # Get initial expression vector if S_init not specified
    if S_init is None:
        assert(I_0 is not None), """If no S_init is specified, I_0 must be specified."""
        S_init = np.array((I_0,) + (0,) * n_tc)
    
    
    # Initialize expression
    cell_names = ["S_" + str(i).zfill(n_tc // 10 + 1) for i in range(n_tc + 1)]
    df = pd.DataFrame(
        {
            "cell": cell_names,
            "Signal expression": np.array(S_init),
        }
    )
    df['step'] = 0
    
    # Get adjacency matrix
    A = np.diag((1,) * (n_tc), -1) + np.diag((1,) * (n_tc), 1)
    
    # Run simulation
    df_ls = [df]
    S = S_init
    
    # Integrate over time
    for step in np.arange(steps):
        S = update_fun(S, A, dt, params=params)
        df_ls.append(
            pd.DataFrame(
                {
                    "cell": cell_names,
                    "Signal expression": S,
                    "step": step + 1,
                }
            )
        )
    
    # Construct output dataframe
    df = pd.concat(df_ls)
    df["step"] = [int(x) for x in df["step"]]
    df["time"] = df["step"] * dt
    
    return df

############# Lattice functions (X, A, meta_df)

def hex_grid(rows, cols=0, r=1, sigma=0, **kwargs):
    """
    Returns XY coordinates of a regular 2D hexagonal grid 
    (rows x cols) with edge length r. Points are optionally 
    passed through a Gaussian filter with std. dev. = sigma * r.
    """
    
    # Check if square grid
    if cols == 0:
        cols = rows
    
    # Populate grid 
    x_coords = np.linspace(-r * (cols - 1) / 2, r * (cols - 1) / 2, cols)
    y_coords = np.linspace(-np.sqrt(3) * r * (rows - 1) / 4, np.sqrt(3) * r * (rows - 1) / 4, rows)
    X = []
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            X.append(np.array([x + (j % 2) * r / 2, y]))
    X = np.array(X)
    
    # Apply Gaussian filter if specified
    if sigma != 0:
        return np.array([np.random.normal(loc=x, scale=sigma*r) for x in X])
    else:
        return X

def hex_grid_square(n, **kwargs):
    """Returns XY coordinates of n points on a square regular 2D hexagonal grid with edge 
    length r, passed through a Gaussian filter with std. dev. = sigma * r."""
    
    # Get side length for square grid
    rows = int(np.ceil(np.sqrt(n)))
    
    return hex_grid(rows, **kwargs)[:n]

def hex_grid_circle(radius, sigma=0.0, r=1):
    """Returns XY coordinates of all points on a regular 2D hexagonal grid of edge length r within 
    distance radius of the origin, passed through a Gaussian filter with std. dev. = sigma * r."""
    
    # Get side length for a square grid
    rad = ceil(radius)
    num_rows = rad * 2 + 1;
    
    # Populate square grid with points
    X = []
    for i, x in enumerate(np.linspace(-r * (num_rows - 1) / 2, r * (num_rows - 1) / 2, num_rows)):
        for j, y in enumerate(
            np.linspace(-np.sqrt(3) * r * (num_rows - 1) / 4, np.sqrt(3) * r * (num_rows - 1) / 4, num_rows)
        ):
            X.append(np.array([x + ((rad - j) % 2) * r / 2, y]))
    
    # Pass each point through a Gaussian filter
    if (sigma > 0):
        X = np.array([np.random.normal(loc=x, scale=sigma*r) for x in X])
    
    # Select points within radius and return
    dists = [np.linalg.norm(x) for x in X]
    
    return np.array([X[i] for i in np.argsort(dists) if np.linalg.norm(dists[i]) <= radius])


def update_S(S, A, fun, dt, params):
    """Wrapper function to update expression vector S using update function fun"""
    
    # Get Sj, the sum of all neighbor expresion
    Sj = np.dot(A, S)
    
    # Get number of neighbors
    n = np.sum(A, axis=0)
    
    # Replace zero-neighbor cases with n = 1 to avoid 0/0 division
    n = np.maximum(n, 1)
    
    alpha, theta, beta, Kpf, ppf, gamma = params
    params = Sj, alpha, theta, beta, Kpf, ppf, gamma
    
    # Run update while catching divide by zero warnings
    old_settings = np.seterr(divide='ignore')
    new_S = fun(S, n, dt, params)
    np.seterr(**old_settings);
    
    return new_S


def get_center_cells(X, n_center=1):
    """Returns indices of the n_cells cells closest to the origin given their coordinates as an array X."""
    return np.argpartition([np.linalg.norm(x) for x in X], n_center)[:n_center]


def initialize_lattice(
    n_cells,
    init_S=0,
    n_senders=1,
    sender_fun=get_center_cells,
    S_sender=100,
    col_names=["Signal expression"],
    *args,
    **kwargs
):
    """Returns a tuple of the expression matrix and a DataFrame for the first time-point of a time-series of signal propagation on a lattice."""

    # Set initial receiver expression and set sender cell expression
    S = np.zeros(n_cells) + init_S
    senders = sender_fun(X, n_center=n_senders)
    for sender in senders:
        S[sender] = S_sender

    # Make dataframe
    out_df = pd.DataFrame(np.array([S]).T)
    out_df.index.names = ["cell"]
    out_df.columns = col_names
    out_df = out_df.reset_index()
    out_df["step"] = 0

    return S, out_df


def lattice_signaling_sim(
    n_cells,
    A,
    steps,
    dt,
    params,
    update_fun,
    init_S=0,
    n_senders=1,
    sender_fun=get_center_cells,
    S_sender=100,
    col_names=["Signal expression"],
    *args,
    **kwargs
):
    """Returns a DataFrame of lateral signaling on a lattice."""
    S, df = initialize_lattice(**locals())
    ls = [df]
    
    senders = get_center_cells(X, n_center=n_senders)

    for step in np.arange(1, steps):

        # Run update
        S = update_S(S, A, update_fun, dt, params)

        # Fix sender cells in high-ligand state
        for sender in senders:
            S[sender] = S_sender

        # Append to data list
        df = pd.DataFrame(np.array([S]).T)
        df.index.names = ["cell"]
        df.columns = col_names
        df = df.reset_index()
        df["step"] = step
        ls.append(df)

    df = pd.concat(ls)
    df["step"] = [int(x) for x in df["step"]]
    df["time"] = df["step"] * dt

    return df

###### Functions for regular lattices

def initialize_lattice_sim_regular(
    S_init,
    *args,
    **kwargs
):
    """Returns a tuple of the expression matrix and a DataFrame for the first time-point of a 
    time-series of signal propagation from one sender cell at the center of a regular lattice."""
    
    # Get initial expression
    S_init = np.array(S_init)
    
    digits = floor(log10(S_init.shape[0] - 2)) + 1
    cell_names = ["I"] + ["S_" + str(i).zfill(digits) for i in range(S_init.shape[0] - 1)]
    # Make dataframe
    out_df = pd.DataFrame(
        {
            "cell_ix": np.arange(S_init.shape[0]),
            "cell": cell_names,
            "Signal expression": S_init,
            "step": 0,
        }
    )
    
    return S_init, out_df, cell_names, digits


def lattice_signaling_sim_regular(
    R,
    steps,
    dt,
    params,
    update_fun,
    rho=1.01,
    I_0=None, 
    S_init=None,
    *args,
    **kwargs
):
    """Returns a DataFrame of simulated lateral signaling on a regular lattice of cells."""
    X = hex_grid_circle(R)
    N = X.shape[0] - 1
    A = sp.distance.squareform(sp.distance.pdist(X) < rho) + 0
    
    # Get initial expression vector if init_S not specified
    if S_init is None:
        assert(I_0 is not None), """If no S_init is specified, I_0 must be specified."""
        S_init = np.array((I_0, I_0) + (0,) * N)
    
    # Initialize expression
    S, df, cell_names, digits = initialize_lattice_sim_regular(S_init)
    ls = [df]
    
    # Add a row and column to A for the amount of inducer I
    A = np.vstack((np.zeros((1, N + 2)), np.hstack((np.zeros((N + 1, 1)), A))))
    
    for step in np.arange(steps):
        # Run update
        S = update_fun(S, A, dt, params)
        
        # Append to data list
        df = pd.DataFrame(
            {
                "cell": cell_names,
                "Signal expression": S,
                "step": step + 1,
                "cell_ix": np.arange(N + 2),
            }
        )
        ls.append(df)
    
    # Construct output DataFrame
    df = pd.concat(ls)
    df["step"] = [int(x) for x in df["step"]]
    df["time"] = df["step"] * dt
    
    locs = np.concatenate((((0, 0),), X))
    df['X_coord'] = [locs[int(ix), 0] for ix in df['cell_ix'].values]
    df['Y_coord'] = [locs[int(ix), 1] for ix in df['cell_ix'].values]
    
    return df.reset_index(drop=True)


####### 2D non-linear induction, with PAR

def update_2D_nl(S, A, dt, params):
    """update function for 2D induction with a Hill function on a regular lattice"""
    alpha, n, k_s, p_s = params
    
    f = lambda s_i_bar: biocircuits.reg.act_hill(s_i_bar/k_s, p_s)
    
    dS_dt = alpha * f((1/n) * np.dot(A, S)) - S
    dS_dt[0] = 0
    dS_dt[1] = S[0] - S[1]
    return np.maximum(S + dS_dt * dt, 0)


def update_2D_nl_par(S, A, dt, params):
    """update function for 2D induction with a Hill function and PAR on a regular lattice"""
    alpha, n, k_s, p_s, k_r, p_r, logic = params
    
    if (logic[:2].lower() == 'an'):
        f = lambda s_i, s_i_bar: biocircuits.reg.aa_and(x=s_i/k_r, y=s_i_bar/k_s, nx=p_r, ny=p_s)
    elif (logic[:2].lower() == 'or'):
        f = lambda s_i, s_i_bar: biocircuits.reg.aa_or(x=s_i/k_r, y=s_i_bar/k_s, nx=p_r, ny=p_s)
    elif (logic[:2].lower() == 'su'):
        f = lambda s_i, s_i_bar: (1/2) * (biocircuits.reg.act_hill(s_i/k_r, p_r) + biocircuits.reg.act_hill(s_i_bar/k_s, p_s))
    
    dS_dt = alpha * f(S, (1/n) * np.dot(A, S)) - S
    dS_dt[0] = 0
    dS_dt[1] = S[0] - S[1]
    return np.maximum(S + dS_dt * dt, 0)


###### 1-D chained induction simulation for arbitrary number of species

def ci_sim2(
    n_tc, 
    params, 
    steps, 
    dt, 
    I_0, 
    E_init=None, 
    update_fun=update_1D_ci_nl, 
    E_colnames = None,
):
    """Returns a data frame of simulated data for chained non-linear induction in 1D."""
    
    # Get initial expression vector if E_init not specified
    if E_init is None:
        E_init = np.zeros((n_tc + 1, I_0.size))
        E_init[0, :] = I_0.flatten()
    
    # Initialize expression
    cell_names = ["cell_" + str(i).zfill(n_tc // 10 + 1) for i in range(n_tc + 1)]
    if E_colnames is None:
        E_colnames = [str(i) + " expression" + i for i, _ in enumerate(E_init.shape[1])]
    
    df = pd.DataFrame(
        {
            "cell": cell_names,
            "step": 0,
            "time": 0,
        }
    )
    df = df.join(pd.DataFrame(
        {k : v for k, v in zip(E_colnames, E_init.T)}, 
        index=df.index
    ))
    
    # Get adjacency matrix
    A = np.diag((1,) * (n_tc), -1) + np.diag((1,) * (n_tc), 1)
    
    # Run simulation
    df_ls = [df]
    E = E_init
    
    # Integrate over time
    for step in np.arange(1, steps + 1):
        E = update_fun(E, A, dt, params=params, I=I_0)
        
        df = pd.DataFrame(
            {
                "cell": cell_names,
                "step": step,
                "time": step * dt,
            }
        )
        df = df.join(pd.DataFrame(
            {k : v for k, v in zip(E_colnames, E.T)}, 
            index=df.index
        ))
        
        df_ls.append(df)
    
    # Construct output dataframe
    df = pd.concat(df_ls)
    df["step"] = [int(x) for x in df["step"]]
    
    return df


###### Make a time-lapse of single-gene ("signal") expression on a lattice

def Regular2DLattice_vid_mp4(
    df,
    R,
    dt,
    val="Signal expression",
    colors=cc.palette.fire[:250:5],
    levels=None,
    points_opts=None,
    title=None,
    title_format=None,
    fps=20,
    **kwargs
):
    """Returns an mp4 video of gene expression of cells on a regular lattice."""

    if levels is None:
        levels = [x for x in np.linspace(df[val].min(), df[val].max(), len(colors) + 1)]
    
    if points_opts is None:
        padding = 1/(2*R)
        points_opts = dict(
            padding=padding,
            aspect="equal",
            s=13300 * 1/(R**2) / (1 + 2*padding)**2,
#             s=600,
            marker="h",
            color=val,
            color_levels=levels,
            cmap=colors,
            colorbar=True,
            xaxis="bare",
            yaxis="bare",
        )
    
    if title is None:
        title = "time = {0:.2f}" 
    if title_format is None:
        def title_format(dt, step, **kwargs):
            return (dt * step,)
    
    # Plot lattice colored by expression
    def plot_lattice(step, dt):
        step_data = df.loc[df["step"] == step, :]
        points_opts['title'] = title.format(*title_format(**locals()))
        
        plt = hv.Points(
            data=step_data, kdims=["X_coord", "Y_coord"], vdims=[val]
        ).opts(**points_opts)

        return plt
    
    steps = df['step'].max()
    hmap = hv.HoloMap([(step, plot_lattice(step, dt)) for step in range(steps + 1)])
    
    return hmap



def Voronoi_hmap_nr(
    df,
    R,
    dt,
    val="expression",
    cmap="viridis",
    crange=None,
    levels=None,
    points_opts=dict(),
    title="",
    title_format=None,
    progress_bar=False,
    **kwargs
):
    """Returns an mp4 video of expression on a non-regular lattice."""

    padding = 1/(2*R)
    opts = dict(
        padding=padding,
        aspect="equal",
        color='z',
        cmap=cmap,
        colorbar=True,
        xaxis="bare",
        yaxis="bare",
    )
    
    opts.update(points_opts)
    points_opts = opts
    
    if not title:
        title = "time = {0:.2f}" 
    if title_format is None:
        def title_format(dt, step, **kwargs):
            return (dt * step,)
    
    # Plot lattice colored by expression
    def plot_lattice(step, dt):

        # Extract data
        step_data = df.loc[df["step"] == step, ['X_coord', 'Y_coord', val]]
        X = np.array([step_data['X_coord'].values, step_data['Y_coord'].values]).T
        S = np.array([step_data[val].values]).reshape(-1,1)

        # Construct polygons of valid cells
        polygons = valid_regions(vor=sp.Voronoi(X), R=R)
        polygons = [{('x', 'y'): region[1], "z": step_data[val].values[region[0]]} for region in polygons]

        # Add formatted title
        points_opts['title'] = title.format(*title_format(**locals()))

        # Make plot
        plt = hv.Polygons(
            polygons, vdims=["z"],
        ).opts(**points_opts)
        
        # Set color range
        if crange is not None:
            plt = plt.redim.range(z=crange)

        return plt
    
    steps = df['step'].max()
    
    # Make iterator w/ or w/out progress bar
    if progress_bar:
        iterator = tqdm.tqdm(range(steps + 1))
    else:
        iterator = range(steps + 1)
    
    hmap = hv.HoloMap([(step, plot_lattice(step, dt)) for step in iterator])
    
    return hmap

###### Cell division functions

@numba.njit
def shoelace_area(points):
    """Returns the area enclosed by a convex polygon. Uses the shoelace method to 
    calculate area given an ordered Numpy array of 2D Cartesian coordinates.
    """
    area = np.dot(
        points[:, 0], np.roll(points[:, 1], shift=1)
    ) - np.dot(
        np.roll(points[:, 0], shift=1), points[:, 1]
    )
    return np.abs(area) / 2


def voronoi_areas(vor):
    """Given the Voronoi tesselation of a set of points, returns the area of each 
    point's Voronoi region. `vor` is an object generated by scipy.spatial.Voronoi."""
    
    # For each point, get the vertices for its region
    region_verts = [vor.regions[i] for i in vor.point_region]
    
    areas = np.zeros(vor.points.shape[0])
    for i, vert_idx in enumerate(region_verts):
        
        # Idenfity infinite regions
        if any([idx < 0 for idx in vert_idx]):
            areas[i] = np.inf
            
        # Calculate area of finite regions
        else:
            vert_coords = np.array([vor.vertices[i] for i in vert_idx])
            areas[i] = shoelace_area(vert_coords)
    
    return areas


def p_largest_cells(vor, R, p=0.1):
    """Returns the indices of the cells with the largest volumes defined 
    as the areas of their Voronoi regions. Excludes cells with infinite 
    volume and cells with Voronoi ridges outside the radius R.
    vor is a voronoi tesselation object returned by scipy.spatial.Voronoi
    The largest volumes are the top (100 * p)% of cell volumes."""
    
    # Catch trivial case
    if (p == 0):
        return np.array(())
    
    # For each cell, get the vertices and area of its Voronoi region
    verts = [np.array(vor.regions[i]) for i in vor.point_region]
    areas = voronoi_areas(vor)
    
    # For each cell, calculate whether all of its vertices lie inside the 
    #  boundary of radius R
    verts_in_bounds = np.zeros_like(areas)
    for i, vert_idx in enumerate(verts):
        verts_in_bounds[i] = np.all(np.linalg.norm(vor.vertices[vert_idx], axis=1) < R)
    
    # Exclude infinite areas by setting them equal to 0
    areas = np.where(areas == np.inf, 0, areas)
    
    # Exclude other boundary cells by setting equal to 0
    areas = areas * verts_in_bounds
    
    # Calculate how many cells to return
    n_largest = floor(p * areas.size)
    
    # Return indices of largest cells
    return np.argsort(-areas)[:n_largest].astype(np.int32)


def n_largest_cells(vor, R, n=2):
    """Returns the indices of the cells with the largest volumes defined 
    as the areas of their Voronoi regions. Excludes cells with infinite 
    volume and cells with Voronoi ridges outside the radius R.
    vor is a voronoi tesselation object returned by scipy.spatial.Voronoi
    The largest volumes are the top (100 * p)% of cell volumes."""
    
    # For each cell, get the vertices and area of its Voronoi region
    verts = [np.array(vor.regions[i]) for i in vor.point_region]
    areas = voronoi_areas(vor)
    
    # For each cell, calculate whether all of its vertices lie inside the 
    #  boundary of radius R
    verts_in_bounds = np.zeros_like(areas)
    for i, vert_idx in enumerate(verts):
        verts_in_bounds[i] = np.all(np.linalg.norm(vor.vertices[vert_idx], axis=1) < R)
    
    # Exclude infinite areas by setting them equal to 0
    areas = np.where(areas == np.inf, 0, areas)
    
    # Exclude other boundary cells by setting equal to 0
    areas = areas * verts_in_bounds
    
    # Return indices of largest cells
    return np.argsort(-areas)[:n].astype(np.int32)

@numba.njit
def angle(v1, v2):
    """Calculates angle from point p1 to p2 in radians"""
    dx, dy = (v2 - v1).T
    return np.arctan2(dy, dx)

@numba.njit
def where_bounds(target, arr):
    """Returns (lower_bound, upper_bound) such that lower_bound is the index of the 
    greatest element of arr less than target and upper_bound is the index of the 
    smallest element of arr greater than target."""
    
    lower_bound = np.argmin(arr)
    upper_bound = np.argmax(arr)
    
    assert ((target >= arr[lower_bound]) & (target <= arr[upper_bound])), "target value is outside range of array"
    
    for i, el in enumerate(arr):
        if ((el < target) & (el > arr[lower_bound])):
            lower_bound = i
        if ((el > target) & (el < arr[upper_bound])):
            upper_bound = i
    
    return lower_bound, upper_bound

@numba.njit
def lines_intersect(xy1, xy2, xy3, xy4):
    """Get the point of intersection between two lines, one passing 
    through xy1 and xy2 and the other passing through xy3 and xy4.
    Uses Cramer's rule.
    """
    
    # Unpack coordinates
    x1, y1 = xy1
    x2, y2 = xy2
    x3, y3 = xy3
    x4, y4 = xy4
    
    # Compute coefficients of form ax + by = c
    a1, b1, c1 = y1 - y2, x2 - x1, x2*y1 - x1*y2
    a2, b2, c2 = y3 - y4, x4 - x3, x4*y3 - x3*y4
    
    # Identify if solution is trivial (intersect at origin)
    if ((c1 == 0) & (c2 == 0)):
        return np.array((0, 0))
    
    # Otherwise, compute determinants of matrices
    D = np.linalg.det(np.array([[a1, b1], [a2, b2]]))
    Dx = np.linalg.det(np.array([[c1, b1], [c2, b2]]))
    Dy = np.linalg.det(np.array([[a1, c1], [a2, c2]]))
    
    return np.array([Dx/D, Dy/D])


def divide_cell(centroid, vertices):
    """Returns the daughter cells of a cell division event as XY coordinates of the new centroids.
    Cells are dividided along their long axis, defined as the longest line segment inside the polygon
    that passes through the centroid and a vertex.
    """
    
    axis_lengths = np.empty(vertices.shape[0])
    poles = np.empty_like(vertices)

    for i, vert in enumerate(vertices):

        # Get all other vertices
        other_verts = vertices[np.arange(vertices.shape[0]) != i, :]

        # Get angle between vert and centroid
        cent_angle = angle(vert, centroid)

        # Get angles between vert and all other verts
        vert_angles = np.array([angle(vert, vv) for vv in other_verts])

        # If centroid angle is outside the range of vert_angles, the opposite vertices are
        #  the ones at the upper and lower bounds. Otherwise, find the vertices v1 and v2
        #  between which the centroid angle lies.
        if (cent_angle < vert_angles.min()) | (cent_angle > vert_angles.max()):
            v1i, v2i = np.argmin(vert_angles), np.argmax(vert_angles)
        else:
            v1i, v2i = where_bounds(cent_angle, vert_angles)
        v1, v2 = other_verts[v1i], other_verts[v2i]

        # Get the point of intersection between the line passing through the two opposite
        #  vertices and the line passing through vert and centroid using Cramer's rule
        poles[i] = lines_intersect(centroid, vert, v1, v2)

        # Get length of the axis, from vertex to pole
        axis_lengths[i] = np.linalg.norm(poles[i] - vert)

    # Get the long axis and the new cell centroids
    long_axis_ix = np.argmax(axis_lengths)  # index of vertex on long axis
    return np.mean((centroid, vertices[long_axis_ix]), axis=0), np.mean((centroid, poles[long_axis_ix]), axis=0)

###### Functions for running simulations on Non-regular lattices

def initialize_nrlattice(
    R,
    sigma,
    S_init=0,
    n_species=1,
    n_senders=1,
    S_sender=None,
    sender_fun=get_center_cells,
    unique_ID_sep="-",
    *args,
    **kwargs
):
    """Returns a tuple of the expression matrix and a DataFrame for the first time-point 
    of a time-series of signal propagation on a lattice.
    """
    
    # Initialize cell locations, number of cells, Voronoi tesselation, and connectivity
    X = hex_grid_circle(radius=R, sigma=sigma)
    vor = sp.Voronoi(X)
    
    # Get the indices of sender cells and initialize their expression
    #  (Default expression is equal to inducer)
    senders = sender_fun(X, n_senders)
    S = np.zeros((X.shape[0], n_species)) + S_init
    if S_sender is None:
        S_sender = I
    for i in senders:
        S[i] = S_sender
    
    # Give each cell a unique ID string.
    #  The first 3 digits are the index of the cell in the initial location matrix X.
    #  The next 2 digits are the generation number of the cell (how many divisions its
    #    lineage has undergone). This starts at 0 for all cells at time 0.
    #  The last 15 digits are a sequence of binary digits encoding cell lineage. When a
    #    cell divides and its daughters enter generation #G, each daughter is assigned
    #    either a 0 or a 1 at digit #G (eg. generation 2 is encoded by the 2nd digit from the right).
    
    # Cell IDs, in the same order as their positions in X and S
    unique_IDs = np.array([str(i).zfill(3) + ":00" + ":" + unique_ID_sep*15 for i in range(X.shape[0])])
    
    df = pd.DataFrame(
        {
            "cell": unique_IDs,
            "sender": np.isin(np.arange(unique_IDs.size), senders),
            "expression": S.flatten(),
            "step": 0,
            "time": 0,
            "X_coord": X[:, 0],
            "Y_coord": X[:, 1],
        }
    )
    
    return X, vor, S, unique_IDs, senders, df


def append_daughter_IDs(IDs, mom, sep=":"):
    """Returns a Numpy array of unique ID strings after the cell at index mom divides into two daughter cells."""
    
    new_id1 = IDs[mom].split(sep)
    
    # Increment generation to get first daughter's ID
    gen = int(new_id1[1]) + 1
    new_id1[1] = str(gen).zfill(2)
    
    # Increment lineage to get second daughter's ID
    new_id2 = new_id1[:2] + [new_id1[2][:15 - gen] + '1' + new_id1[2][:gen - 1]]
    
    # Append daughters to IDs
    return np.concatenate((IDs, (sep.join(new_id1), sep.join(new_id2))))


def lattice_adjacency(vor, R):
    """Returns the graph transition matrix used by lattice simulation functions to perform 
    time integration of ODEs.
    """

    # Initialize the weighted, undirected adjacency matrix `adj`
    adj = np.zeros((vor.npoints,) * 2)
    
    # Identify cells (regions) as inert if their common ridge has an infinite vertex or a vertex 
    #  outside the radius R. Otherwise, add the ridge length to the adjacency matrix.
    inert_cell = np.zeros((vor.npoints, 1))
    for cells, ridge_verts in vor.ridge_dict.items():
        
        # If a vertex is infinite, both cells are inert
        if (ridge_verts[0] < 0) | (ridge_verts[1] < 0):
            inert_cell[cells, :] = True
            
        # If a vertex lies outside the circle, both cells are inert
        elif (np.any(np.linalg.norm(vor.vertices[ridge_verts, :], axis=1) > R)):
            inert_cell[cells, :] = True
        
        # Otherwise, populate adjacency matrix symmetrically with ridge length
        else: 
            adj[cells[::-1]] = adj[cells] = np.linalg.norm(np.subtract(*vor.vertices[ridge_verts]))
    
    # For inert cells, set their row in the adjacency matrix to 0 (cannot receive signal)
    adj = adj * (1 - inert_cell)
    
    # Row-normalize adjacency matrix (divide each row by its degree).
    # If a row represents an inert cell, set degree to 1 to avoid 0/0 warnings/errors.
    degrees = np.sum(adj, axis=1, keepdims=True) + inert_cell
    adj = adj / degrees

    return adj


def update_S_nr_thresh(S, A, dt, I, senders, params, *args, **kwargs):
    """Returns the time-integrated expression of S based on the normalized 
    adjacency matrix A of a non-regular lattice of cells over the time-step dt. 
    Signal expression is induced above a threshold enforced by a Hill function.
    """
    alpha, k_s, p_s = params
    
    f = lambda A, S: biocircuits.reg.act_hill(np.dot(A, S) / k_s, n=p_s)
    
    dS_dt = alpha * f(A, S) - S
    for sender in senders:
        dS_dt[sender, :] = I - S[sender, :]
    return np.maximum(S + dS_dt * dt, 0)


def division_events(dt, steps, doubling_time, divs_per_doubling):
    """Returns the steps at which division events may occur. A division event
    may consist of one or multiple cell divisions in the same time-step.
    """
    # Get the times of each division event. During each division event, the largest (1/divs_per_doubling) %
    #  of cells will undergo division. First division event happens between steps 0 and 1 (time = dt/2).
    assert (doubling_time >= dt), """Time-steps must be shorter than the doubling time."""
    
    div_event_times = np.arange(dt, dt * steps, doubling_time / divs_per_doubling)
    
    # Get the steps at which division events will occur
    time = np.arange(steps + 1) * dt
    
    return np.array([where_bounds(target=div_event, arr=time)[1] for div_event in div_event_times])


def nrlattice_signaling_sim(
    R,
    sigma,
    steps,
    dt,
    I_0,
    params,
    doubling_time=np.log(2),
    divs_per_doubling=5,
    update_fun=update_S_nr_thresh,
    I_off_step = -1,
    div_off_step = -1,
    S_init=0,
    n_senders=1,
    sender_fun=get_center_cells,
    S_sender=None,
    init_kwargs=dict(),
    *args,
    **kwargs
):
    """
    """
    
    # Initialize inducer and sender. If no initial sender expression is provided, use inducer level
    I = I_0
    if S_sender is None:
        S_sender = I_0
    
    # Initialize lattice
    X, vor, S, unique_IDs, senders, df = initialize_nrlattice(
        R=R,
        sigma=sigma,
        S_init=S_init,
        n_species=np.array(S_sender).size,
        n_senders=n_senders,
        S_sender=S_sender,
        **init_kwargs
    )
    
    # Get modified adjacency matrix to integrate expression over time
    A = lattice_adjacency(vor=vor, R=R)
    
    # Get the steps at which division events (one or multiple divisions) will occur and the
    #  proportion of cells that will divide during each division events.
    div_event_steps = division_events(dt, steps, doubling_time, divs_per_doubling)
    div_prop = 1/divs_per_doubling
    
    # Initialize division to be ON
    div_is_on = np.array([True])
    
    ls = [df]
    for step in np.arange(1, steps + 1):
        
        # Execute any division events
        div_events = np.count_nonzero(div_event_steps == step)
        
        # Check if division should be shut OFF. If so, set mothers to be empty (no division).
        if (step == div_off_step):
            div_is_on = np.array([False])
            mothers = np.array([])
            
        # If not, select the largest cells for division and get daughter cell locations.
        #  Defaults to empty Numpy array if div_events=0.         
        if div_is_on:
            mothers = p_largest_cells(R=R, vor=vor, p=div_prop * div_events)
        
        for mom in mothers:
            # Get the locations of daughter cells after division
            vertices = np.array([vor.vertices[i] for i in vor.regions[vor.point_region[mom]]])
            daughters = divide_cell(centroid=vor.points[mom], vertices=vertices)

            # Append arrays with daughter cells' locations, expression (same as mom's), and new IDs
            X = np.concatenate((X, daughters))
            S = np.concatenate((S, S[(mom, mom), :]))
            unique_IDs = append_daughter_IDs(unique_IDs, mom)

            # If mom was a sender, append daughters to senders
            if (mom in senders):
                senders = np.concatenate((senders, (X.shape[0]-2, X.shape[0]-1)))

        # Update arrays if there were any divisions
        if (mothers.size > 0):
            
            # Remove mothers from arrays
            X = np.delete(X, mothers, axis=0)
            S = np.delete(S, mothers, axis=0)
            unique_IDs = np.delete(unique_IDs, mothers, axis=0)

            # Remove any mothers from senders and revise sender indices to match the new indices
            senders = senders[np.isin(senders, mothers, invert=True)]
            senders = senders - np.array([np.sum(sender > mothers) for sender in senders])
            
            # Re-calculate Voronoi and update lattice adjacency
            vor = sp.Voronoi(X)
            A = lattice_adjacency(vor=vor, R=R)
        
        # Check if inducer has been turned off. I_off_step defaults to -1 (inducer never turns off)
        if (step == I_off_step):
            I = 0
        
        # Run update
        S = update_fun(S, A, dt, I, senders, params)
        
        # Append to data list
        df = pd.DataFrame(
            {
                "cell": unique_IDs,
                "sender": np.isin(np.arange(unique_IDs.size), senders),
                "expression": S.flatten(),
                "step": step,
                "time": step * dt,
                "X_coord": X[:, 0],
                "Y_coord": X[:, 1],
            }
        )
        ls.append(df)
    
    # Construct output DataFrame
    df = pd.concat(ls)
    return df.reset_index(drop=True)


def valid_regions(vor, R):
    """Returns a list of 2-tuples, each containing the index of a valid region of the 
    Voronoi tesselation vor and a Numpy array of its coordinates. Valid regions are 
    regions with vertices that are all finite and lie inside a circle centered at the 
    origin with radius R.
    """

    # Get the vertices for each region, excluding infinite regions
    regs = [np.array(i) for i in vor.regions]
    regs = [
        (idx, vor.vertices[regs[reg_idx]])
        for idx, reg_idx in enumerate(vor.point_region)
        if np.all(regs[reg_idx] >= 0)
    ]

    # Exclude regions with a vertex outside the circle
    regs = [
        reg for i, reg in enumerate(regs) if np.all(np.linalg.norm(reg[1], axis=1) < R)
    ]

    return regs


def div_probability(V, dt, doubling_time, V_sat):
    """Returns the probability of cell division in the next dt time-units, 
    dependent on the cell volume and a threshold volume V_sat. b tunes the
    sharpness of transition between proliferation and arrest.
    """
    return biocircuits.reg.act_hill(x=V/V_sat, n=8) * (dt / doubling_time)


def nrlattice_signaling_sim_prob(
    R,
    sigma,
    steps,
    dt,
    I_0,
    desired_cells,
    params,
    doubling_time=np.log(2),
    update_fun=update_S_nr_thresh,
    I_off_step=-1,
    S_init=0,
    n_senders=1,
    sender_fun=get_center_cells,
    S_sender=None,
    init_kwargs=dict(),
    update_kwargs=dict(),
    *args,
    **kwargs
):
    """Probabilistic division
    """

    # Initialize inducer and sender. If no initial sender expression is provided, use inducer level
    I = I_0
    if S_sender is None:
        S_sender = I_0

    # Initialize lattice
    X, vor, S, unique_IDs, senders, df = initialize_nrlattice(
        R=R,
        sigma=sigma,
        S_init=S_init,
        n_species=np.array(S_sender).size,
        n_senders=n_senders,
        S_sender=S_sender,
        **init_kwargs
    )

    # Get modified adjacency matrix to integrate expression over time
    A = lattice_adjacency(vor=vor, R=R)

    ls = [df]
    for step in np.arange(1, steps + 1):

        # Get cell volumes for valid cells as a list of 2-tuples
        volumes = np.array([[idx, vol] for idx, vol in enumerate(voronoi_areas(vor))])
        volumes = np.array([volumes[i, :] for i, _ in valid_regions(vor, R)])

        # Convert cell volume to probability of division and generate
        #  random numbers to decide outcome (division or not)
        volumes[:, 1] = div_probability(
            volumes[:, 1] / (np.pi * R ** 2), dt=dt, doubling_time=doubling_time, V_sat = 2.2 / desired_cells

        )
        
        mothers = np.array([i for i in volumes[:, 0]]).astype(np.int32)
        
        mothers = mothers[
            volumes[:, 1] > np.random.uniform(low=0, high=1, size=volumes.shape[0])
        ]
        
        # Execute divisions
        for mom in mothers:

            # Get the locations of daughter cells after division
            vertices = np.array(
                [vor.vertices[i] for i in vor.regions[vor.point_region[mom]]]
            )
            daughters = divide_cell(centroid=vor.points[mom], vertices=vertices)

            # Append arrays with daughter cells' locations, expression (same as mom's), and new IDs
            X = np.concatenate((X, daughters))
            S = np.concatenate((S, S[(mom, mom), :]))
            unique_IDs = append_daughter_IDs(unique_IDs, mom)

            # If mom was a sender, append daughters to senders
            if mom in senders:
                senders = np.concatenate((senders, (X.shape[0] - 2, X.shape[0] - 1)))

        # Update arrays if there were any divisions
        if mothers.size > 0:

            # Remove mothers from arrays
            X = np.delete(X, mothers, axis=0)
            S = np.delete(S, mothers, axis=0)
            unique_IDs = np.delete(unique_IDs, mothers, axis=0)

            # Remove any mothers from senders and revise sender indices to match the new indices
            senders = senders[np.isin(senders, mothers, invert=True)]
            senders = senders - np.array(
                [np.sum(sender > mothers) for sender in senders]
            )

            # Re-calculate Voronoi and update lattice adjacency
            vor = sp.Voronoi(X)
            A = lattice_adjacency(vor=vor, R=R)

        # Check if inducer has been turned off. I_off_step defaults to -1 (inducer never turns off)
        if step == I_off_step:
            I = 0

        # Run update
        S = update_fun(S, A, dt, I, senders, params, X=X, **update_kwargs)

        # Append to data list
        df = pd.DataFrame(
            {
                "cell": unique_IDs,
                "sender": np.isin(np.arange(unique_IDs.size), senders),
                "expression": S.flatten(),
                "step": step,
                "time": step * dt,
                "X_coord": X[:, 0],
                "Y_coord": X[:, 1],
            }
        )
        ls.append(df)

    # Construct output DataFrame
    df = pd.concat(ls)
    return df.reset_index(drop=True)

def nrlattice_signaling_sim_prob_dildiv(
    R,
    sigma,
    steps,
    dt,
    I_0,
    desired_cells,
    params,
    doubling_time=np.log(2),
    update_fun=update_S_nr_thresh,
    I_off_step=-1,
    S_init=0,
    n_senders=1,
    sender_fun=get_center_cells,
    S_sender=None,
    init_kwargs=dict(),
    update_kwargs=dict(),
    *args,
    **kwargs
):
    """
    Probabilistic division. Division events lead to dilution of signal, each daughter inheriting
    half the mother's expression.
    """

    # Initialize inducer and sender. If no initial sender expression is provided, use inducer level
    I = I_0
    if S_sender is None:
        S_sender = I_0

    # Initialize lattice
    X, vor, S, unique_IDs, senders, df = initialize_nrlattice(
        R=R,
        sigma=sigma,
        S_init=S_init,
        n_species=np.array(S_sender).size,
        n_senders=n_senders,
        S_sender=S_sender,
        **init_kwargs
    )

    # Get modified adjacency matrix to integrate expression over time
    A = lattice_adjacency(vor=vor, R=R)

    ls = [df]
    for step in np.arange(1, steps + 1):

        # Get cell volumes for valid cells as a list of 2-tuples
        volumes = np.array([[idx, vol] for idx, vol in enumerate(voronoi_areas(vor))])
        volumes = np.array([volumes[i, :] for i, _ in valid_regions(vor, R)])

        # Convert cell volume to probability of division and generate
        #  random numbers to decide outcome (division or not)
        volumes[:, 1] = div_probability(
            volumes[:, 1] / (np.pi * R ** 2), 
            dt=dt, 
            doubling_time=doubling_time, 
            V_sat = 2.2 / desired_cells
        )
        
        mothers = np.array([i for i in volumes[:, 0]]).astype(np.int32)
        
        mothers = mothers[
            volumes[:, 1] > np.random.uniform(low=0, high=1, size=volumes.shape[0])
        ]
        
        # Execute divisions
        for mom in mothers:

            # Get the locations of daughter cells after division
            vertices = np.array(
                [vor.vertices[i] for i in vor.regions[vor.point_region[mom]]]
            )
            daughters = divide_cell(centroid=vor.points[mom], vertices=vertices)

            # Append arrays with daughter cells' locations, expression (same as mom's), and new IDs
            X = np.concatenate((X, daughters))
            S = np.concatenate((S, S[(mom, mom), :]/2))
            unique_IDs = append_daughter_IDs(unique_IDs, mom)

            # If mom was a sender, append daughters to senders
            if mom in senders:
                senders = np.concatenate((senders, (X.shape[0] - 2, X.shape[0] - 1)))

        # Update arrays if there were any divisions
        if mothers.size > 0:

            # Remove mothers from arrays
            X = np.delete(X, mothers, axis=0)
            S = np.delete(S, mothers, axis=0)
            unique_IDs = np.delete(unique_IDs, mothers, axis=0)

            # Remove any mothers from senders and revise sender indices to match the new indices
            senders = senders[np.isin(senders, mothers, invert=True)]
            senders = senders - np.array(
                [np.sum(sender > mothers) for sender in senders]
            )

            # Re-calculate Voronoi and update lattice adjacency
            vor = sp.Voronoi(X)
            A = lattice_adjacency(vor=vor, R=R)

        # Check if inducer has been turned off. I_off_step defaults to -1 (inducer never turns off)
        if step == I_off_step:
            I = 0

        # Run update
        S = update_fun(S, A, dt, I, senders, params, X=X, **update_kwargs)

        # Append to data list
        df = pd.DataFrame(
            {
                "cell": unique_IDs,
                "sender": np.isin(np.arange(unique_IDs.size), senders),
                "expression": S.flatten(),
                "step": step,
                "time": step * dt,
                "X_coord": X[:, 0],
                "Y_coord": X[:, 1],
            }
        )
        ls.append(df)

    # Construct output DataFrame
    df = pd.concat(ls)
    return df.reset_index(drop=True)


##### Update functions for density-dependent expression

# Non-regular sim, non-linear (thresholded) signaling, collapse above a critical local density
def update_S_nr_thresh_dens(S, A, dt, I, senders, params, X, crit_dist, *args, **kwargs):
    """Returns the time-integrated expression of S based on the normalized 
    adjacency matrix A of a non-regular lattice of cells over the time-step dt. 
    Signal expression is induced above a threshold enforced by a Hill function.
    When a cell is too close to any other cell (pairwise distance less than the 
    cutoff crit_dist), the cell's environment is "locally dense" and the signaling 
    threshold becomes effectively infinite (signaling stops).
    """

    alpha, k_s, p_s = params
    
    # Get pairwise distance matrix and identify cell(s) with a high local density
    D = sp.distance.squareform(sp.distance.pdist(X))
    locally_dense = np.any((D > 0) & (D < crit_dist), axis=1)
    
    # For cells with high local density, set their signal input Sj to 0
    Sj = (np.dot(A, S) / k_s) * ~locally_dense.reshape(-1,1)

    # Evaluate Hill term
    f = biocircuits.reg.act_hill(Sj, n=p_s)

    # Calculate change in expression
    dS_dt = alpha * f - S
    
    for sender in senders:
        dS_dt[sender, :] = I - S[sender, :]
        
    return np.maximum(S + dS_dt * dt, 0)


def update_S_nr_thresh_dens_dil(S, A, dt, I, senders, params, X, crit_dist, *args, **kwargs):
    """Returns the time-integrated expression of S based on the normalized 
    adjacency matrix A of a non-regular lattice of cells over the time-step dt. 
    Signal expression is induced above a threshold enforced by a Hill function.
    When a cell is too close to any other cell (pairwise distance less than the 
    cutoff crit_dist), the cell's environment is "locally dense" and the signaling 
    threshold becomes effectively infinite (signaling stops).
    The parameter `mu` denotes the fraction of signal decay due to degradation. 
    In this model, decay due to dilution is added to the decay unless the cells is
    "locally dense." Therefore, this should be used with a simulator that does not
    treat cell division as causing signal dilution (i.e. daughters of division 
    inherit their mother's expression completely.)
    """

    alpha, k_s, p_s, mu = params
    
    # Get pairwise distance matrix and identify cell(s) with a high local density
    D = sp.distance.squareform(sp.distance.pdist(X))
    locally_dense = np.any((D > 0) & (D < crit_dist), axis=1).reshape(-1,1)
    
    # For cells with high local density, set their signal input Sj to 0
    Sj = (np.dot(A, S) / k_s) * ~locally_dense

    # Evaluate Hill term
    f = biocircuits.reg.act_hill(Sj, n=p_s)

    # Calculate change in expression
    dS_dt = alpha * f - (mu + ~locally_dense) * S
    
    for sender in senders:
        dS_dt[sender, :] = I - ((mu + ~locally_dense) * S)[sender, :]
        
    return np.maximum(S + dS_dt * dt, 0)


def update_S_nr_thresh_dens_nodil(S, A, dt, I, senders, params, X, crit_dist, *args, **kwargs):
    """Returns the time-integrated expression of S based on the normalized 
    adjacency matrix A of a non-regular lattice of cells over the time-step dt. 
    Signal expression is induced above a threshold enforced by a Hill function.
    When a cell is too close to any other cell (pairwise distance less than the 
    cutoff crit_dist), the cell's environment is "locally dense" and the signaling 
    threshold becomes effectively infinite (signaling stops).
    The parameter `mu` denotes the fraction of signal decay due to degradation. 
    In this model, decay due to dilution is not calculated by the update function.
    Therefore, this should be used with a simulator that treats cell division as a
    dilutional event (i.e. daughter cells inherit half their mother's expression.)
    """

    alpha, k_s, p_s, mu = params
    
    # Get pairwise distance matrix and identify cell(s) with a high local density
    D = sp.distance.squareform(sp.distance.pdist(X))
    locally_dense = np.any((D > 0) & (D < crit_dist), axis=1).reshape(-1,1)
    
    # For cells with high local density, set their signal input Sj to 0
    Sj = (np.dot(A, S) / k_s) * ~locally_dense

    # Evaluate Hill term
    f = biocircuits.reg.act_hill(Sj, n=p_s)

    # Calculate change in expression
    dS_dt = alpha * f - mu * S
    
    for sender in senders:
        dS_dt[sender, :] = I - (mu * S)[sender, :]
        
    return np.maximum(S + dS_dt * dt, 0)



