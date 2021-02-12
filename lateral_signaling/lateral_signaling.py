####### Load depenendencies
import os

import numpy as np
import pandas as pd
from math import ceil
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import Rbf
import scipy.stats


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

def get_DDE_rhs(func, *func_args):
    """
    Returns a function `rhs` with call signature 
    
      rhs(S, S_delay, *dde_args) 
      
    that can be passed to `lsig.integrate_DDE` and 
    `lsig.integrate_DDE_varargs`. This is equivalent 
    to calling
    
      func(S, S_delay, *func_args, *dde_args)
    
    Examples of args in `func_args` include:
    
    Adj         :  Adjacency matrix encoding cell neighbors
    sender_idx  :  Index (indices) of sender cells, which
                     undergo different signaling 
    
    """

    def rhs(S, S_delay, *dde_args):
        return func(S, S_delay, *func_args, *dde_args)

    return rhs

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
    where_vars,
    progress_bar=False,
    min_delay=5,
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
    
    # Make variable args a 2D array of appropriate shape
    vvals = np.atleast_2d(var_vals).T
    
    # Make variable indices iterable
    vidx = np.atleast_1d(where_vars)
    
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
        v = vvals[past_step]
        for i, vi in enumerate(vidx):
            dde_args[vi] = v[i]
        
        # Integrate
        dE_dt = rhs(E, E_delay, *dde_args)
        E = np.maximum(0, E + dE_dt * dt) 
        E_save[step] = E
    
    return E_save


@numba.njit
def get_L_vals(xs, vs, vertices):
    """
    Returns the cell-cell indices and the lengths of cell-cell 
    contacts given information about the Voronoi tesselation.
    
    xs is an (n x 2) array containing the indices of `n` pairs of 
        adjacent cells. For a Scipy Voronoi object `vor`, this is 
        equivalent to `vor.ridge_points`. Order should be preserved 
        with vs.
       
    vs is an (n x 2) array containing the indices of vertices for each
        cell-cell interface (`n` in total). Order should be preserved 
        with xs.
        
    vertices is an (m x 2) array containing the `m` vertices in the 
        Voronoi tesselation in 2D Cartesian coordinates.
    """
    
    # Count number of non-infinite Voronoi ridges
    n_l = 0
    for v1, v2 in vs:
        if (v1 >= 0) & (v2 >= 0):
            n_l += 1
    
    Lij = np.zeros((n_l, 2), dtype=np.int_)
    L_vals = np.empty(n_l, dtype=np.float32)
    
    k = 0
    for i, x12 in enumerate(xs):
        v1, v2 = vs[i]
        
        # Infinite Voronoi edges have zero length
        if (v1 < 0) | (v2 < 0):
            continue
        
        # Get length of cell-cell ridge
        ell = np.linalg.norm(vertices[v1] - vertices[v2])
        Lij[k] = x12
        L_vals[k] = ell
        
        k += 1
        
    return Lij, L_vals


@numba.njit
def get_L_vals_gaps(xs, vs, pts, vertices, cr):
    """
    Returns the cell-cell indices and the lengths of cell-cell 
    contacts given information about the Voronoi tesselation and
    a uniform cell radius `cr`.
    """
    
    # Make matrix cell-cell contact lengths
    n_l = 0
    for v1, v2 in vs:
        if (v1 >= 0) & (v2 >= 0):
            n_l += 1
    
    Lij = np.zeros((n_l, 2), dtype=np.int_)
    L_vals = np.empty(n_l, dtype=np.float32)
    
    k = 0
    for i, x_pair in enumerate(xs):
        v1, v2 = vs[i]
        # Infinite Voronoi edges have zero length
        if (v1 < 0) | (v2 < 0):
            continue
        
        # Get length of Voronoi ridge
        ell_vor = np.linalg.norm(vertices[v1] - vertices[v2])
        
        # Get length of circles intersection
        ell_cir = circle_intersect_length(pts[x_pair[0]], pts[x_pair[1]], cr)
        
        # Store the lower value of interface length
        L_vals[k] = np.minimum(ell_vor, ell_cir)
        
        # Store point indices
        Lij[k] = x_pair

        k += 1
        
    return Lij, L_vals


def make_L_gaps(vor, cr):
    """
    """
    n = vor.npoints
    xs = vor.ridge_points
    vs = np.array(vor.ridge_vertices)
    pts = vor.points
    vertices = vor.vertices
    
    Lij, L_vals = get_L_vals_gaps(xs, vs, pts, vertices, cr=cr)
    L = csr_matrix((L_vals, (*Lij.T,)), shape=(n, n))

    return L + L.T


def make_L(vor):
    """
    Return a Scipy CSRMatrix object (sparse matrix) encoding 
    the length of cell-cell contacts given a Scipy Voronoi object 
    `vor` describing the Voronoi tesselation of a set of cell centroids.
    """
    
    n = vor.npoints
    xs = vor.ridge_points
    vs = np.array(vor.ridge_vertices)
    vertices = vor.vertices
    
    Lij, L_vals = get_L_vals(n, xs, vs, vertices)
    L = csr_matrix((L_vals, (*Lij.T,)), shape=(n, n))
    
    return L + L.T



@numba.njit
def get_B_vals_gaps(betarho_func, rhos, xs, vs, pts, cr):
    """
    Returns the cell-cell indices and the phenomenological beta-function 
    values given information about the Voronoi tesselation and
    a uniform cell radius `cr`.
    """
    
    # Make matrix cell-cell contact lengths
    n_b = 0
    for v1, v2 in vs:
        if (v1 >= 0) & (v2 >= 0):
            n_b += 1
    
    Bij = np.zeros((n_b, 2), dtype=np.int_)
    B_vals = np.zeros(n_b, dtype=np.float32)
    
    k = 0
    for i, x_pair in enumerate(xs):
        
        # Infinite Voronoi edges have zero length
        v1, v2 = vs[i]
        if (v1 < 0) | (v2 < 0):
            continue
        
        # Check if cell-cell distance is too large to interact
        x1, x2 = x_pair
        bij_bool = np.linalg.norm(pts[x1] - pts[x2]) < (cr * 2)
        
        if bij_bool:
            # Get average beta(rho) and store 
            val = (betarho_func(rhos[x1]) + betarho_func(rhos[x2])) / 2
            B_vals[k] = val
        
        # Store indices
        Bij[k] = x_pair
        k += 1
        
    return Bij, B_vals


def make_B_gaps(vor, cr, betarho_func, rhos):
    """
    """
    n = vor.npoints
    xs = vor.ridge_points
    vs = np.array(vor.ridge_vertices)
    pts = vor.points
    
    Bij, B_vals = get_B_vals_gaps(betarho_func, rhos, xs, vs, pts, cr)
    B = csr_matrix((B_vals, (*Bij.T,)), shape=(n, n))

    return B + B.T


####### Functions for calculating cell-cell contacts 


@numba.njit
def circle_intersect_length(c1, c2, r):
    """Length of the intersection between two circles of equal radius `r`."""
    
    # Get distance between circle centers
    d = np.linalg.norm(c1 - c2)
    
    # Get length of interface, catching the case where circles do not intersect
    ell2 = np.maximum(r**2 - (d**2)/4, 0)
    return 2 * np.sqrt(ell2)
    
@numba.njit
def circle_intersect_length2(c1, c2, r1, r2):
    """Length of the intersection between two circles of radii `r1` and `r2`."""
    
    # Get distance between circle centers
    d = np.linalg.norm(c1 - c2)
    
    # Check if circles do not intersect
    if d >= (r1 + r2):
        return 0
    
    # Else, calculate intersection length
    return 2 * r1 * np.sqrt(1 - ((r1**2 + d**2 - r2**2) / (2 * r1 * d))**2)


@numba.njit
def A_cells_um(nc, rho, A_c_rho1=800):
    """
    Returns the area of `nc` cells at density `rho` in 
    micrometers^2.
    `A_c_rho1` is the area of each cell at `rho=1` in
    micrometers^2.
    """
    return nc * A_c_rho1 / rho


@numba.njit
def beta_to_rad(beta, dist=1):
    return dist/(np.sqrt(4-beta**2))

@numba.njit
def rad_to_beta(rad, dist=1):
    return np.sqrt(4 - (dist**2 / rad**2))


####### General utilities

@numba.njit
def normalize(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)

@numba.njit
def logistic(t, g, rho_0, rho_max):
    """Return logistic equation evaluated at time `t`."""
    return rho_0 * rho_max / (rho_0 + (rho_max - rho_0) * np.exp(-g * t))

####### Cell Adjacency

def gaussian_irad_Adj(
    X, irad, dtype=np.float32, sparse=False, row_stoch=False, **kwargs
):
    """
    Construct adjacency matrix for a non-periodic set of 
    points (cells). Adjacency is determined by calculating pairwise 
    distance and applying a threshold `irad` (interaction radius)
    """
    
    n = X.shape[0]
    d = pdist(X)
    a = scipy.stats.norm.pdf(d, loc=0, scale=irad/2).astype(dtype)
    a[d >= irad] = 0
    A = squareform(a)
    
    if row_stoch:
        rowsum = np.sum(A, axis=1)[:, np.newaxis]
        A = np.divide(A, rowsum)
    else:
        A = A > 0
        
    if sparse:
        A = csr_matrix(A)
    
    return A


def irad_Adj(
    X, irad, dtype=np.float32, sparse=False, row_stoch=False, **kwargs
):
    """
    Construct adjacency matrix for a non-periodic set of 
    points (cells). Adjacency is determined by calculating pairwise 
    distance and applying a threshold `irad` (interaction radius)
    """
    
    n = X.shape[0]
    A = squareform(pdist(X)) <= irad
    A = A - np.eye(n)
    
    if row_stoch:
        rowsum = np.sum(A, axis=1)[:, np.newaxis]
        A = np.divide(A, rowsum)

    if sparse:
        A = csr_matrix(A)
    
    return A


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

def hex_Adj(rows, cols=0, dtype=np.float32, sparse=False, **kwargs):
    """
    """
    # Make hexagonal grid
    X = hex_grid(rows, cols, **kwargs)
    
    # Construct adjacency matrix
    if sparse:
        Adj = make_Adj_sparse(rows, cols, dtype=dtype, **kwargs)
    else:
        Adj = make_Adj(rows, cols, dtype=dtype, **kwargs)
    
    return X, Adj

def make_Adj_sparse(rows, cols=0, dtype=np.float32, **kwargs):
    """Construct adjacency matrix for a periodic hexagonal 
    lattice of dimensions rows x cols.

    Returns a `scipy.sparse.csr_matrix` object."""

    # Check if square
    if cols == 0:
        cols = rows

    # Initialize neighbor indices
    n = rows * cols
    nb_j = np.zeros(6 * n, dtype=int)
    for i in range(cols):
        for j in range(rows):

            # Get neighbors of cell at location i, j
            nb_col, nb_row = np.array(
                [
                    (i    , j + 1),
                    (i    , j - 1),
                    (i - 1, j    ),
                    (i + 1, j    ),
                    (i - 1 + 2*(j%2), j - 1),
                    (i - 1 + 2*(j%2), j + 1),
                ]
            ).T

            nb_col = nb_col % cols
            nb_row = nb_row % rows

            nb = nb_col * rows + nb_row
            nb_j[6*(i * rows + j) : 6*(i * rows + j + 1)] = nb

    nb_i = np.repeat(np.arange(n).astype(int), 6)
    Adj_vals = np.ones(6 * n, dtype=np.float32)

    return csr_matrix((Adj_vals, (nb_i, nb_j)), shape=(n, n))


####### Plot/animate 


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
    ax1=None,
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
    if ax1 is None:
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
        crk,
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
            xlim=xlim,
            ylim=ylim,
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
#     cols, rows = xx.shape
    
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

####### Scipy-interpolated heatmap plotting

def plot_interp_mesh(
    ax,
    X,
    var,
    n_interp=120,
    vmin=None,
    vmax=None,
    cmap="CET_L8",
    title=None,
    axis_off=True,
    xlim=(),
    ylim=(),
    aspect=None,
    pcolormesh_kwargs=dict(),
    **kwargs
):
    ax.clear()
    if axis_off:
        ax.axis("off")
    
    if not xlim:
        xlim=[X[:, 0].min(), X[:, 0].max()]
    if not ylim:
        ylim=[X[:, 1].min(), X[:, 1].max()]
    
    if aspect is None:
        aspect=1
    
    if type(n_interp) is not tuple:
        n_interp_y = n_interp_x = n_interp
    else:
        n_interp_y, n_interp_x = n_interp
    
    rbfi = Rbf(X[:, 0], X[:, 1], var)  # radial basis function interpolator
    xi = np.linspace(*xlim, n_interp_x)
    yi = np.linspace(*ylim, n_interp_y)
    xxi, yyi = np.meshgrid(xi, yi)
    zzi = rbfi(xxi, yyi)               # interpolated values
    
    ax.pcolormesh(
        xxi, 
        yyi, 
        zzi, 
        shading="auto", 
        cmap=cc.cm[cmap],
        vmin=vmin,
        vmax=vmax,
        **pcolormesh_kwargs
    )

#     ax.imshow(
#         zzi, 
#         cmap=cc.cm[cmap], 
#         interpolation='nearest',
#         vmin=vmin,
#         vmax=vmax,
#         **imshow_kwargs
#     )

    ax.set(
        xlim=xlim,
        ylim=ylim,
        aspect=aspect,
        **kwargs
    )

    if title is not None:
        ax.set_title(title)


def animate_interp_mesh(
    X_arr,
    var_t,
    n_interp=120,
    vmin=None,
    vmax=None,
    n_frames=100,
    file_name=None,
    dir_name="plots",
    fps=20,
    cmap="CET_L8",
    title_fun=None,
    axis_off=False,
    aspect=None,
    xlim=(),
    ylim=(),
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
        plot_interp_mesh(
            ax,
            X_t[skip * i],
            var_t[skip * i],
            n_interp=n_interp,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            axis_off=axis_off,
            xlim=xlim,
            ylim=ylim,
            aspect=aspect,
            pcolormesh_kwargs=pcolormesh_kwargs,
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


def inspect_interp_mesh(
    X,
    var,
    n_interp=120,
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
    
    plot_interp_mesh(
        ax,
        X_,
        var_,
        n_interp=n_interp,
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

def inspect_grid_interp_mesh(
    t,
    X_arr,
    var_t,
    nrows,
    ncols,
    vmin,
    vmax,
    n_interp=120,
    xlim=None,
    ylim=None,
    title_fun=None,
    cmap="kgy",
    axis_off=True,
    figsize=(10,6),
    **kwargs
):
    nt = t.size
    nplot = nrows * ncols
    
    if title_fun is None:
        title_fun = lambda i: f"Time = {t[i]:.2f}"
    
    if X_arr.ndim == 2:
        X_t = np.repeat(X_arr[np.newaxis, :], nt, axis=0)
    else:
        X_t = X_arr.copy()
    
    # Render frames
    idx = [int(i) for i in np.linspace(0, nt-1, nplot)]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for ax, i in zip(axs.flat, idx):
        
        title = title_fun(i)
        
        inspect_interp_mesh(
            ax=ax,
            X=X_t[i],
            var=var_t[i], 
            n_interp=n_interp,
            idx=i,
            vmin=vmin, 
            vmax=vmax,
            cmap=cmap,
            xlim=xlim,
            ylim=ylim,
            title=title,
            axis_off=axis_off,
            **kwargs
        )

####### Signaling code

@numba.njit
def beta_rho_isqrt(rho, *args):
    return 1/np.sqrt(np.maximum(rho, 1))

@numba.njit
def beta_rho_exp(rho, m, *args):
    return np.exp(-m * np.maximum(rho - 1, 0))

@numba.njit
def beta_rho_lin(rho, m, *args):
    return m - m * np.maximum(rho, 1) + 1

def tc_rhs_beta_normA(S, S_delay, Adj, sender_idx, beta_func, func_args, alpha, k, p, delta, lambda_, rho):
    """
    Right-hand side of the transciever circuit delay 
    differential equation. Uses a matrix of cell-cell contact 
    lengths `L`.
    """

    # Get signaling as a function of density
    beta = beta_func(rho, *func_args)
    
    # Get input signal across each interface
    S_bar = beta * (Adj @ S_delay)

    # Calculate dE/dt
    dS_dt = (
        alpha
        * (S_bar ** p)
        / (k ** p + (delta * S_delay) ** p + S_bar ** p)
        - S
    )

    # Set sender cell to zero
    dS_dt[sender_idx] = 0

    return dS_dt    
    
####################################################
#######              Figures                 #######
####################################################

####### Maximum propagation area vs. density 

def max_prop_area_vs_density(
    t,
    X,
    sender_idx,
    rho_min,
    rho_max,
    n_rho,
    g,
    rhs,
    dde_args,
    delay,
    thresh,
    where_vars,
    min_delay=5,
    progress_bar=False,
):

    # Get # cells and # time-points
    n = X.shape[0]
    nt = t.size

    # Sample starting densities
    rho_0_space = np.linspace(rho_min, rho_max, n_rho)
    
    # Set initial fluorescence
    S0 = np.zeros(n, dtype=np.float32)
    S0[sender_idx] = 1

    # Initialize results vector
    X_rho0_t = np.empty((n_rho, nt, n, 2), dtype=np.float32)
    S_rho0_t = np.empty((n_rho, nt, n), dtype=np.float32)
    A_rho0_t = np.empty((n_rho, nt), dtype=np.float32)
    rho_rho0_t = np.empty((n_rho, nt), dtype=np.float32)

    iterator = range(n_rho)
    if progress_bar:
        iterator = tqdm.tqdm(iterator)
    for i in iterator:

        # Get parameters
        rho_0 = rho_0_space[i]
        rho_t = logistic(t, g, rho_0=rho_0, rho_max=rho_max)
    #     r_t = 1/np.sqrt(rho_t_)
    #     ell_t = r_t / np.sqrt(3)

        # Simulate
        S_t = integrate_DDE_varargs(
            t,
            rhs,
            var_vals=rho_t,
            dde_args=dde_args,
            E0=S0,
            delay=delay,
            where_vars=where_vars,
            min_delay=min_delay,
        )
        
        rho_rho0_t[i] = rho_t

        r_t = 1/np.sqrt(rho_t)
        X_t = np.array([X] * nt) * r_t[:, np.newaxis, np.newaxis]
        X_rho0_t[i] = X_t
        
        S_rho0_t[i] = S_t
        
        n_act = (S_t > thresh).sum(axis=1)
        A_rho0_t[i] = A_cells_um(n_act, rho_t)
    
    results = dict(
        t=t, 
        dde_args=dde_args,
        rho_0_space=rho_0_space,
        rho_rho0_t=rho_rho0_t,
        X_rho0_t=X_rho0_t, 
        S_rho0_t=S_rho0_t,
        A_rho0_t=A_rho0_t,
    )
    
    return results

def inspect_max_prop_results(
    results,
    run,
    rows,
    cols,
    nrows=2,
    ncols=4,
    cmap="kgy",
    set_xylim="final",
    xlim=(),
    ylim=(),
    vmax=None,
    sender_idx=np.nan,
    vmax_mult_k=1,
    which_k=1,
):

    t, X_t, S_t, rho_t = (
        results["t"],
        results["X_rho0_t"][run],
        results["S_rho0_t"][run],
        results["rho_rho0_t"][run],
    )

    if set_xylim == "final":
        X_fin = X_t[-1]
        xlim_ = X_fin[:, 0].min() * 0.95, X_fin[:, 0].max() * 0.95
        ylim_ = X_fin[:, 1].min() * 0.95, X_fin[:, 1].max() * 0.95
    elif set_xylim == "fixed":
        xlim_, ylim_ = xlim, ylim
    else:
        xlim_ = ylim_ = None

    if vmax is None:
        vmax = S_t.max()
    elif vmax == "tc_only":
        S_t_tc = S_t.copy()
        S_t_tc[:, sender_idx] = 0
        vmax = S_t_tc.max()
    elif vmax == "mult_k":
        vmax = results["dde_args"][which_k] * vmax_mult_k
    
    # Render frames
    nplot = nrows * ncols
    idx = [int(i) for i in np.linspace(0, t.size - 1, nplot)]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 6))
    for ax, i in zip(axs.flat, idx):
        title = f"Time = {t[i]:.2f}, " + r"$\rho$" + f" = {rho_t[i]:.2f}"
        inspect_colormesh(
            ax=ax,
            X=X_t[i],
            rows=rows,
            cols=cols,
            var=S_t,
            idx=i,
            vmin=0,
            vmax=vmax,
            cmap=cmap,
            xlim=xlim_,
            ylim=ylim_,
            title=title,
        )


def plot_max_prop_results(results):

    rho_0_space, A_rho0_t = (
        results["rho_0_space"],
        results["A_rho0_t"],
    )
    
    # Get max activated area for each condition
    max_area = A_rho0_t.max(axis=1)
    
    # Make data
    data = {
        "max_area": max_area,
        "rho_0": rho_0_space * 1250,
    }
    
    plot = hv.Scatter(
        data=data,
        kdims=["rho_0"], 
        vdims=["max_area"],
    ).opts(
        color=cc.glasbey_category10[2],
        xlabel=r"plating density ($mm^2$)",
        xlim=(1000, 5250),
        xticks=[1250, 2500, 3750, 5000],
        ylabel=r"max. activated area ($\mu m^2$)",
    #     ylim=(   0, 200000),
        ylim=(0,None),
#         title="Inverse sqrt, delta = 0"
    )
    
    return plot


def run_max_prop_area(*args, **kwargs):
    
    results = max_prop_area_vs_density(*args, **kwargs)
    plot = plot_max_prop_results(results)
    
    return results, plot

def save_max_prop_video(
    results,
    run,
    rows,
    cols,
    file_name,
    dir_name="plots",
    cmap="kgy",
    set_xylim="final",
    n_interp=100,
    n_frames=100,
    fps=15, 
    xlim=(),
    ylim=(),
    vmax=None,
    sender_idx=np.nan,
    vmax_mult_k=1,
    which_k=1,
    anim_kwargs=dict(),
    **kwargs
):

    t, X_t, S_t, rho_t = (
        results["t"],
        results["X_rho0_t"][run],
        results["S_rho0_t"][run],
        results["rho_rho0_t"][run],
    )

    if set_xylim == "final":
        X_fin = X_t[-1]
        xlim_ = X_fin[:, 0].min() * 0.95, X_fin[:, 0].max() * 0.95
        ylim_ = X_fin[:, 1].min() * 0.95, X_fin[:, 1].max() * 0.95
    elif set_xylim == "fixed":
        xlim_, ylim_ = xlim, ylim
    elif set_xylim == "fit":
        X_t = X_t[0]
        xlim_ = X_t[:, 0].min() * 0.95, X_t[:, 0].max() * 0.95
        ylim_ = X_t[:, 1].min() * 0.95, X_t[:, 1].max() * 0.95
    else:
        xlim_ = ylim_ = None

    if vmax is None:
        vmax = S_t.max()
    elif vmax == "tc_only":
        S_t_tc = S_t.copy()
        S_t_tc[:, sender_idx] = 0
        vmax = S_t_tc.max()
    elif vmax == "mult_k":
        vmax = results["dde_args"][which_k] * vmax_mult_k
    
    # Function for plot title
    title_fun = lambda i: f"Time = {t[i]:.2f}, " + r"$\rho$" + f" = {rho_t[i]:.2f}"

    # Make video
    animate_interp_mesh(
        X_arr=X_t,
        var_t=S_t,
        n_interp=n_interp,
        n_frames=n_frames,
        file_name=file_name,
        dir_name=dir_name,
        fps=fps, 
        vmin=0, 
        vmax=vmax, 
        cmap="kgy",
        title_fun=title_fun,
        xlim=xlim_,
        ylim=ylim_,
        **anim_kwargs,
    )

####### Biphasic propagation




####### Drug effects




####### Basal promoter activity


def basal_activity_phase(
    t,
    n,
    log_lambda_minmax,
    log_alpha_minmax,
    n_lambda,
    n_alpha,
    n_reps,
    rhs,
    dde_args,
    delay,
    thresh,
    where_lambda=4,
    where_alpha=0,
    min_delay=5,
    seed=2021,
    progress_bar=True,
):
    
    # Sample free parameters
    lambda_space = np.logspace(*log_lambda_minmax, n_lambda)
    alpha_space  = np.logspace(*log_alpha_minmax, n_alpha)
    rep_space    = np.arange(n_reps)
    free_params  = (rep_space, lambda_space, alpha_space)

    # Get all pairwise combinations of free parameters
    param_space = np.meshgrid(*free_params)
    param_space = np.array(param_space, dtype=np.float32).T.reshape(-1, len(free_params))

    # Get indices for pairwise combinations
    param_idx = np.meshgrid(*[np.arange(p.size) for p in free_params])
    param_idx = np.array(param_idx, dtype=int).T.reshape(-1, len(free_params))

    # Set seed
    np.random.seed(seed)

    # Minimally perturb initial fluorescence
    rv_shape = n_lambda, n_reps, n
    rv_mean  = np.sqrt(np.pi/2) * lambda_space[:, np.newaxis, np.newaxis]
    init_arr = scipy.stats.halfnorm.rvs(0, rv_mean, rv_shape)
    init_arr = np.transpose(init_arr, (1, 0, 2))

    # Get parameters
    args = list(dde_args)
    
    # Initialize results vector
    S_fin = np.empty((n_reps, n_lambda, n_alpha, n), dtype=np.float32)

    iterator = range(param_space.shape[0])
    if progress_bar:
        iterator = tqdm.tqdm(iterator)
    
    for i in iterator:

        # Get parameters
        rep, li, ai = param_idx[i]
        lambda__, alpha_ = lambda_space[li], alpha_space[ai]

        # Package parameters
        args[where_lambda] = lambda__
        args[where_alpha]  = alpha_

        # Get initial conditions
        S0 = init_arr[rep, li]

        # Simulate
        S_t = integrate_DDE(
            t,
            rhs,
            dde_args=args,
            E0=S0,
            delay=delay,
            min_delay=min_delay,
        )

        # Store endpoint of simulation
        S_fin[rep, li, ai] = S_t[-1]

    # Calculate % of cells activated at final time-point
    S_activated = S_fin > thresh
    S_act_prop = S_activated.mean(axis=(0, 3))

    # Calculate % of cells activated at final time-point
    S_mean = S_fin.mean(axis=(0, 3))

    results = dict(
        t=t, 
        n=n,
        S_fin=S_fin,
        S_activated=S_activated,
        S_act_prop=S_act_prop,
        S_mean=S_mean,
        dde_args=dde_args,
        lambda_space=lambda_space,
        alpha_space=alpha_space,
        param_space=param_space,
        init_arr=init_arr,
    )
    
    return results


def basal_activity_plots(results):
    
    # Retrieve relevant results
    lambda_space, alpha_space, S_act_prop, S_mean = (
        results["lambda_space"],
        results["alpha_space"],
        results["S_act_prop"],
        results["S_mean"],
    )
    
    # Construct and return plots
    data = {
        "lambda": lambda_space, 
        "alpha": alpha_space, 
        "% cells activated": S_act_prop.T * 100,
        "mean_fluorescence": S_mean.T,
    }

    p1 = hv.QuadMesh(
        data=data,
        kdims=["lambda", "alpha"],
        vdims=["% cells activated"]
    ).opts(
        logx=True, 
        logy=True,
    #     color = "% activation", 
        cmap="kb",
    #     aspect=2.2,
        xlabel="λ",
        ylabel="α",
    #     title="Genome insertion site can influence \nself-activation of transceiver clones",
        colorbar=True,
    )

    p2 = hv.QuadMesh(
        data=data,
        kdims=["lambda", "alpha"],
        vdims=["mean_fluorescence"]
    ).opts(
        logx=True, 
        logy=True,
    #     color = "% activation", 
        cmap="viridis",
    #     aspect=2.2,
        xlabel="λ",
        ylabel="α",
        clabel="mean fluorescence",
    #     title="Genome insertion site can influence \nself-activation of transceiver clones",
        colorbar=True,
    ).redim.range(mean_fluorescence=(0, 1))
    
    t1 = hv.Text(2e-5, 6e-1, 'Inducible', fontsize=14).opts(color="white")
    t2 = hv.Text(5e-3, 5e0, 'Self-\nactivating', fontsize=14).opts(color="white")
    t3 = hv.Text(5e-3, 5e0, 'Self-\nactivating', fontsize=14).opts(color="black")
    
    return p1, p1 * t1 * t2, p2, p2 * t1 * t3


def run_basal_activity(
    *args,
    **kwargs,
):
    results = basal_activity_phase(*args, **kwargs)
    plots   = basal_activity_plots(results)
    
    return results, plots

####### Inhibitor gradient



####### Density vs. inhibitor effect



####### 

    
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


def get_center_cells(X, n_center=1):
    """Returns indices of the n_cells cells closest to the origin given their coordinates as an array X."""
    return np.argpartition([np.linalg.norm(x) for x in X], n_center)[:n_center]


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

