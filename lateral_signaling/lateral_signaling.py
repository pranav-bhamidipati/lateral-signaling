####### Load depenendencies
import os
import warnings

import numpy as np
import pandas as pd
from math import ceil
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
from scipy.sparse import csr_matrix, identity, diags
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import Rbf
import scipy.stats

import numba
import tqdm
import time

from matplotlib import animation
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry import Polygon, Point
from descartes import PolygonPatch

import holoviews as hv
import colorcet as cc
import matplotlib as mplF
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
hv.extension('matplotlib')


####### Differential equation right-hand-side functions

def signal_rhs(
    S,
    S_delay,
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
    rho,
):
    """
    Right-hand side of the transciever circuit delay differential
    equation. Uses a matrix of cell-cell adjacency `Adj`.
    """

    # Get signaling as a function of density
    beta = beta_func(rho, *beta_args)

    # Get input signal across each interface
    S_bar = beta * (Adj @ S_delay)

    # Calculate dE/dt
    dS_dt = (
        lambda_
        + alpha * (S_bar ** p) / (k ** p + (delta * S_delay) ** p + S_bar ** p)
        - S
    )

    # Set sender cell to zero
    dS_dt[sender_idx] = 0

    return dS_dt




####### General utils

# Vectorized integer ceiling
ceiling = np.vectorize(ceil)

# Vectorized rounding
vround = np.vectorize(round)

@numba.njit
def normalize(x, xmin, xmax):
    """Normalize `x` given explicit min/max values. """
    return (x - xmin) / (xmax - xmin)


@numba.njit
def logistic(t, g, rho_0, rho_max):
    """Returns logistic equation evaluated at time `t`."""
    return rho_0 * rho_max / (rho_0 + (rho_max - rho_0) * np.exp(-g * t))


@numba.njit
def logistic_inv(rho, g, rho_0, rho_max):
    """Inverse of logistic equation (returns time at a given density `rho`)."""
    return np.log(rho * (rho_max - rho_0) / (rho_0 * (rho_max - rho))) / g


@numba.njit
def cart2pol(xy):
    r = np.linalg.norm(xy)
    x, y = xy.flat
    theta = np.arctan2(y, x)
    return np.array([r, theta])


@numba.njit
def pol2cart(rt):
    r, theta = rt.flat
    return r * np.array([np.cos(theta), np.sin(theta)])


####### Color utils

# Color swatches

pinks = [
    "#fde0dd",
    "#fa9fb5",
    "#c51b8a",
]

greens = [
    "#e5f5e0",
    "#a1d99b",
    "#31a354",
    "#006d2c",
]

yob = [
    "#fff7bc",
    "#fec44f",
    "#d95f0e",
    "#9a3403",
]

cols_blue = [
    "#2e5a89",
    "#609acf",
    "#afc4cd",
]

cols_teal = [
    "#37698a",
    "#1dab99",
    "#5c6c74",
]

cols_green = [
    "#37698a",
    "#80c343",
    "#9cb6c1",
]

cols_red = [
    "#e16566",
    "#efc069",
    "#b0c5cd",
]

purple = "#8856a7"
gray   = "#aeaeae"
black  = "#060605"

col_light_gray = "#eeeeee"
col_gray       = "#aeaeae"
col_black      = "#060605"

# Make a custom version of the "KGY" colormap
kgy_original = cc.cm["kgy"]
kgy = ListedColormap(kgy_original(np.linspace(0, 0.92, 256)))


def rgb_as_int(rgb):
    """Coerce RGB iterable to a tuple of integers"""
    if any([v >= 1. for v in rgb]):
        _rgb = tuple(rgb)
    else:
        _rgb = tuple((round(255 * c) for c in rgb))
    
    return _rgb

    
def rgb_as_float(rgb):
    """Coerce RGB iterable to an ndarray of floats"""
    if any([v >= 1. for v in rgb]) or any([type(v) is int for v in rgb]):
        _rgb = (np.asarray(rgb) / 255).astype(float)
    else:
        _rgb = np.asarray(rgb).astype(float)
    
    return _rgb


def sample_cycle(cycle, size): 
    """Sample a continuous colormap at regular intervals to get a linearly segmented map"""
    return hv.Cycle(
        [cycle[i] for i in ceiling(np.linspace(0, len(cycle) - 1, size))]
    )

def hex2rgb(h):
    """Convert 6-digit hex code to RGB values (0, 255)"""
    h = h.lstrip('#')
    return tuple(int(h[(2*i):(2*(i + 1))], base=16) for i in range(3))


def rgba2hex(rgba, background=(255, 255, 255)):
    """
    Adapted from StackOverflow
    ------------------
    
    Question: Convert RGBA to RGB in Python
    Link: https://stackoverflow.com/questions/50331463/convert-rgba-to-rgb-in-python/50332356
    Asked: May 14 '18 at 13:25
    Answered: Nov 7 '19 at 12:40
    User: Feng Wang
    """
    
    rgb = np.zeros((3,), dtype=np.uint8)
    *_rgb, a = rgba

    for i, _c in enumerate(_rgb):
        
        # Convert vals in [0., 1.] to [0, 255]
        if _c <= 1.:
            c = int(_c * 255)
        else:
            c = _c
        
        # Calculate new values
        new_c = round(a * c + (1 - a) * background[i])
        rgb[i] = new_c
    
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _hexa2hex(h, alpha, background="#ffffff"):
    """
    Returns hex code of the color observed when displaying 
    color `h` (in hex code) with transparency `alpha` on a 
    background color `background` (default white)
    """
    
    # Convert background to RGB
    bg = hex2rgb(background)
    
    # Get color in RGBA
    rgba = *hex2rgb(h), alpha
    
    # Convert to HEX without transparency
    return rgba2hex(rgba, bg)

# Vectorized function to convert HEX and alpha to HEX given background
hexa2hex = np.vectorize(_hexa2hex)


####### Lattice generation and adjacency

# Vertices of a regular hexagon centered at (0,0) with width 1.
_hex_vertices = (
    np.array([
        np.cos(np.arange(0, 2 * np.pi, np.pi / 3) + np.pi / 6), 
        np.sin(np.arange(0, 2 * np.pi, np.pi / 3) + np.pi / 6),
    ]).T 
    / np.sqrt(3)
)

_hex_x, _hex_y = _hex_vertices.T

def hex_grid(rows, cols=0, r=1., sigma=0, **kwargs):
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
        X = np.array([np.random.normal(loc=x, scale=sigma*r) for x in X])
    
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


def k_step_Adj(k, rows, cols=0, dtype=np.float32, row_stoch=False, **kwargs):
    """
    """
    
    if not cols:
        cols = rows
        
    # Construct adjacency matrix
    a = make_Adj_sparse(rows, cols, dtype=dtype, **kwargs)
    
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

def hex_Adj(rows, cols=0, dtype=np.float32, sparse=False, row_stoch=False, **kwargs):
    """
    """
    # Make hexagonal grid
    X = hex_grid(rows, cols, **kwargs)
    
    # Construct adjacency matrix
    if sparse:
        A = make_Adj_sparse(rows, cols, dtype=dtype, **kwargs)
        
        # Make row-stochastic (rows sum to 1)
        if row_stoch:
            
            # Calculate inverse of rowsum
            inv_rowsum = diags(np.array(1/A.sum(axis=1)).ravel())
            
            # Multiply by each row
            A = np.dot(inv_rowsum, A)
            
    else:
        A = make_Adj(rows, cols, dtype=dtype, **kwargs)

        # Make row-stochastic (rows sum to 1)
        if row_stoch:
            rowsum = np.sum(A, axis=1)[:, np.newaxis]
            A = np.divide(A, rowsum)
    
    return X, A

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


######### Spatial utils


######### Statistics

def data_to_hist(d, bins, data_range=(0, 1000)):
    """Convert sampled data to a frequency distribution (histogram)"""
    return np.histogram(d, bins=bins, range=data_range)


def ecdf_vals(d):
    """
    Returns the empirical CDF values of a data array `d`. 
    
    Arguments:
    d : 1d_array
        Empirical data values
    
    Returns:
    x : 1d_array
        sorted `d`
    y : 1d_array
        Empirical cumulative distribution values
    """
    x = np.sort(d)
    y = np.linspace(1, 0, x.size, endpoint=False)[::-1]
    return x, y


##### Image and ROI functions


def rescale_img(im, interval=(None, None), dtype=np.float64, imask_val=0):
    """
    Returns an image with intensity values rescaled to the range (0,1).
    """
    
    # Get
    _im = im.copy()
    
    # Get min and max for rescaling
    if interval[0] is None:
        interval = (_im.min(), interval[1])
    if interval[1] is None:
        interval = (interval[0], _im.max())
    
    # Perform rescaling
    _im = (_im - interval[0]) / (interval[1] - interval[0])
    
    # Clip values to range [0, 1]
    _im = np.maximum(np.minimum(_im, 1), 0)
    
    return _im


def rescale_masked_img(im, mask, interval=(None, None), dtype=np.float64, imask_val=0):
    """
    Returns an image with intensity values inside a mask rescaled to the 
    range (0,1) and values outside the max set to a constant value.
    """
    
    # Get masked intensity values
    vals = im[mask]
    
    # Get min and max for rescaling
    if interval[0] is None:
        interval = (vals.min(), interval[1])
    if interval[1] is None:
        interval = (interval[0],vals.max())
        
    # Perform rescaling
    vals = (vals - interval[0]) / (interval[1] - interval[0])
    vals = np.maximum(np.minimum(vals, 1), 0)

    # Construct output
    imf = np.ones_like(im, dtype=dtype) * imask_val
    imf[mask] = vals
    
    return imf


@numba.njit
def get_lp_corners(src, dst, width):
    """Given source and destination points, return the coordinates of the corners 
    of the line profile along that line segment with specified width."""
    
    assert (src.dtype is dst.dtype), "src and dst should have the same dtype"
    
    dtype = src.dtype
    src = src.ravel()
    dst = dst.ravel()
    
    # Get slope perpendicular to the line segment
    pslope = - (dst[0] - src[0]) / (dst[1] - src[1])
    
    # Get increments in x and y direction
    dx = width / (2 * np.sqrt(1 + pslope ** 2))
    dy = pslope * dx
    
    # Add/subtract increments from each point
    corners = np.empty((4, 2), dtype=dtype)
    corners[:, 0]  =  dx
    corners[:, 1]  =  dy
    corners[0]    *=  -1
    corners[3]    *=  -1
    corners[:2]   += src
    corners[2:]   += dst
    
    return corners


def transform_point(point, center1, radius1, center2, radius2):
    """
    Convert a point on one circle to its transformed location on another circle.
    """    
    pt = (point - center1) * (center2 / center1) + center2
    return pt.ravel()


@numba.njit
def verts_to_circle(xy):
    """
    Finds the least-squares estimate of the center of
    a circle given a set of points in R^2.
    
    Parameters
    ----------
    xy  :  (N x 2) Numpy array, float
        (x, y) coordinates of points sampled from the
            edge of the circle
           
    Returns
    -------
    xy_c  :  (2,) Numpy array, float
        (x, y) coordinates of least-squares estimated
            center of circle
    
    R  :  float
        Radius of circle
        
    Source: "Least-Squares Circle Fit" by Randy Bullock (bullock@ucar.edu)
    Link:   https://dtcenter.org/sites/default/files/community-code/met/docs/write-ups/circle_fit.pdf
    """
    
    # Unpack points
    N = xy.shape[0]
    x, y = xy.T
    
    # Get mean x and y
    xbar, ybar = x.mean(), y.mean()
    
    # Transform to zero-centered coordinate system
    u, v = x - xbar, y - ybar
    
    # Calculate sums used in estimate
    Suu  = np.sum(u**2)
    Suv  = np.sum(u * v)
    Svv  = np.sum(v**2)
    
    Suuu = np.sum(u**3)
    Suuv = np.sum((u**2) * v)
    Suvv = np.sum(u * (v**2))
    Svvv = np.sum(v**3)
    
    # Package sums into form Aw = b
    A = np.array([
        [Suu, Suv],
        [Suv, Svv]
    ])
    b = 1 / 2 * np.array([
        [Suuu + Suvv],
        [Svvv + Suuv],
    ])
    
    # Solve linear system for center coordinates
    uv_c = (np.linalg.pinv(A) @ b).ravel()
    
    # Calculate center and radius
    xy_c = uv_c + np.array([xbar, ybar])
    R = np.sqrt((uv_c ** 2).sum() + (Suu + Svv) / N)
    
    return xy_c, R


# @numba.njit
def make_circular_mask(h, w, center=None, radius=None):
    """
    Construct a mask to select elements within a circle
    
    Source: User `alkasm` on StackOverflow
    Link:   https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
    """
    
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

# ####### Constants

# # Measured carrying capacity density of transceiver cell line
# #   in dimensionless units. 
# rho_max_ = 5.63040245

# # Length of one dimensionless distance unit in microns
# length_scale = np.sqrt(
#     8 / (3 * np.sqrt(3)) 
#     / (1250)  # cells per mm^2
#     * (1e6)   # mm^2  per μm^2
# )

####### Unit conversions

@numba.njit
def t_to_units(dimless_time, ref_growth_rate=7.28398176e-01):
    """Convert dimensionless time to real units for a growth process.
    
    Returns
    -------
    time  :  number or numpy array (dtype float)
        Time in units (hours, days, etc.)
    
    Parameters
    ----------
    
    dimless_time  :  number or numpy array
        Time in dimensionless units. An exponentially growing
        function (e.g. cell population) grows by a factor of `e` 
        over 1 dimensionless time unit.
    
    ref_growth_rate  :  float
        The rate of growth, in the user's defined units. An exponentially
        growing function (e.g. cell population) grows by a factor of `e` 
        over a time of `1 / growth_rate`.
        Defaults to 7.28398176e-01 days.
    """
    return dimless_time / ref_growth_rate


@numba.njit
def g_to_units(dimless_growth_rate, ref_growth_rate=7.28398176e-01):
    """Convert dimensionless growth rate to real units for a growth process.
    
    Returns
    -------
    growth_rate  :  number or numpy array (dtype float)
        Time in units (hours, days, etc.)
    
    Parameters
    ----------
    
    dimless_growth_rate  :  number or numpy array
        Time in dimensionless units. An exponentially growing
        function (e.g. cell population) grows by a factor of `e` 
        over 1 dimensionless time unit.
    
    ref_growth_rate  :  float
        The rate of growth, in units of `1 / Time`. An exponentially
        growing function (e.g. cell population) grows by a factor of `e` 
        over a time of `1 / growth_rate`.
        Defaults to 7.28398176e-01 days.
    """
    return dimless_growth_rate * ref_growth_rate

@numba.njit
def rho_to_units(rho, ref_density=1250):
    """Convert dimensionless growth rate to real units for a growth process.
    
    Returns
    -------
    density  :  number or numpy array (dtype float)
        Time in units (hours, days, etc.)
    
    Parameters
    ----------
    
    rho  :  number or numpy array
        Cell density in dimensionless units.
    
    ref_density  :  number or numpy array
        The cell density at a dimensionless density of `rho = 1`. 
        Defaults to 1250 cells / mm^2.
    """
    return rho * ref_density


@numba.njit
def ncells_to_area(ncells, rho, ref_density=1250):
    """Return theoretical area taken up by `ncells` cells.
    
    Returns
    -------
    area  :  number or numpy array (dtype float)
        Area in units (mm^2, μm^2, etc.)
    
    Parameters
    ----------
    
    ncells  :  number or numpy array (dtype int)
        Number of cells
    
    rho  :  number or numpy array
        Cell density in dimensionless units.
    
    ref_density  :  number or numpy array
        The cell density at a dimensionless density of `rho = 1`
        in units of inverse area. Defaults to 1250 (mm^-2).
    """
    return ncells / (rho * ref_density)


####### Delay diff eq integration

def get_DDE_rhs(func, *func_args):
    """
    Returns a function `rhs` with call signature 
    
      rhs(E, E_delay, *dde_args) 
      
    that can be passed to `lsig.integrate_DDE` and 
    `lsig.integrate_DDE_varargs`. This is equivalent 
    to calling
    
      func(E, E_delay, *func_args, *dde_args)
    
    Examples of args in `func_args` include:
    
    Adj         :  Adjacency matrix encoding cell neighbors
    sender_idx  :  Index (indices) of sender cells, which
                     undergo different signaling 
    
    """

    def rhs(E, E_delay, *dde_args):
        return func(E, E_delay, *func_args, *dde_args)

    return rhs

def integrate_DDE(
    t_span,
    rhs,
    dde_args,
    E0,
    delay,
    sender_idx=None,
    progress_bar=False,
    min_delay=5,
    past_state="senders_off",
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
    E_save = np.zeros((n_t, n_c), dtype=np.float32)
    E_save[0] = E = E0
    
    if "senders_on".startswith(past_state):
        past_func = lambda E_save, step: E_save[max(0, step - step_delay)]
    
    elif "senders_off".startswith(past_state):
        def past_func(E_save, step): 
            E_past = E_save[max(0, step - step_delay)].copy()
            E_past[sender_idx] = E_past[sender_idx] * (step >= step_delay)
            return E_past
        
    elif "zero".startswith(past_state):
        past_func = lambda E_save, step: np.zeros_like(E_save)
    
    # Construct time iterator
    iterator = np.arange(1, n_t)
    if progress_bar:
        iterator = tqdm.tqdm(iterator)
    
    for step in iterator:
        
        # Get past E
        E_delay = past_func(E_save, step)
        
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
    sender_idx=None,
    progress_bar=False,
    min_delay=5,
    varargs_type="1darray",
    past_state="senders_off",
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
    E_save = np.zeros((n_t, n_c), dtype=np.float32)
    E_save[0] = E = E0
    
    if "senders_on".startswith(past_state):
        past_func = lambda E_save, step: E_save[max(0, step - step_delay)]
    
    elif "senders_off".startswith(past_state):
        def past_func(E_save, step): 
            E_past = E_save[max(0, step - step_delay)].copy()
            E_past[sender_idx] = E_past[sender_idx] * (step >= step_delay)
            return E_past
        
    elif "zero".startswith(past_state):
        past_func = lambda E_save, step: np.zeros_like(E_save)
    
    # Coax variable arguments into appropriate iterable type
    if varargs_type.startswith("1darray"):
        
        # Make variable args a 2D array of appropriate shape
        vvals = np.atleast_2d(var_vals).T

    elif varargs_type.startswith("list"):
        
        # Just make sure it's a list
        if type(var_vals) != "list":
            vvals = list(var_vals)
        else:
            vvals = var_vals
    
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
        E_delay = past_func(E_save, step)
        
        # Get past variable value(s)
        for i, vi in enumerate(vidx):
            past_step = max(0, step - step_delay)
            dde_args[vi] = vvals[i][past_step]
        
        # Integrate
        dE_dt = rhs(E, E_delay, *dde_args)
        E = np.maximum(0, E + dE_dt * dt) 
        E_save[step] = E
    
    return E_save


##### Visualization

def ecdf(d, *args, **kwargs):
    """Construct an ECDF from 1D data array `d`"""
    x = np.sort(d)
    y = np.linspace(1, 0, x.size, endpoint=False)[::-1]
    return hv.Scatter(np.array([x, y]).T, *args, **kwargs)


def remove_RB_spines(plot, element):
    """Hook to remove right and bottom spines from Holoviews plot"""
    plot.state.axes[0].spines["right"].set_visible(False)
    plot.state.axes[0].spines["bottom"].set_visible(False)
    
    
def remove_RT_spines(plot, element):
    """Hook to remove right and top spines from Holoviews plot"""
    plot.state.axes[0].spines["right"].set_visible(False)
    plot.state.axes[0].spines["top"].set_visible(False)


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


def plot_hex_sheet(
    ax,
    X,
    var,
    r=1.,
    vmin=None,
    vmax=None,
    cmap="CET_L8",
    ec=None,
    title=None,
    poly_kwargs=dict(),
    xlim=(),
    ylim=(),
    axis_off=True,
    aspect=None,
    sender_idx=np.array([], dtype=int),
    sender_clr="#e330ff",
    colorbar=False,
    cbar_aspect=20,
    cbar_kwargs=dict(),
    extend=None,
    poly_padding=0.1,
    **kwargs
):
    
    # Clear axis (allows you to reuse axis when animating)
#     ax.clear()
    if axis_off:
        ax.axis("off")
    
    # Get min/max values in color space
    if vmin is None:
        vmin = var.min()
    if vmax is None:
        vmax = var.max()
    
    # Get colors based on supplied values of variable
    if type(cmap) is str:
        _cmap = cc.cm[cmap]
    else:
        _cmap = cmap
    colors = _cmap(normalize(var, vmin, vmax))
    
    # Replace sender(s) with appropriate color
    if isinstance(sender_clr, str):
        _sender_clr = hex2rgb(sender_clr)
    else:
        _sender_clr = list(sender_clr)
        
    if len(_sender_clr) == 3:
        _sender_clr = [*rgb_as_float(_sender_clr), 1.]
    else:
        _sender_clr = [rgb_as_float(_sender_clr[:3]), _sender_clr[3]]
    
    colors[sender_idx] = _sender_clr
    
    # Make polygon size slightly larger than cell 
    #  so there's no visual gaps between cells
    _r = r * (1 + poly_padding)
    
    # Plot cells as polygons
    for i, (x, y, c) in enumerate(zip(*X.T, colors)):
        ax.fill(
            _r * _hex_x + x, 
            _r * _hex_y + y, 
            fc=c, 
            ec=ec, 
        )
        
    # Set figure args, accounting for defaults
    if title is not None:
        ax.set_title(title)
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
    )

    if colorbar:
        
        # Calculate colorbar extension if necessary
        if extend is None:
            n = var.shape[0]        
            ns_mask = ~np.isin(np.arange(n), sender_idx)
            is_under_min = var.min(initial=0.0, where=ns_mask) < vmin
            is_over_max  = var.max(initial=0.0, where=ns_mask) > vmax
            extend = ("neither", "min", "max", "both")[is_under_min + 2 * is_over_max]
    
        # Construct colorbar
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin, vmax), 
                cmap=cmap_), 
            ax=ax,
            aspect=cbar_aspect,
            extend=extend,
            **cbar_kwargs
        )


def plot_var(
    ax,
    i,
    X,
    var,
    cell_radii=None,
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
    sender_idx=np.array([], dtype=int),
    sender_clr=("bmw", 150),
    colorbar=False,
    cbar_aspect=20,
    cbar_kwargs=dict(),
    extend=None,
    **kwargs
):
    ax.clear()
    if axis_off:
        ax.axis("off")
        
    vor = Voronoi(X)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    
    if cell_radii is None:
        cell_radii = np.ones(vor.npoints) * 5
    
    if vmin is None:
        vmin = var.min()
    if vmax is None:
        vmax = var.max()
    
    if type(cmap) is str:
        cmap_ = cc.cm[cmap]
    else:
        cmap_ = cmap
    cols = cmap_(normalize(var, vmin, vmax))
    sender_col = cc.cm[sender_clr[0]](sender_clr[1] / 256)
    for j, region in enumerate(regions):
        poly = Polygon(vertices[region])
        circle = Point(X[j]).buffer(cell_radii[j])
        cell_poly = circle.intersection(poly)
        if cell_poly.area != 0:
            if j in sender_idx:
                ax.add_patch(PolygonPatch(cell_poly, fc=sender_col, **ppatch_kwargs))
            else:
                ax.add_patch(PolygonPatch(cell_poly, fc=cols[j], **ppatch_kwargs))
    
    if plot_ifc:
        pts = [Point(*x).buffer(rad) for x, rad in zip(X, cell_radii)]
        Adj = make_Adj(int(np.sqrt(X.shape[0])))
        infcs = []
        for i, j in zip(*Adj.nonzero()):
            if (j - i > 0) & (pts[i].intersects(pts[j])):
                infc = pts[i].boundary.intersection(pts[j].boundary)
                infcs.append(infc)
        ax.add_collection(LineCollection(infcs, color=ifcc, **lcoll_kwargs))
    
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
        )
    
    if title is not None:
        ax.set_title(title)

    if extend is None:
        
        # Extend colorbar if necessary
        n = var.shape[0]        
        ns_mask = ~ np.isin(np.arange(n), sender_idx)
        is_under_min = var.min(initial=0.0, where=ns_mask) < vmin
        is_over_max  = var.max(initial=0.0, where=ns_mask) > vmax
        extend = ("neither", "min", "max", "both")[is_under_min + 2 * is_over_max]
        
    if colorbar:
        
        # Construct colorbar
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin, vmax), 
                cmap=cmap_), 
            ax=ax,
            aspect=cbar_aspect,
            extend=extend,
            **cbar_kwargs
        )

        
def animate_var(
    X,
    var_t,
    cell_radii=None,    
    n_frames=100,
    file_name=None,
    dir_name="plots",
    path=None,
    xlim=None,
    ylim=None,
    fps=20,
    vmin=None,
    vmax=None,
    #     ec="red",
    cmap="CET_L8",
    sender_idx=np.array([], dtype=int),
    sender_clr=("bmw", 150),
    ifcc="red",
    ppatch_kwargs=dict(edgecolor="gray"),
    lcoll_kwargs=dict(),
    title_fun=None,
    plot_ifc=False,
    colorbar=False,
    cbar_aspect=20,
    cbar_kwargs=dict(),
    extend=None,
    **kwargs
):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    skip = int((var_t.shape[0]) / n_frames)
    
    if title_fun is not None:
        tf=title_fun
    else:
        tf=lambda x: None
    
    if cell_radii is None:
        cell_radii = np.ones(X.shape[0]) * 5
    
    def anim(i, init_cbar=False):
        if cell_radii.ndim == 2:
            cr = cell_radii[skip * i]
        else:
            cr = cell_radii.copy()
        
        plot_var(
            ax,
            skip * i,
            X,
            var_t[skip * i],
            cell_radii=cr,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ifcc=ifcc,
            ppatch_kwargs=ppatch_kwargs,
            lcoll_kwargs=lcoll_kwargs,
            title=tf(skip * i),
            plot_ifc=plot_ifc,
            sender_idx=sender_idx,
            sender_clr=sender_clr,
            colorbar=init_cbar,
            cbar_aspect=cbar_aspect,
            cbar_kwargs=cbar_kwargs,
            extend=extend,
            **kwargs,
        )
        
    # Initialize colorbar
    if colorbar:
        anim(0, True);
    
    # Construct default file path if `path` not supplied
    if path is None:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)  
        if file_name is None:
            file_name = "animation_%d" % time.time()
        vpath = os.path.join(dir_name, file_name)
    else:
        vpath = path
    
    # Add extension
    if not vpath.endswith(".mp4"):
        vpath = vpath + ".mp4"
    
    print("Writing to:", vpath)

    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=fps, bitrate=1800)

    an = animation.FuncAnimation(fig, anim, frames=n_frames, interval=200)
    an.save(vpath, writer=writer, dpi=264)

def animate_var_scalebar(
    X,
    var_t,
    cell_radii=None,    
    n_frames=100,
    file_name=None,
    dir_name="plots",
    path=None,
    xlim=None,
    ylim=None,
    fps=20,
    vmin=None,
    vmax=None,
    #     ec="red",
    cmap="CET_L8",
    scalebar=True,
    scale_factor=1,
    sbar_kwargs=dict(),
    ifcc="red",
    ppatch_kwargs=dict(edgecolor="gray"),
    lcoll_kwargs=dict(),
    title_fun=None,
    plot_ifc=False,
    sender_idx=np.array([], dtype=int),
    sender_clr=("bmw", 150),
    extend=None,
    **kwargs
):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    skip = int((var_t.shape[0]) / n_frames)
    
    if title_fun is not None:
        tf=title_fun
    else:
        tf=lambda x: None
    
    if scalebar and (not sbar_kwargs):
        sbar_kwargs = dict(
            units="um", 
            color="w", 
            box_color="k", 
            box_alpha=0.3, 
            font_properties=dict(weight=1000), 
            width_fraction=0.03,
            location="lower right",
        )
    
    if cell_radii is None:
        cell_radii = np.ones(X.shape[0]) * 5
    
    
    def anim(i):
        if cell_radii.ndim > 1:
            cr = cell_radii[skip * i]
        else:
            cr = cell_radii.copy()

        plot_var(
            ax,
            skip * i,
            X,
            var_t[skip * i],
            cell_radii=cr,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ifcc=ifcc,
            ppatch_kwargs=ppatch_kwargs,
            lcoll_kwargs=lcoll_kwargs,
            title=tf(skip * i),
            plot_ifc=plot_ifc,
            sender_idx=sender_idx,
            sender_clr=sender_clr,
            extend=extend,
            **kwargs,
        )
        
        if scalebar:
            scalebar_ = ScaleBar(
                scale_factor,   # unit distance in real units
                **sbar_kwargs
            )
            ax.add_artist(scalebar_)

    if path is None:

        # Construct default path if `path` not supplied
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)  
        if file_name is None:
            file_name = "animation_%d" % time.time()
        vpath = os.path.join(dir_name, file_name)
    
    else:
        vpath = path
    
    if not vpath.endswith(".mp4"):
        vpath = vpath + ".mp4"
    
    print("Writing to:", vpath)

    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=fps, bitrate=1800)

    an = animation.FuncAnimation(fig, anim, frames=n_frames, interval=200)
    an.save(vpath, writer=writer, dpi=264)


def animate_var_lattice_scalebar(
    X_arr,
    var_t,
    cell_radii=None,    
    n_frames=100,
    file_name=None,
    dir_name="plots",
    path=None,
    xlim=None,
    ylim=None,
    fps=20,
    vmin=None,
    vmax=None,
    #     ec="red",
    cmap="CET_L8",
    colorbar=False,
    cbar_aspect=20,
    cbar_kwargs=dict(),
    extend=None,
    scalebar=True,
    scale_factor=1,
    sbar_kwargs=dict(),
    ifcc="red",
    ppatch_kwargs=dict(edgecolor="gray"),
    lcoll_kwargs=dict(),
    title_fun=None,
    plot_ifc=False,
    sender_idx=np.array([], dtype=int),
    sender_clr=("bmw", 150),
    **kwargs
):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    skip = int((var_t.shape[0]) / n_frames)
    
    if title_fun is not None:
        tf=title_fun
    else:
        tf=lambda x: None
    
    if scalebar and (not sbar_kwargs):
        sbar_kwargs = dict(
            units="um", 
            color="w", 
            box_color="k", 
            box_alpha=0.3, 
            font_properties=dict(weight=1000), 
            width_fraction=0.03,
            location="lower right",
        )
    
    X_ndim = X_arr.ndim

    if X_ndim > 2:
        n = X_arr.shape[1]
    else:
        n = X_arr.shape[0]
    
    if cell_radii is None:
        cell_radii = np.ones(n) * 5
    
    def anim(i, init_cbar=False):
        if cell_radii.ndim > 1:
            cr = cell_radii[skip * i]
        else:
            cr = cell_radii.copy()
        
        if X_ndim > 2:
            Xi = X_arr[skip * i]
        else:
            Xi = X_arr.copy()
        
        plot_var(
            ax,
            skip * i,
            Xi,
            var_t[skip * i],
            cell_radii=cr,
            xlim=xlim,
            ylim=ylim,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            colorbar=init_cbar,
            cbar_aspect=cbar_aspect,
            cbar_kwargs=cbar_kwargs,
            extend=extend,
            ifcc=ifcc,
            ppatch_kwargs=ppatch_kwargs,
            lcoll_kwargs=lcoll_kwargs,
            title=tf(skip * i),
            plot_ifc=plot_ifc,
            sender_idx=sender_idx,
            sender_clr=sender_clr,
            **kwargs,
        )
        
        if scalebar:
            scalebar_ = ScaleBar(
                scale_factor,   # unit distance in real units
                **sbar_kwargs
            )
            ax.add_artist(scalebar_)
        
    # Initialize colorbar
    if colorbar:
        anim(0, True);
    
    # Construct default file path if `path` not supplied
    if path is None:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)  
        if file_name is None:
            file_name = "animation_%d" % time.time()
        vpath = os.path.join(dir_name, file_name)
    else:
        vpath = path
    
    # Append file extension
    if not vpath.endswith(".mp4"):
        vpath = vpath + ".mp4"
    
    print("Writing to:", vpath)

    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=fps, bitrate=1800)

    an = animation.FuncAnimation(fig, anim, frames=n_frames, interval=200)
    an.save(vpath, writer=writer, dpi=264)
    
    
def inspect_out(*args, **kwargs):
    """Deprecated: Please use inspect_hex()"""
    
    warnings.warn("inspect_out() is deprecated; use inspect_hex().", warnings.DeprecationWarning)
    
    return inspect_hex(*args, **kwargs)
    
    
def inspect_hex(
    X,
    var_t,
    ax=None,
    idx=-1,
    cell_radii=None,
    vmin=None,
    vmax=None,
    cmap="CET_L8",
    #     ec="red",
    ifcc="red",
    ppatch_kwargs=dict(edgecolor="gray"),
    lcoll_kwargs=dict(),
    plot_ifc=False,
    sender_idx=np.array([], dtype=int),
    sender_clr=("bmw", 150),
    colorbar=False,
    cbar_aspect=20,
    cbar_kwargs=dict(),
    extend=None,
    **kwargs
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    
    nt = var_t.shape[0]
    k = np.arange(nt)[idx]
    
    if X.ndim == 2:
        Xk = X.copy()
    else:
        Xk = X[k]
    
    if cell_radii is None:
#         Xk_centered = Xk - Xk.mean(axis=0)
#         max_rad = np.linalg.norm(Xk_centered, axis=1).max() * 2
#         crk = max_rad * np.ones(Xk.shape[0], dtype=np.float32) 
        crk = np.ones(X.shape[0]) * 5
        
    elif cell_radii.ndim == 2:
        crk = cell_radii[k]
        
    else:
        crk = cell_radii.copy()
    
    plot_var(
        ax,
        k,
        Xk,
        var_t[k],
        cell_radii=crk,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        ifcc=ifcc,
        ppatch_kwargs=ppatch_kwargs,
        lcoll_kwargs=lcoll_kwargs,
        plot_ifc=plot_ifc,
        sender_idx=sender_idx,
        sender_clr=sender_clr,
        colorbar=colorbar,
        cbar_aspect=cbar_aspect,
        cbar_kwargs=cbar_kwargs,
        extend=extend,
       **kwargs
    )
    

def inspect_grid_hex(
    t,
    X_t,
    var_t,
    plt_idx=None,
    axs=None,
    nrows=None,
    ncols=None,
    cell_radii=None,
    vmin=None,
    vmax=None,
    xlim=None,
    ylim=None,
    title_fun=None,
    cmap="kgy",
    axis_off=True,
    figsize=(10,6),
    sender_idx=np.array([], dtype=int),
    sender_clr=("bmw", 150),
    colorbar=False,
    cbar_aspect=20,
    cbar_kwargs=dict(),
    **kwargs
):
    nt = t.size
    
    if title_fun is None:
        title_fun = lambda i: f"Time = {t[i]:.2f}"
    
    kw = kwargs.copy()
    if xlim is not None:
        kw["xlim"] = xlim
    if ylim is not None:
        kw["ylim"] = ylim
    
    # Render frames
    if plt_idx is None:
        nplot = nrows * ncols
        plt_idx = np.array([int(i) for i in np.linspace(0, nt-1, nplot)])
    
    if axs is None:
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        
    for ax, i in zip(axs.flat, plt_idx):
        
        title = title_fun(i)
        
        inspect_hex(
            X=X_t,
            var_t=var_t, 
            ax=ax,
            idx=i,
            cell_radii=cell_radii,
            vmin=vmin, 
            vmax=vmax,
            cmap=cmap,
            title=title,
            axis_off=axis_off,
            sender_idx=sender_idx,
            sender_clr=sender_clr,
            colorbar=colorbar,
            cbar_aspect=cbar_aspect,
            cbar_kwargs=cbar_kwargs,
            **kw
        )

    
    return fig, axs


def animate_var_lattice(
    X_arr,
    var_t,
    cell_radii=None,
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
    plot_ifc=False,
    sender_idx=np.array([], dtype=int),
    sender_clr=("bmw", 150),
    extend=None,
    **kwargs
):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    skip = int((var_t.shape[0]) / n_frames)
    
    if title_fun is not None:
        tf=title_fun
    else:
        tf=lambda x: None

    X_ndim = X_arr.ndim

    if X_ndim > 2:
        n = X_arr.shape[1]
    else:
        n = X_arr.shape[0]
    
    if cell_radii is None:
        cell_radii = np.ones(n) * 5
        
    def anim(i):
        if cell_radii.ndim > 1:
            cr = cell_radii[skip * i]
        else:
            cr = cell_radii.copy()

        if X_arr.ndim > 2:
            X = X_arr[skip * i]
        else:
            X = X_arr.copy()
    
        plot_var(
            ax,
            skip * i,
            X,
            var_t[skip * i],
            cell_radii=cr,
            xlim=xlim,
            ylim=ylim,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ifcc=ifcc,
            ppatch_kwargs=ppatch_kwargs,
            lcoll_kwargs=lcoll_kwargs,
            title=tf(skip * i),
            plot_ifc=plot_ifc,
            sender_idx=sender_idx,
            sender_clr=sender_clr,
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
    nt = var_t.shape[0]
    
    if X_arr.ndim == 2:
        X_t = np.repeat(X_arr[np.newaxis, :], nt, axis=0)
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
    skip = int(nt / n_frames)
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
    sender_idx=np.array([], dtype=int),
    sender_clr=("bmw", 150),
    sender_kwargs=dict(),
    colorbar=False,
    cbar_aspect=20,
    cbar_kwargs=dict(),
    extend=None,
    **kwargs
):
    ax.clear()
    if axis_off:
        ax.axis("off")
    
    if type(cmap) is str:
        cmap_ = cc.cm[cmap]
    else:
        cmap_ = cmap
    
    if vmin is None:
        vmin = var.min()
    if vmax is None:
        vmax = var.max()
    
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
        cmap=cmap_,
        vmin=vmin,
        vmax=vmax,
        **pcolormesh_kwargs
    )
    
    sender_col = cc.cm[sender_clr[0]](sender_clr[1] / 256)
    ax.plot(*X[sender_idx].T, color=sender_col, **sender_kwargs)
    
    
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
    )
    
    if title is not None:
        ax.set_title(title)

    if extend is None:
        
        # Extend colorbar if necessary
        n = var.shape[0]        
        ns_mask = ~ np.isin(np.arange(n), sender_idx)
        is_under_min = var.min(initial=0.0, where=ns_mask) < vmin
        is_over_max  = var.max(initial=0.0, where=ns_mask) > vmax
        extend = ("neither", "min", "max", "both")[is_under_min + 2 * is_over_max]
    
    if colorbar:
        
        # Construct colorbar
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin, vmax), 
                cmap=cmap_), 
            ax=ax,
            aspect=cbar_aspect,
            extend=extend,
            **cbar_kwargs
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
    nt = var_t.shape[0]
    
    if X_arr.ndim == 2:
        X_t = np.repeat(X_arr[np.newaxis, :], nt, axis=0)
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
    skip = int(nt / n_frames)
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
    var_t,
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
    sender_idx=np.array([], dtype=int),
    sender_clr=("bmw", 150),
    sender_kwargs=dict(),
    colorbar=False,
    cbar_aspect=20,
    cbar_kwargs=dict(),
    extend=None,
    **kwargs
):
    """
    """
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        
    nt = var_t.shape[0]
    k = np.arange(nt)[idx]
    
    if X.ndim == 2:
        Xk = X.copy()
    else:
        Xk = X[k]
    
    plot_interp_mesh(
        ax,
        Xk,
        var_t[k],
        n_interp=n_interp,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        title=title,
        axis_off=axis_off,
        xlim=xlim,
        ylim=ylim,
        aspect=aspect,
        pcolormesh_kwargs=pcolormesh_kwargs,
        sender_idx=sender_idx,
        sender_clr=sender_clr,
        sender_kwargs=sender_kwargs,
        colorbar=colorbar,
        cbar_aspect=cbar_aspect,
        cbar_kwargs=cbar_kwargs,
        extend=extend,
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


####################################################################
############ DEPRECATED ############################################
####################################################################


####### Functions for calculating cell-cell contacts 

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
    rho_0_space,
    rho_max,
    g,
    rhs,
    dde_args,
    delay,
    thresh,
    where_vars,
    min_delay=5,
    progress_bar=False,
    varargs_type="list",
):

    # Get # cells and # time-points
    n = X.shape[0]
    nt = t.size
    n_rho = rho_0_space.size

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
            var_vals=[rho_t],
            dde_args=dde_args,
            E0=S0,
            delay=delay,
            where_vars=where_vars,
            min_delay=min_delay,
            varargs_type=varargs_type,
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
        rho_max=rho_max,
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
    

############# Lattice functions (X, A, meta_df)

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


###### Geometry utilities

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

