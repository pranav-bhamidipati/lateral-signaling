####### Load depenendencies
from functools import partial
import os
from typing import OrderedDict
import warnings
from copy import deepcopy

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
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

hv.extension("matplotlib")

####### Steady-state expression is computed from simulations
from steady_state import ss_sacred_dir

_steady_state_error = f"Simulations for steady-state approximation not found in directory: {ss_sacred_dir.resolve().absolute()}"

try:
    assert ss_sacred_dir.exists(), f"Directory does not exist: {ss_sacred_dir}"
    import steady_state as ss

    ss_args = ss._initialize()
    get_steady_state = partial(ss._get_steady_state, *ss_args)
    get_steady_state_vector = np.vectorize(get_steady_state)
except Exception as e:
    if isinstance(e, IndexError) or isinstance(e, AssertionError):

        def get_steady_state(*args, **kwargs):
            raise FileNotFoundError(_steady_state_error)

        def get_steady_state_vector(*args, **kwargs):
            raise FileNotFoundError(_steady_state_error)

    else:
        raise e

####### Utilities for parallel computing

_dask_client_default_kwargs = dict(
    threads_per_worker=1,
    memory_limit="auto",
    interface="lo",
    timeout=600,
)

####### Constants

# Density at rho = 1
ref_density_mm = 1250.0  # cells / mm^2
ref_density_um = ref_density_mm / 1e6  # cells / um^2

# Cell diameter at rho = 1
#  (Approximates each cell as a hexagon)
ref_cell_diam_mm = np.sqrt(2 / (np.sqrt(3) * ref_density_mm))  # mm
ref_cell_diam_um = ref_cell_diam_mm * 1e3  # um


####### Differential equation right-hand-side functions


def receiver_rhs(
    R,
    R_delay,
    Adj,
    sender_idx,
    beta_func,
    beta_args,
    alpha,
    k,
    p,
    lambda_,
    g,
    rho,
    S_delay,
    gamma_R,
):

    # Get signaling as a function of density
    beta = beta_func(rho, *beta_args)

    # Get input signal across each interface
    S_bar = beta * (Adj @ S_delay)

    # Calculate dR/dt
    dR_dt = alpha * (S_bar ** p) / (k ** p + S_bar ** p) - R
    dR_dt[sender_idx] = 0

    return dR_dt


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


def reporter_rhs(
    R,
    R_delay,
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
    S_delay,
    gamma_R,
):

    # Get signaling as a function of density
    beta = beta_func(rho, *beta_args)

    # Get input signal across each interface
    S_bar = beta * (Adj @ S_delay)

    # Calculate dR/dt
    dR_dt = (
        alpha * (S_bar ** p) / (k ** p + (delta * S_delay) ** p + S_bar ** p) - R
    ) * gamma_R

    dR_dt[sender_idx] = 0

    return dR_dt


####### General utils

# Vectorized integer ceiling
ceiling = np.vectorize(ceil)

# Vectorized rounding
vround = np.vectorize(round)


@numba.njit
def normalize(x, xmin, xmax):
    """Normalize `x` given explicit min/max values."""
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

# Colors for specific uses
_sender_clr = "#e330ff"
_sender_clr2 = "#C43EA8BF"
_gfp_green = "#0B7E18"
_receiver_red = "#C25656E6"

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
col_light_gray = "#eeeeee"
gray = "#aeaeae"
dark_gray = "#5a5a5a"
darker_gray = "#303030"
black = "#060605"

# Custom version of the "KGY" colormap
kgy_original = cc.cm["kgy"]
kgy = ListedColormap(kgy_original(np.linspace(0, 0.92, 256)))

# Custom categorical color list(s)
growthrate_colors = [purple, greens[3], yob[1]]


def rgb_as_int(rgb):
    """Coerce RGB iterable to a tuple of integers"""
    if any([v >= 1.0 for v in rgb]):
        _rgb = tuple(rgb)
    else:
        _rgb = tuple((round(255 * c) for c in rgb))

    return _rgb


def rgb_as_float(rgb):
    """Coerce RGB iterable to an ndarray of floats"""
    if any([v >= 1.0 for v in rgb]) or any([type(v) is int for v in rgb]):
        _rgb = (np.asarray(rgb) / 255).astype(float)
    else:
        _rgb = np.asarray(rgb).astype(float)

    return _rgb


def sample_cycle(cycle, size):
    """Sample a continuous colormap at regular intervals to get a linearly segmented map"""
    return hv.Cycle([cycle[i] for i in ceiling(np.linspace(0, len(cycle) - 1, size))])


def blend_hex(hex1, hex2, c=0.5):
    """Returns a blend of two colors."""
    blend_rgb = c * np.asfarray(hex2rgb(hex1)) + (1 - c) * np.asfarray(hex2rgb(hex2))
    return rgb2hex(tuple(round(i) for i in blend_rgb))


def hex2rgb(h):
    """Convert 6-digit hex code to RGB values (0, 255)"""
    h = h.lstrip("#")
    return tuple(int(h[(2 * i) : (2 * (i + 1))], base=16) for i in range(3))


def rgb2hex(rgb):
    """Converts rgb colors to hex"""

    RGB = np.zeros((3,), dtype=np.uint8)
    for i, _c in enumerate(rgb):

        # Convert vals in [0., 1.] to [0, 255]
        if _c <= 1.0:
            c = int(_c * 255)
        else:
            c = _c

        # Calculate new values
        RGB[i] = round(c)

    return "#{:02x}{:02x}{:02x}".format(*RGB)


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
        if _c <= 1.0:
            c = int(_c * 255)
        else:
            c = _c

        # Calculate new values
        rgb[i] = round(a * c + (1 - a) * background[i])

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
    np.array(
        [
            np.cos(np.arange(0, 2 * np.pi, np.pi / 3) + np.pi / 6),
            np.sin(np.arange(0, 2 * np.pi, np.pi / 3) + np.pi / 6),
        ]
    ).T
    / np.sqrt(3)
)

_hex_x, _hex_y = _hex_vertices.T


def hex_grid(rows, cols=0, r=1.0, sigma=0, **kwargs):
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
    y_coords = np.linspace(
        -np.sqrt(3) * r * (rows - 1) / 4, np.sqrt(3) * r * (rows - 1) / 4, rows
    )
    X = []
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            X.append(np.array([x + (j % 2) * r / 2, y]))
    X = np.array(X)

    # Apply Gaussian filter if specified
    if sigma != 0:
        X = np.array([np.random.normal(loc=x, scale=sigma * r) for x in X])

    return X


def hex_grid_square(n, **kwargs):
    """Returns XY coordinates of n points on a square regular 2D hexagonal grid with edge
    length r, passed through a Gaussian filter with std. dev. = sigma * r."""

    # Get side length for square grid
    rows = int(np.ceil(np.sqrt(n)))

    return hex_grid(rows, **kwargs)[:n]


def hex_grid_circle(radius, sigma=0.0):
    """Returns XY coordinates of all points on a regular 2D hexagonal grid of edge length `1` within
    distance radius of the origin, passed through a Gaussian filter with std. dev. = `sigma`."""

    # Get side length for a square grid
    rad = ceil(radius)
    num_rows = rad * 2 + 1

    # Populate square grid with points
    X = hex_grid(num_rows)
    # X = []
    # for i, x in enumerate(np.linspace(-r * (num_rows - 1) / 2, r * (num_rows - 1) / 2, num_rows)):
    #     for j, y in enumerate(
    #         np.linspace(-np.sqrt(3) * r * (num_rows - 1) / 4, np.sqrt(3) * r * (num_rows - 1) / 4, num_rows)
    #     ):
    #         X.append(np.array([x + ((rad - j) % 2) * r / 2, y]))

    # Pass each point through a Gaussian filter
    if sigma > 0:
        # X = np.array([np.random.normal(loc=x, scale=sigma) for x in X])
        cov = np.eye(X.shape[1]) * sigma
        X += np.array([np.random.multivariate_normal(x, cov) for x in X])

    # Select points within radius and return
    dists = [np.linalg.norm(x) for x in X]

    return np.array(
        [X[i] for i in np.argsort(dists) if np.linalg.norm(dists[i]) <= radius]
    )


def get_center_cells(X, n_center=1):
    """Returns indices of the n_cells cells closest to the origin given their coordinates as an array X."""
    return np.argpartition([np.linalg.norm(x) for x in X], n_center)[:n_center]


def get_weighted_Adj(
    X, r_int, dtype=np.float32, sparse=False, row_stoch=False, atol=1e-8, **kwargs
):
    """
    Construct adjacency matrix for a non-periodic set of
    points (cells). Adjacency is determined by calculating pairwise
    distance and applying a threshold `r_int` (radius of interaction).
    Within this radius, weights are calculated from pairwise distance
    as the value of the PDF of a Normal distribution with standard
    deviation `r_int / 2`.
    """

    n = X.shape[0]
    d = pdist(X)
    a = scipy.stats.norm.pdf(d, loc=0, scale=r_int / 2)
    a[d > (r_int + atol)] = 0
    A = squareform(a)

    if row_stoch:
        rowsum = np.sum(A, axis=1)[:, np.newaxis]
        A = np.divide(A, rowsum)
    else:
        A = (A > 0).astype(dtype)

    if sparse:
        A = csr_matrix(A)

    return A


def gaussian_irad_Adj(
    X, irad, dtype=np.float32, sparse=False, row_stoch=False, **kwargs
):
    """
    ===DEPRECATED===

    Construct adjacency matrix for a non-periodic set of
    points (cells). Adjacency is determined by calculating pairwise
    distance and applying a threshold `irad` (interaction radius)
    """

    # No longer should use this one
    warnings.warn(
        "gaussian_irad_Adj() is deprecated; use get_weighted_Adj().", DeprecationWarning
    )

    n = X.shape[0]
    d = pdist(X)
    a = scipy.stats.norm.pdf(d, loc=0, scale=irad / 2)
    a[d >= irad] = 0
    A = squareform(a)

    if row_stoch:
        rowsum = np.sum(A, axis=1)[:, np.newaxis]
        A = np.divide(A, rowsum)
    else:
        A = (A > 0).astype(dtype)

    if sparse:
        A = csr_matrix(A)

    return A


def irad_Adj(X, irad, dtype=np.float32, sparse=False, row_stoch=False, **kwargs):
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
    """ """

    if not cols:
        cols = rows

    # Construct adjacency matrix
    a = make_Adj_sparse(rows, cols, dtype=dtype, **kwargs)

    # Add self-edges
    n = rows * cols
    eye = identity(n).astype(dtype)
    A = a + eye

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
    Adj = np.zeros((n, n), dtype=dtype)
    for i in range(cols):
        for j in range(rows):

            # Get neighbors of cell at location i, j
            nb = np.array(
                [
                    (i, j + 1),
                    (i, j - 1),
                    (i - 1, j),
                    (i + 1, j),
                    (i - 1 + 2 * (j % 2), j - 1),
                    (i - 1 + 2 * (j % 2), j + 1),
                ]
            )

            nb[:, 0] = nb[:, 0] % cols
            nb[:, 1] = nb[:, 1] % rows

            # Populate Adj
            nbidx = np.array([ni * rows + nj for ni, nj in nb])
            Adj[i * rows + j, nbidx] = 1

    return Adj


def hex_Adj(rows, cols=0, dtype=np.float32, sparse=False, row_stoch=False, **kwargs):
    """ """
    # Make hexagonal grid
    X = hex_grid(rows, cols, **kwargs)

    # Construct adjacency matrix
    if sparse:
        A = make_Adj_sparse(rows, cols, dtype=dtype, **kwargs)

        # Make row-stochastic (rows sum to 1)
        if row_stoch:

            # Calculate inverse of rowsum
            inv_rowsum = diags(np.array(1 / A.sum(axis=1)).ravel())

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
                    (i, j + 1),
                    (i, j - 1),
                    (i - 1, j),
                    (i + 1, j),
                    (i - 1 + 2 * (j % 2), j - 1),
                    (i - 1 + 2 * (j % 2), j + 1),
                ]
            ).T

            nb_col = nb_col % cols
            nb_row = nb_row % rows

            nb = nb_col * rows + nb_row
            nb_j[6 * (i * rows + j) : 6 * (i * rows + j + 1)] = nb

    nb_i = np.repeat(np.arange(n).astype(int), 6)
    Adj_vals = np.ones(6 * n, dtype=np.float32)

    return csr_matrix((Adj_vals, (nb_i, nb_j)), shape=(n, n))


######### Spatial utils


######### Statistics


@numba.njit
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
        interval = (interval[0], vals.max())

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

    assert src.dtype is dst.dtype, "src and dst should have the same dtype"

    dtype = src.dtype
    src = src.ravel()
    dst = dst.ravel()

    # Get slope perpendicular to the line segment
    pslope = -(dst[0] - src[0]) / (dst[1] - src[1])

    # Get increments in x and y direction
    dx = width / (2 * np.sqrt(1 + pslope ** 2))
    dy = pslope * dx

    # Add/subtract increments from each point
    corners = np.empty((4, 2), dtype=dtype)
    corners[:, 0] = dx
    corners[:, 1] = dy
    corners[0] *= -1
    corners[3] *= -1
    corners[:2] += src
    corners[2:] += dst

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
    Suu = np.sum(u ** 2)
    Suv = np.sum(u * v)
    Svv = np.sum(v ** 2)

    Suuu = np.sum(u ** 3)
    Suuv = np.sum((u ** 2) * v)
    Suvv = np.sum(u * (v ** 2))
    Svvv = np.sum(v ** 3)

    # Package sums into form Aw = b
    A = np.array([[Suu, Suv], [Suv, Svv]])
    b = (
        1
        / 2
        * np.array(
            [
                [Suuu + Suvv],
                [Svvv + Suuv],
            ]
        )
    )

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
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

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
def t_to_units(dimless_time, ref_growth_rate=6.160221865205395e-01):
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


@numba.vectorize
def _nc2a_ufunc(ncells, rho, ref_density):
    return ncells / (rho * ref_density)


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
    return _nc2a_ufunc(ncells, rho, ref_density)


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
    assert (
        step_delay >= min_delay
    ), "Delay time is too short. Lower dt or lengthen delay."
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
    assert (
        step_delay >= min_delay
    ), "Delay time is too short. Lower dt or lengthen delay."
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


##### Visualization options

# Colorbar
_vmin = 0.0
_vmax = 0.3
cbar_kwargs = dict(ticks=[_vmin, _vmax], label="GFP (AU)")

# Scalebar
sbar_kwargs = dict(
    dx=ref_cell_diam_um,
    units="um",
    color="w",
    box_color="w",
    box_alpha=0,
    font_properties=dict(weight=1000, size=10),
    width_fraction=0.03,
    location="lower right",
)

# Container for plotting kwargs
plot_kwargs = dict(
    # sender_idx=sender_idx,
    # xlim=xlim,
    # ylim=ylim,
    vmin=_vmin,
    vmax=_vmax,
    cmap=kgy,
    colorbar=True,
    cbar_aspect=12,
    extend=None,
    cbar_kwargs=cbar_kwargs,
    scalebar=False,
    sbar_kwargs=sbar_kwargs,
)


##### Visualization util functions


def ecdf(d, *args, **kwargs):
    """Construct an ECDF from 1D data array `d`"""
    x = np.sort(d)
    y = np.linspace(1, 0, x.size, endpoint=False)[::-1]
    return hv.Scatter(np.array([x, y]).T, *args, **kwargs)


def default_rcParams(
    SMALL_SIZE=12,
    MEDIUM_SIZE=14,
    BIGGER_SIZE=16,
):
    """Set default parameters for Matplotlib"""

    # Set font sizes
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


#### The following code is based on the `bebi103` package


def predictive_regression(
    samples,
    samples_x,
    percentiles=[68, 90],
    key_dim="__x",
):
    """
    Compute a predictive regression plot from samples.

    Heavily inspired by the `beb103` package by Justin Bois. The
    main difference is using matplotlib instead of Bokeh.

    See `bebi103.viz.predictive_regression` for documentation.
    """

    if not isinstance(samples, np.ndarray):
        raise RuntimeError("Samples can only be Numpy arrays.")

    if not isinstance(samples_x, np.ndarray):
        raise RuntimeError("samples_x can only be a Numpy array.")

    if len(percentiles) > 4:
        raise RuntimeError("Can specify maximally four percentiles.")

    if samples.shape[1] != len(samples_x):
        raise ValueError(
            "`samples_x must have the same number of entries as `samples` does columns."
        )

    # Build ptiles
    percentiles = np.sort(percentiles)[::-1]
    ptiles = [pt for pt in percentiles if pt > 0]
    ptiles = (
        [50 - pt / 2 for pt in percentiles]
        + [50]
        + [50 + pt / 2 for pt in percentiles[::-1]]
    )
    ptiles_str = [str(pt) for pt in ptiles]

    df_pred = pd.DataFrame(
        data=np.percentile(samples, ptiles, axis=0).transpose(),
        columns=ptiles_str,
    )
    df_pred[key_dim] = samples_x
    df_pred = df_pred.sort_values(by=key_dim)

    return df_pred


def plot_predictive_regression(
    df_pred=None,
    data=None,
    key_dim="__x",
    ax=None,
    figsize=(8, 8),
    colors=cc.cm["gray"](np.linspace(0.2, 0.85, 7))[::-1],
    median_lw=2,
    ci_kwargs=dict(),
    median_kwargs=dict(),
    data_kwargs=dict(),
):
    """
    Compute a predictive regression plot from samples.

    Heavily inspired by the `beb103` package by Justin Bois. The
    main difference is using matplotlib instead of Bokeh.

    See `bebi103.viz.predictive_regression` for documentation.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    plt.sca(ax)

    # Confidence regions
    ptiles_str = df_pred.columns.tolist()
    ptiles_str.remove(key_dim)
    n = (len(ptiles_str) - 1) // 2
    for i in range(n):
        plt.fill_between(
            key_dim,
            ptiles_str[i],
            ptiles_str[2 * n - i],
            data=df_pred,
            color=colors[i],
            edgecolor=(0, 0, 0, 0),
            **ci_kwargs,
        )

    # Median as a line
    plt.plot(
        key_dim,
        ptiles_str[n],
        data=df_pred,
        linewidth=median_lw,
        color=colors[-1],
        **median_kwargs,
    )

    # It's useful to have data as a data frame
    if data is not None:
        if type(data) == tuple and len(data) == 2 and len(data[0]) == len(data[1]):
            data = np.vstack(data).transpose()
        df_data = pd.DataFrame(data=data, columns=["__data_x", "__data_y"])
        df_data = df_data.sort_values(by="__data_x")

        # Plot data points
        data_color = data_kwargs.pop("c", "k")
        data_size = data_kwargs.pop("s", 15)
        data_marker = data_kwargs.pop("marker", "o")
        data_alpha = data_kwargs.pop("alpha", 1.0)
        plt.scatter(
            "__data_x",
            "__data_y",
            data=df_data,
            c=data_color,
            s=data_size,
            marker=data_marker,
            alpha=data_alpha,
            **data_kwargs,
        )

    return plt.gca()


def remove_RB_spines(plot, element):
    """Hook to remove right and bottom spines from Holoviews plot"""
    plot.state.axes[0].spines["right"].set_visible(False)
    plot.state.axes[0].spines["bottom"].set_visible(False)


def remove_RT_spines(plot, element):
    """Hook to remove right and top spines from Holoviews plot"""
    plot.state.axes[0].spines["right"].set_visible(False)
    plot.state.axes[0].spines["top"].set_visible(False)


def xaxis_label_left(plot, element):
    """Hook to move the x-axis label to location 'left' in a Holoviews plot"""
    _cax = plot.state.axes[0]
    _cax.set_xlabel(_cax.get_xlabel(), loc="left")


def yaxis_label_bottom(plot, element):
    """Hook to move the y-axis label to location 'bottom' in a Holoviews plot"""
    _cax = plot.state.axes[0]
    _cax.set_ylabel(_cax.get_ylabel(), loc="bottom")


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


##### Plotting and animating


def plot_hex_sheet(
    ax,
    X,
    var,
    rho=1.0,
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
    sender_clr=_sender_clr,
    colorbar=False,
    cbar_aspect=20,
    extend=None,
    cbar_kwargs=dict(),
    poly_padding=0.0,
    scalebar=False,
    sbar_kwargs=dict(),
    **kwargs,
):

    # Clear axis (allows you to reuse axis when animating)
    # ax.clear()
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
    colors = np.asarray(_cmap(normalize(var, vmin, vmax)))

    # Replace sender(s) with appropriate color
    if isinstance(sender_clr, str):
        _sender_clr = hex2rgb(sender_clr)
    else:
        _sender_clr = list(sender_clr)

    if len(_sender_clr) == 3:
        _sender_clr = [*rgb_as_float(_sender_clr), 1.0]
    else:
        _sender_clr = [*rgb_as_float(_sender_clr[:3]), _sender_clr[3]]

    colors[sender_idx] = _sender_clr

    # Get polygon size. Optionally increase size
    #  so there's no visual gaps between cells
    _r = (1 + poly_padding) / np.sqrt(rho)

    # Plot cells as polygons
    for i, (x, y) in enumerate(X):

        ax.fill(_r * _hex_x + x, _r * _hex_y + y, fc=colors[i], ec=ec, **kwargs)

    # Set figure args, accounting for defaults
    if title is not None:
        ax.set_title(title)
    if not xlim:
        xlim = [X[:, 0].min(), X[:, 0].max()]
    if not ylim:
        ylim = [X[:, 1].min(), X[:, 1].max()]
    if aspect is None:
        aspect = 1
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
            is_over_max = var.max(initial=0.0, where=ns_mask) > vmax
            _extend = ("neither", "min", "max", "both")[is_under_min + 2 * is_over_max]
        else:
            _extend = extend

        # Construct colorbar
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap=_cmap),
            ax=ax,
            aspect=cbar_aspect,
            extend=_extend,
            **cbar_kwargs,
        )

    if scalebar:
        sb_kw = plot_kwargs["sbar_kwargs"].copy()
        sb_kw.update(sbar_kwargs)
        _scalebar = ScaleBar(**sb_kw)
        ax.add_artist(_scalebar)


def animate_hex_sheet(
    fpath,
    X_t,
    var_t,
    rho_t=1.0,
    fig=None,
    ax=None,
    anim=None,
    n_frames=100,
    fps=20,
    dpi=300,
    title_fun=None,
    writer="ffmpeg",
    fig_kwargs=dict(),
    plot_kwargs=dict(),
    _X_func=None,
    _var_func=None,
    _rho_func=None,
    **kwargs,
):

    nt = var_t.shape[0]
    n = X_t.shape[-2]

    if _X_func is None:

        if X_t.ndim == 2:
            _X_func = lambda i: X_t
        elif X_t.ndim == 3:
            _X_func = lambda i: X_t[i]

    if _var_func is None:

        if var_t.ndim == 1:
            _var_func = lambda i: var_t
        elif var_t.ndim == 2:
            _var_func = lambda i: var_t[i]

    if _rho_func is None:

        _rho_t = np.asarray(rho_t)
        if rho_t.ndim == 0:
            _rho_func = lambda i: rho_t
        elif rho_t.ndim == 1:
            _rho_func = lambda i: rho_t[i]

    if title_fun is None:
        tf = lambda i: None

    # Generate figure and axes if necessary
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(**fig_kwargs)

    # If colorbar is specified, plot only once
    if "colorbar" in plot_kwargs:
        if plot_kwargs["colorbar"]:

            # Unpack args for colorbar
            _var = var_t[0]
            _sender_idx = plot_kwargs["sender_idx"]
            _vmin = plot_kwargs["vmin"]
            _vmax = plot_kwargs["vmax"]
            _cmap = plot_kwargs["cmap"]
            _cbar_aspect = plot_kwargs["cbar_aspect"]
            _cbar_kwargs = plot_kwargs["cbar_kwargs"]

            # Calculate colorbar extension if necessary
            if "extend" in plot_kwargs:
                _extend = plot_kwargs["extend"]
            else:
                _extend = None

            if _extend is None:
                n = _var.shape[0]
                ns_mask = ~np.isin(np.arange(n), _sender_idx)
                is_under_min = _var.min(initial=0.0, where=ns_mask) < _vmin
                is_over_max = _var.max(initial=0.0, where=ns_mask) > _vmax
                _extend = ("neither", "min", "max", "both")[
                    is_under_min + 2 * is_over_max
                ]

            # Construct colorbar
            cbar = plt.colorbar(
                plt.cm.ScalarMappable(
                    norm=mpl.colors.Normalize(_vmin, _vmax), cmap=_cmap
                ),
                ax=ax,
                aspect=_cbar_aspect,
                extend=_extend,
                **_cbar_kwargs,
            )

    # Turn off further colorbar plotting during animation
    _plot_kwargs = deepcopy(plot_kwargs)
    _plot_kwargs["colorbar"] = False

    frames = vround(np.linspace(0, nt - 1, n_frames))

    # Animate using plot_hex_sheet() if no animation func supplied
    if anim is None:

        def anim(**kw):
            ax.clear()
            plot_hex_sheet(ax=ax, **kw)

    # Make wrapper function that changes arguments with each frame
    def _anim(i):

        # Get changing arguments
        var_kw = dict(
            X=_X_func(frames[i]),
            var=_var_func(frames[i]),
            rho=_rho_func(frames[i]),
            title=title_fun(frames[i]),
        )

        # Plot frame of animation
        anim(**var_kw, **_plot_kwargs)

    try:
        _writer = animation.writers[writer](fps=fps, bitrate=1800)
    except RuntimeError:
        print(
            """
        The `ffmpeg` writer must be installed inside the runtime environment.
        Writer availability can be checked in the current enviornment by executing 
        `matplotlib.animation.writers.list()` in Python. Install location can be
        checked by running `which ffmpeg` on a command line/terminal.
        """
        )

    _anim_FA = animation.FuncAnimation(fig, _anim, frames=n_frames, interval=200)

    # Get path and print to output
    _fpath = str(fpath)
    if not _fpath.endswith(".mp4"):
        _fpath += ".mp4"
    print("Writing to:", _fpath)

    # Save animation
    _anim_FA.save(
        _fpath,
        writer=_writer,
        dpi=dpi,
        progress_callback=lambda i, n: print(f"Frame {i+1} / {n}"),
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
    ppatch_kwargs=dict(edgecolor="gray"),
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
    **kwargs,
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
        xlim = [X[:, 0].min(), X[:, 0].max()]
    if not ylim:
        ylim = [X[:, 1].min(), X[:, 1].max()]
    if aspect is None:
        aspect = 1
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
        ns_mask = ~np.isin(np.arange(n), sender_idx)
        is_under_min = var.min(initial=0.0, where=ns_mask) < vmin
        is_over_max = var.max(initial=0.0, where=ns_mask) > vmax
        extend = ("neither", "min", "max", "both")[is_under_min + 2 * is_over_max]

    if colorbar:

        # Construct colorbar
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap=cmap_),
            ax=ax,
            aspect=cbar_aspect,
            extend=extend,
            **cbar_kwargs,
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
    **kwargs,
):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    skip = int((var_t.shape[0]) / n_frames)

    if title_fun is not None:
        tf = title_fun
    else:
        tf = lambda x: None

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
        anim(0, True)

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
    **kwargs,
):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    skip = int((var_t.shape[0]) / n_frames)

    if title_fun is not None:
        tf = title_fun
    else:
        tf = lambda x: None

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
                scale_factor, **sbar_kwargs  # unit distance in real units
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
    **kwargs,
):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    skip = int((var_t.shape[0]) / n_frames)

    if title_fun is not None:
        tf = title_fun
    else:
        tf = lambda x: None

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
                scale_factor, **sbar_kwargs  # unit distance in real units
            )
            ax.add_artist(scalebar_)

    # Initialize colorbar
    if colorbar:
        anim(0, True)

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

    warnings.warn("inspect_out() is deprecated; use inspect_hex().", DeprecationWarning)

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
    **kwargs,
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
        **kwargs,
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
    figsize=(10, 6),
    sender_idx=np.array([], dtype=int),
    sender_clr=("bmw", 150),
    colorbar=False,
    cbar_aspect=20,
    cbar_kwargs=dict(),
    **kwargs,
):
    nt = t.size

    if title_fun is None:
        title_fun = lambda i: f"Time = {t[i]:.2f}"

    kw = deepcopy(kwargs)
    if xlim is not None:
        kw["xlim"] = xlim
    if ylim is not None:
        kw["ylim"] = ylim

    # Render frames
    if plt_idx is None:
        nplot = nrows * ncols
        plt_idx = np.array([int(i) for i in np.linspace(0, nt - 1, nplot)])

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
            **kw,
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
    **kwargs,
):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    skip = int((var_t.shape[0]) / n_frames)

    if title_fun is not None:
        tf = title_fun
    else:
        tf = lambda x: None

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
    **kwargs,
):
    ax.clear()
    if axis_off:
        ax.axis("off")

    #     cols = cc.cm[cmap](normalize(var, vmin, vmax))

    xx = X[:, 0].reshape(cols, rows)
    yy = X[:, 1].reshape(cols, rows)
    #     cols, rows = xx.shape

    xi, yi = np.meshgrid(np.arange(cols), np.arange(rows), indexing="ij")
    zz = var[rows * xi + yi]

    ax.pcolormesh(
        xx,
        yy,
        zz,
        shading="gouraud",
        cmap=cc.cm[cmap],
        vmin=vmin,
        vmax=vmax,
        **pcolormesh_kwargs,
    )

    if not xlim:
        xlim = [X[:, 0].min(), X[:, 0].max()]
    if not ylim:
        ylim = [X[:, 1].min(), X[:, 1].max()]
    if aspect is None:
        aspect = 1

    ax.set(xlim=xlim, ylim=ylim, aspect=aspect, **kwargs)

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
    **kwargs,
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
        tf = title_fun
    else:
        tf = lambda x: None

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
    **kwargs,
):
    """ """

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
        **kwargs,
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
    **kwargs,
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
    zzi = rbfi(xxi, yyi)  # interpolated values

    ax.pcolormesh(
        xxi,
        yyi,
        zzi,
        shading="auto",
        cmap=cmap_,
        vmin=vmin,
        vmax=vmax,
        **pcolormesh_kwargs,
    )

    sender_col = cc.cm[sender_clr[0]](sender_clr[1] / 256)
    ax.plot(*X[sender_idx].T, color=sender_col, **sender_kwargs)

    if not xlim:
        xlim = [X[:, 0].min(), X[:, 0].max()]
    if not ylim:
        ylim = [X[:, 1].min(), X[:, 1].max()]
    if aspect is None:
        aspect = 1
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
        ns_mask = ~np.isin(np.arange(n), sender_idx)
        is_under_min = var.min(initial=0.0, where=ns_mask) < vmin
        is_over_max = var.max(initial=0.0, where=ns_mask) > vmax
        extend = ("neither", "min", "max", "both")[is_under_min + 2 * is_over_max]

    if colorbar:

        # Construct colorbar
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap=cmap_),
            ax=ax,
            aspect=cbar_aspect,
            extend=extend,
            **cbar_kwargs,
        )

    #     ax.imshow(
    #         zzi,
    #         cmap=cc.cm[cmap],
    #         interpolation='nearest',
    #         vmin=vmin,
    #         vmax=vmax,
    #         **imshow_kwargs
    #     )

    ax.set(xlim=xlim, ylim=ylim, aspect=aspect, **kwargs)

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
    **kwargs,
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
        tf = title_fun
    else:
        tf = lambda x: None

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
    **kwargs,
):
    """ """

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
        **kwargs,
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
    figsize=(10, 6),
    **kwargs,
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
    idx = [int(i) for i in np.linspace(0, nt - 1, nplot)]
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
            **kwargs,
        )


####################################################################
############ DEPRECATED ############################################
####################################################################


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
    return dist / (np.sqrt(4 - beta ** 2))


@numba.njit
def rad_to_beta(rad, dist=1):
    return np.sqrt(4 - (dist ** 2 / rad ** 2))


@numba.njit
def beta_rho_exp(rho, m, *args):
    return np.exp(-m * np.maximum(rho - 1, 0))


@numba.njit
def beta_rho_with_low_density(rho, m, q):
    return np.where(rho < 1, rho ** q, np.exp(-m * (rho - 1)))


_beta_function_dictionary = OrderedDict(
    [
        ["exponential", beta_rho_exp],
        ["exponential_low_density", beta_rho_with_low_density],
    ]
)


# def set_beta_func(func_name, func):
#     _beta_function_dictionary[str(func_name)] = func


def get_beta_func(func_name):
    return _beta_function_dictionary[str(func_name)]


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

        r_t = 1 / np.sqrt(rho_t)
        X_t = np.array([X] * nt) * r_t[:, np.newaxis, np.newaxis]
        X_rho0_t[i] = X_t

        S_rho0_t[i] = S_t

        n_act = (S_t > thresh).sum(axis=1)
        A_rho0_t[i] = ncells_to_area(n_act, rho_t)

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

    plot = hv.Scatter(data=data, kdims=["rho_0"], vdims=["max_area"],).opts(
        color=cc.glasbey_category10[2],
        xlabel=r"plating density ($mm^2$)",
        xlim=(1000, 5250),
        xticks=[1250, 2500, 3750, 5000],
        ylabel=r"max. activated area ($\mu m^2$)",
        #     ylim=(   0, 200000),
        ylim=(0, None),
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
    **kwargs,
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
    alpha_space = np.logspace(*log_alpha_minmax, n_alpha)
    rep_space = np.arange(n_reps)
    free_params = (rep_space, lambda_space, alpha_space)

    # Get all pairwise combinations of free parameters
    param_space = np.meshgrid(*free_params)
    param_space = np.array(param_space, dtype=np.float32).T.reshape(
        -1, len(free_params)
    )

    # Get indices for pairwise combinations
    param_idx = np.meshgrid(*[np.arange(p.size) for p in free_params])
    param_idx = np.array(param_idx, dtype=int).T.reshape(-1, len(free_params))

    # Set seed
    np.random.seed(seed)

    # Minimally perturb initial fluorescence
    rv_shape = n_lambda, n_reps, n
    rv_mean = np.sqrt(np.pi / 2) * lambda_space[:, np.newaxis, np.newaxis]
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
        args[where_alpha] = alpha_

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
        data=data, kdims=["lambda", "alpha"], vdims=["% cells activated"]
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

    p2 = (
        hv.QuadMesh(data=data, kdims=["lambda", "alpha"], vdims=["mean_fluorescence"])
        .opts(
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
        )
        .redim.range(mean_fluorescence=(0, 1))
    )

    t1 = hv.Text(2e-5, 6e-1, "Inducible", fontsize=14).opts(color="white")
    t2 = hv.Text(5e-3, 5e0, "Self-\nactivating", fontsize=14).opts(color="white")
    t3 = hv.Text(5e-3, 5e0, "Self-\nactivating", fontsize=14).opts(color="black")

    return p1, p1 * t1 * t2, p2, p2 * t1 * t3


def run_basal_activity(
    *args,
    **kwargs,
):
    results = basal_activity_phase(*args, **kwargs)
    plots = basal_activity_plots(results)

    return results, plots


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
    **kwargs,
):
    """Returns an mp4 video of gene expression of cells on a regular lattice."""

    if levels is None:
        levels = [x for x in np.linspace(df[val].min(), df[val].max(), len(colors) + 1)]

    if points_opts is None:
        padding = 1 / (2 * R)
        points_opts = dict(
            padding=padding,
            aspect="equal",
            s=13300 * 1 / (R ** 2) / (1 + 2 * padding) ** 2,
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
        points_opts["title"] = title.format(*title_format(**locals()))

        plt = hv.Points(data=step_data, kdims=["X_coord", "Y_coord"], vdims=[val]).opts(
            **points_opts
        )

        return plt

    steps = df["step"].max()
    hmap = hv.HoloMap([(step, plot_lattice(step, dt)) for step in range(steps + 1)])

    return hmap


###### Geometry utilities


@numba.njit
def shoelace_area(points):
    """Returns the area enclosed by a convex polygon. Uses the shoelace method to
    calculate area given an ordered Numpy array of 2D Cartesian coordinates.
    """
    area = np.dot(points[:, 0], np.roll(points[:, 1], shift=1)) - np.dot(
        np.roll(points[:, 0], shift=1), points[:, 1]
    )
    return np.abs(area) / 2


# @numba.njit
def perimeter(points):
    """Returns the perimeter of a polygon with vertices given by `points`."""
    return np.linalg.norm(points - np.roll(points, shift=1, axis=0), axis=1).sum()


def circularity(points):
    """Returns the circularity of a polygon"""
    return 4 * np.pi * shoelace_area(points) / (perimeter(points) ** 2)


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
