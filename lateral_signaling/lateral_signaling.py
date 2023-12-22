"""Contact-dependent signaling between cells on a hexagonal lattice.
"""
__version__ = "0.0.1"
__author__ = "Pranav Bhamidipati"
__email__ = "pbhamidi@caltech.edu"
__license__ = "MIT"


import os
from typing import OrderedDict, TypeVar
import warnings
from pathlib import Path
from math import ceil

import tqdm
import numpy as np
import numba
import scipy.stats
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform

######################################################################
##########  SET UP DIRECTORIES AT IMPORT-TIME ########################
######################################################################

PathLike = TypeVar("PathLike", str, bytes, Path, os.PathLike, None)

### These paths are set during the `conda` environment creation.
### You can change them manually by re-defining the environment
### variable or edit `environment.yml` and rebuild the env.
__dir = Path(__file__).parent
data_dir = __dir.joinpath(os.getenv("LSIG_DATA_DIR"), "../data").resolve().absolute()
# analysis_dir = __dir.joinpath(os.getenv("LSIG_ANALYSIS_DIR", ../data/analysis)).resolve().absolute()
simulation_dir = (
    __dir.joinpath(os.getenv("LSIG_SIMULATION_DIR", "../data/simulations"))
    .resolve()
    .absolute()
)
plot_dir = (
    __dir.joinpath(os.getenv("LSIG_PLOTTING_DIR", "../figures")).resolve().absolute()
)
temp_plot_dir = (
    __dir.joinpath(os.getenv("LSIG_TEMPPLOTTING_DIR", "../figures/tmp"))
    .resolve()
    .absolute()
)

analysis_dir = data_dir.joinpath("analysis")

if not data_dir.exists():
    warnings.warn(
        f"Invalid path to directory `data_dir` containing supplementary data: "
        f"'{data_dir.resolve().absolute()}'. "
        f"Analysis and plotting routines will not be available."
    )

if not plot_dir.exists():
    warnings.warn(
        f"Invalid path to directory `plot_dir` for plotting outputs: "
        f"'{plot_dir.resolve().absolute()}'. "
    )

if not temp_plot_dir.exists():
    warnings.warn(
        f"Invalid path to directory `temp_plot_dir` for plotting outputs: "
        f"'{temp_plot_dir.resolve().absolute()}'. "
    )

if not simulation_dir.exists():
    warnings.warn(
        f"Invalid path to directory `simulation_dir` containing simulation"
        f"configurations and outputs: "
        f"'{simulation_dir.resolve().absolute()}'"
    )

if not analysis_dir.exists():
    warnings.warn(
        f"Invalid path to directory `analysis_dir` for data analysis outputs:"
        f"'{analysis_dir}'"
    )


####################################################################
##########  SET DATASETS INTERACTIVELY #############################
####################################################################


import _simulation_parameters as sp
import _growth_parameters as gp
import _steady_state as ss

### Parameters used for simulation of the system
simulation_params = sp.SimulationParameters.empty()


def set_simulation_params(
    simulation_params_json: PathLike = simulation_dir.joinpath("sim_parameters.json"),
):
    """Set simulation parameters from a JSON file."""
    simulation_params = sp.SimulationParameters.update_from_json(simulation_params_json)


### Wild-type growth parameters are read from file
mle_params = gp.MLEGrowthParams.empty()


def set_growth_params(
    growth_params_csv: PathLike = analysis_dir.joinpath(
        "231221_growth_parameters_MLE.csv"
    ),
):
    """Set growth parameters from a CSV file."""
    gp.MLEGrowthParams.update_from_csv(growth_params_csv)


### Steady-state expression is inferred empirically from simulations
# Define dummy functions for steady-state expression
get_steady_state_mean = lambda *args, **kwargs: np.nan
get_steady_state_std = lambda *args, **kwargs: np.nan
get_steady_state_reps = lambda *args, **kwargs: np.nan
get_steady_state_ci_lo = lambda *args, **kwargs: np.nan
get_steady_state_ci_hi = lambda *args, **kwargs: np.nan
get_steady_state_ci = lambda *args, **kwargs: (np.nan, np.nan)
rho_crit_low = np.nan
rho_crit_high = np.nan


def set_steady_state_data(
    ss_sacred_dir: PathLike = simulation_dir.joinpath("20221006_steadystate/sacred"),
):
    """Set steady-state expression data from a directory of Sacred simulations."""
    (
        (
            _get_steady_state_mean,
            _get_steady_state_std,
            _get_steady_state_replicates,
            _get_steady_state_ci_lo,
            _get_steady_state_ci_hi,
        ),
        rho_crit_low,
        rho_crit_high,
    ) = ss._initialize(ss_sacred_dir)
    get_steady_state_mean = numba.vectorize(_get_steady_state_mean)
    get_steady_state_std = numba.vectorize(_get_steady_state_std)
    get_steady_state_reps = numba.vectorize(_get_steady_state_replicates)
    get_steady_state_ci_lo = numba.vectorize(_get_steady_state_ci_lo)
    get_steady_state_ci_hi = numba.vectorize(_get_steady_state_ci_hi)

    def get_steady_state_ci(rho, conf_int=0.8):
        return get_steady_state_ci_lo(rho, conf_int), get_steady_state_ci_hi(
            rho, conf_int
        )


######################################################################
##########  UTILITIES FOR PARALLEL COMPUTING #########################
######################################################################


_dask_client_default_kwargs = dict(
    threads_per_worker=1,
    memory_limit="auto",
    interface="lo",
    timeout=600,
)


######################################################################
##########  CONSTANTS ################################################
######################################################################

# Density at rho = 1 (100% confluence in culture)
ref_density_mm2 = 1250.0  # cells / mm^2
ref_density_um2 = ref_density_mm2 / 1e6  # cells / um^2

# Cell diameter at rho = 1. Approximates each cell as a hexagon
ref_cell_diam_mm = np.sqrt(2 / (np.sqrt(3) * ref_density_mm2))  # mm
ref_cell_diam_um = ref_cell_diam_mm * 1e3  # um


######################################################################
##########  DIFFERENTIAL EQUATION RIGHT-HAND SIDES  ##################
######################################################################


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
    dR_dt = alpha * (S_bar**p) / (k**p + S_bar**p) - R
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
        + alpha * (S_bar**p) / (k**p + (delta * S_delay) ** p + S_bar**p)
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
        alpha * (S_bar**p) / (k**p + (delta * S_delay) ** p + S_bar**p) - R
    ) * gamma_R

    dR_dt[sender_idx] = 0

    return dR_dt


######################################################################
##########  GENERAL UTILITIES  #######################################
######################################################################


# Vectorized integer ceiling
@numba.vectorize
def ceiling(x):
    return ceil(x)


# Vectorized rounding
@numba.vectorize
def vround(x):
    return round(x)


@numba.njit
def first_nonzero(arr, atol=1e-8):
    """Returns index of first nonzero entry in an iterable.
    Returns -1 if none found.
    """
    for idx, val in enumerate(arr):
        if not (-atol < val < atol):
            return idx
    return -1


@numba.njit
def first_zero(arr, atol=1e-8):
    """Returns index of first nonzero entry in an iterable.
    Returns -1 if none found.
    """
    for idx, val in enumerate(arr):
        if -atol < val < atol:
            return idx
    return -1


@numba.vectorize
def normalize(x, xmin, xmax):
    """Normalize `x` given explicit min/max values."""
    return (x - xmin) / (xmax - xmin)


@numba.vectorize
def logistic(t, g, rho_0, rho_max):
    """Returns logistic equation evaluated at time `t`."""
    return rho_0 * rho_max / (rho_0 + (rho_max - rho_0) * np.exp(-g * t))


@numba.vectorize
def logistic_inv(rho, g, rho_0, rho_max):
    """Inverse of logistic equation (returns time at a given density `rho`)."""
    if rho_0 < rho_max:
        return np.log(rho * (rho_max - rho_0) / (rho_0 * (rho_max - rho))) / g
    else:
        return np.nan


@numba.vectorize
def logistic_solve_rho_0(rho, t, g, rho_max):
    """Solve logistic equation for initial condition (returns `rho` at time zero)."""
    return rho * rho_max * np.exp(-g * t) / (rho_max - rho * (1 - np.exp(-g * t)))


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


######################################################################
##########  HEXAGONAL LATTICE GENERATION AND ADJACENCY  ##############
######################################################################


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


def get_center_cells(X, n_center=1):
    """Returns indices of the n_cells cells closest to the origin given their coordinates as an array X."""
    return np.argpartition([np.linalg.norm(x) for x in X], n_center)[:n_center]


def transform_lattice(X, rho):
    """Scale lattice coordinates by a factor determined by the relative
    density `rho`.
    """
    return np.multiply.outer(1 / np.sqrt(rho), X)


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
    dx = width / (2 * np.sqrt(1 + pslope**2))
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
    Suu = np.sum(u**2)
    Suv = np.sum(u * v)
    Svv = np.sum(v**2)

    Suuu = np.sum(u**3)
    Suuv = np.sum((u**2) * v)
    Suvv = np.sum(u * (v**2))
    Svvv = np.sum(v**3)

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
    R = np.sqrt((uv_c**2).sum() + (Suu + Svv) / N)

    return xy_c, R


####### Unit conversions


@numba.njit
def _t_to_units(dimless_time, ref_growth_rate):
    return dimless_time / ref_growth_rate


def t_to_units(dimless_time, ref_growth_rate=None):
    """Convert dimensionless time to real units for a growth process.

    Returns
    -------
    time  :  number or numpy array (dtype float)
        Time in units (hours, days, etc.). Defaults to days.

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
        Defaults to the growth rate of wild-type transceivers in units of
        inverse days.
    """
    if ref_growth_rate is None:
        ref_growth_rate = mle_params.g_inv_days
    return _t_to_units(dimless_time, ref_growth_rate)


@numba.njit
def _g_to_units(dimless_growth_rate, ref_growth_rate):
    return dimless_growth_rate * ref_growth_rate


def g_to_units(dimless_growth_rate, ref_growth_rate=None):
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
    if ref_growth_rate is None:
        ref_growth_rate = mle_params.g_inv_days
    return _g_to_units(dimless_growth_rate, ref_growth_rate)


@numba.njit
def rho_to_units(rho, ref_density=ref_density_mm2):
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


def ncells_to_area(ncells, rho, ref_density=ref_density_mm2):
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


@numba.vectorize
def hexagon_side_to_area(side):
    """Return area of a hexagon given side length.

    Returns
    -------
    area :  number or numpy array (dtype float)
        Area of hexagon

    Parameters
    ----------
    side  :  number or numpy array (dtype float)
        Length of side

    """
    return 3 * np.sqrt(3) / 2 * side**2


@numba.vectorize
def area_to_hexagon_side(area):
    """Return side length of a hexagon given area

    Returns
    -------
    side  :  number or numpy array (dtype float)
        Length of side

    Parameters
    ----------

    area :  number or numpy array (dtype float)
        Area of hexagon

    """
    return np.sqrt(area * 2 / (3 * np.sqrt(3)))


@numba.vectorize
def area_to_radius(area):
    """Return radius of a circle given its area

    Returns
    -------
    radius  :  number or numpy array (dtype float)
        Circle radius

    Parameters
    ----------

    area :  number or numpy array (dtype float)
        Circle area

    """
    return np.sqrt(area / np.pi)


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


def get_t_ON(g, rho_0, rho_max=None, rho_crit_low=None):
    """Return the time at which signaling will turn ON/OFF. Based on the
    logistic growth equation and a supplied threshold value."""
    if rho_max is None:
        rho_max = mle_params.rho_max_ratio
    if rho_crit_low is None:
        rho_crit_low = rho_crit_low
    return logistic_inv(rho_crit_low, g, rho_0, rho_max)


def get_t_OFF(g, rho_0, rho_max=None, rho_crit_high=None):
    """Return the time at which signaling will turn ON/OFF. Based on the
    logistic growth equation and a supplied threshold value."""
    if rho_max is None:
        rho_max = mle_params.rho_max_ratio
    if rho_crit_high is None:
        rho_crit_high = rho_crit_high
    return logistic_inv(rho_crit_high, g, rho_0, rho_max)


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


####################################################################
############ DENSITY SENSITIVITY FUNCTIONS #########################
####################################################################


@numba.njit
def beta_rho_exp(rho, m, *args):
    return np.exp(-m * np.maximum(rho - 1, 0))


@numba.njit
def beta_rho_two_sided(rho, m):
    r = np.where(rho < 1, 1 / rho, rho)
    return np.exp(-m * (r - 1))


@numba.njit
def beta_rho_with_low_density(rho, m, q):
    return np.where(rho < 1, rho**q, np.exp(-m * (rho - 1)))


_beta_function_dictionary = OrderedDict(
    [
        ["one_sided", beta_rho_exp],
        ["two_sided", beta_rho_two_sided],
        ["two_sided_different_functions", beta_rho_with_low_density],
    ]
)


def get_beta_func(func_name):
    return _beta_function_dictionary[str(func_name)]


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


#### Visualization module
import _viz as viz
