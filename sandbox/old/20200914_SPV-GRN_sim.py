import numpy as np
import pandas as pd
import scipy.sparse as sprs
import scipy.optimize as opt
from scipy.spatial import ConvexHull

import math
import numba

import triangle as tr

import os
from glob import glob

import tqdm

###########################

@numba.njit
def get_rms(y):
    return np.sqrt(np.mean(y**2))


@numba.njit
def logistic(x, a, b, N):
    return N/(1 + a * np.exp(-b * x))


@numba.njit
def logistic_norm(x, a, b):
    return 1/(1 + a * np.exp(-b * x))


@numba.njit
def rms_mask(m, loc, ip):
    """
    Returns RMS distance of pixels in a mask to location loc, 
    in units of interpixel distance ip.
    
    m  : mask, as a Numpy array of indices of shape (ndim, npix)
    """
    
    ndim, npix = m.shape
    
    # Catch empty masks
    if npix==0:
        return 0
    
    # calculate squared distance
    sqd = np.zeros(npix)
    for i in range(ndim):
        sqd += (m[i] - loc[i])**2
    
    # Root-mean of squared distance (RMSD) in units of distance
    return np.sqrt(np.mean(sqd)) * ip


def chull_mask(m, ip):
    """
    Returns area of the convex hull of pixels in a mask, in units
    of squared distance
    
    m  : mask as a 2D Numpy array of shape (ndim, npix)
    ip : inter-pixel distance
    """
    
    # If not enough points, return 0
    ndim, npix = m.shape
    if npix < 3:
        return 0
    
    return spat.ConvexHull(ip * m.T).volume


@numba.njit
def make_y(x, Lgrid_xy):
    """
    Makes the (9) tiled set of coordinates used to perform the periodic triangulation.

    :param x: Cell centroids (n_c x 2) np.float32 array
    :param Lgrid_xy: (9 x 2) array defining the displacement vectors for each of the 9 images of the tiling
    :return: Tiled set of coordinates (9n_c x 2) np.float32 array
    """
    n_c = x.shape[0]
    y = np.empty((n_c * 9, x.shape[1]))
    for k in range(9):
        y[k * n_c : (k + 1) * n_c] = x + Lgrid_xy[k]
    return y


def remove_repeats(tri, n_c):
    """
    For a given triangulation (nv x 3), remove repeated entries (i.e. rows)

    The triangulation is first re-ordered, such that the first cell id referenced is the smallest. Achieved via
    the function order_tris. (This preserves the internal order -- i.e. CCW)

    Then remove repeated rows via lexsort.

    NB: order of vertices changes via the conventions of lexsort

    Inspired by...
    https://stackoverflow.com/questions/31097247/remove-duplicate-rows-of-a-numpy-array

    :param tri: (nv x 3) matrix, the triangulation
    :return: triangulation minus the repeated entries (nv* x 3) (where nv* is the new # vertices).
    """
    tri = order_tris(np.mod(tri, n_c))
    sorted_tri = tri[np.lexsort(tri.T), :]
    row_mask = np.append([True], np.any(np.diff(sorted_tri, axis=0), 1))
    return sorted_tri[row_mask]


def grid_xy(L):
    grid_x, grid_y = np.mgrid[-1:2, -1:2]
    grid_x[0, 0], grid_x[1, 1] = grid_x[1, 1], grid_x[0, 0]
    grid_y[0, 0], grid_y[1, 1] = grid_y[1, 1], grid_y[0, 0]
    return L * np.array([grid_x.ravel(), grid_y.ravel()]).T


@numba.njit
def order_tris(tri):
    """
    For each triangle (i.e. row in **tri**), order cell ids in ascending order
    :param tri: Triangulation (n_v x 3) np.int32 array
    :return: the ordered triangulation
    """
    nv = tri.shape[0]
    for i in range(nv):
        Min = np.argmin(tri[i])
        tri[i] = tri[i, Min], tri[i, np.mod(Min + 1, 3)], tri[i, np.mod(Min + 2, 3)]
    return tri


def _triangulate_periodic(x, L):
    """
    Calculates the periodic triangulation on the set of points x.
    Stores:
        n_v = number of vertices (int32)
        tris = triangulation of the vertices (nv x 3) matrix.
            Cells are stored in CCW order. As a convention, the first entry has the smallest cell id
            (Which entry comes first is, in and of itself, arbitrary, but is utilised elsewhere)
        vs = coordinates of each vertex; (nv x 2) matrix
        v_neighbours = vertex ids (i.e. rows of vs) corresponding to the 3 neighbours of a given vertex (nv x 3).
            In CCW order, where vertex i {i=0..2} is opposite cell i in the corresponding row of tris
        neighbours = coordinates of each neighbouring vertex (nv x 3 x 2) matrix
    :param x: (nc x 2) matrix with the coordinates of each cell
    """
    # 1. Tile cell positions 9-fold to perform the periodic triangulation
    #   Calculates y from x. y is (9nc x 2) matrix, where the first (nc x 2) are the "true" cell positions,
    #   and the rest are translations
    y = make_y(x, grid_xy(L))
    
    # 2. Perform the triangulation on y
    #   The **triangle** package (tr) returns a dictionary, containing the triangulation.
    #   This triangulation is extracted and saved as tri
    t = tr.triangulate({"vertices": y})
    tri = t["triangles"]
    # Del = Delaunay(y)
    # tri = Del.simplices
    n_c = x.shape[0]
    
    # 3. Find triangles with **at least one** cell within the "true" frame (i.e. with **at least one** "normal cell")
    #   (Ignore entries with -1, a quirk of the **triangle** package, which denotes boundary triangles
    #   Generate a mask -- one_in -- that considers such triangles
    #   Save the new triangulation by applying the mask -- new_tri
    tri = tri[(tri != -1).all(axis=1)]
    one_in = (tri < n_c).any(axis=1)
    new_tri = tri[one_in]
    
    # 4. Remove repeats in new_tri
    #   new_tri contains repeats of the same cells, i.e. in cases where triangles straddle a boundary
    #   Use remove_repeats function to remove these. Repeats are flagged up as entries with the same trio of
    #   cell ids, which are transformed by the mod function to account for periodicity. See function for more details
    n_tri = remove_repeats(new_tri, n_c)
    
    return n_tri
    
#     # tri_same = (tris == n_tri).all()
#     # 6. Store outputs
#     n_v = n_tri.shape[0]
#     tris = n_tri
#     Cents = x[tris]
#     vs = get_vertex_periodic()
    
#     # 7. Manually calculate the neighbours. See doc_string for conventions.
#     n_neigh = get_neighbours(n_tri)
#     v_neighbours = n_neigh
#     neighbours = vs[n_neigh]


@numba.njit
def get_neighbours(tri):
    """
    Given a triangulation, find the neighbouring triangles of each triangle.
    By convention, the column i in the output -- neigh -- corresponds to the triangle that is opposite the cell i in that triangle.
    Can supply neigh, meaning the algorithm only fills in gaps (-1 entries)
    :param tri: Triangulation (n_v x 3) np.int32 array
    :param neigh: neighbourhood matrix to update {Optional}
    :return: (n_v x 3) np.int32 array, storing the three neighbouring triangles. Values correspond to the row numbers of tri
    """
    n_v = tri.shape[0]
    neigh = np.ones_like(tri, dtype=np.int32) * -1
    tri_compare = np.concatenate((tri.T, tri.T)).T.reshape((-1, 3, 2))
    for j in range(n_v):
        tri_sample_flip = np.flip(tri[j])
        tri_i = np.concatenate((tri_sample_flip, tri_sample_flip)).reshape(3, 2)
        for k in range(3):
            if neigh[j, k] == -1:
                neighb, l = np.nonzero(
                    (tri_compare[:, :, 0] == tri_i[k, 0])
                    * (tri_compare[:, :, 1] == tri_i[k, 1])
                )
                neighb, l = neighb[0], l[0]
                neigh[j, k] = neighb
                neigh[neighb, np.mod(2 - l, 3)] = j
    return neigh


@numba.njit
def roll_forward(x):
    """
    Jitted equivalent to np.roll(x,1,axis=1)
    :param x:
    :return:
    """
    return np.column_stack((x[:, 2], x[:, :2]))


@numba.njit
def roll_reverse(x):
    """
    Jitted equivalent to np.roll(x,-1,axis=1)
    :param x:
    :return:
    """
    return np.column_stack((x[:, 1:3], x[:, 0]))


@numba.njit
def circumcenter_periodic(C, L):
    """
    Find the circumcentre (i.e. vertex position) of each triangle in the triangulation.

    :param C: Cell centroids for each triangle in triangulation (n_c x 3 x 2) np.float32 array
    :param L: Domain size (np.float32)
    :return: Circumcentres/vertex-positions (n_v x 2) np.float32 array
    """
    ri, rj, rk = C.transpose(1, 2, 0)
    r_mean = (ri + rj + rk) / 3
    disp = r_mean - L / 2
    ri, rj, rk = np.mod(ri - disp, L), np.mod(rj - disp, L), np.mod(rk - disp, L)
    ax, ay = ri
    bx, by = rj
    cx, cy = rk
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    ux = (
        (ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)
    ) / d
    uy = (
        (ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)
    ) / d
    vs = np.empty((ax.size, 2), dtype=np.float64)
    vs[:, 0], vs[:, 1] = ux, uy
    vs = np.mod(vs + disp.T, L)
    return vs


def get_CV_matrix(tri, n_c):
    n_v = tri.shape[0]
    CV_matrix = np.zeros((n_c, n_v, 3))
    for i in range(3):
        CV_matrix[tri[:, i], np.arange(n_v), i] = 1
    return CV_matrix.astype(np.float32)


@numba.njit
def get_l_interface_dense(n_v, n_c, neighbours, vs, CV_matrix, L):
    """
    Get the length of the interface between each pair of cells.
    LI[i,j] = length of interface between cell i and j = L[j,i] (if using periodic triangulation)
    :param n_v: Number of vertices (**np.int64**)
    :param n_c: Number of cells (**np.int64**
    :param neighbours: Position of the three neighbouring vertices (n_v x 3 x 2)
    :param vs: Positions of vertices (n_v x 3)
    :param CV_matrix: Cell-vertex matrix representation of triangulation (n_c x n_v x 3)
    :param L: Domain size (**np.float32**)
    :return:
    """
    
    h_j = np.empty((n_v, 3, 2))
    for i in range(3):
        h_j[:, i] = vs
    h_jp1 = np.dstack(
        (roll_reverse(neighbours[:, :, 0]), roll_reverse(neighbours[:, :, 1]))
    )
    l = np.mod(h_j - h_jp1 + L / 2, L) - L / 2
    l = np.sqrt(l[:, :, 0] ** 2 + l[:, :, 1] ** 2)
    l = l.astype(np.float32)
    LI = np.zeros((n_c, n_c), dtype=np.float32)
    for i in range(3):
        LI += np.asfortranarray(l[:, i] * CV_matrix[:, :, i]) @ np.asfortranarray(
            CV_matrix[:, :, np.mod(i + 2, 3)].T.astype(np.float32)
        )
    return LI


def get_transition_mtx(*args, **kwargs):
    # Get pairwise interface lengths as sparse matrix
    mtx = get_l_interface_dense(*args, **kwargs)
    mtx = sprs.csr_matrix(mtx)
    
    # Row-normalize and return
    return mtx.multiply(1 / np.sum(mtx, axis=1)) 


def x_to_transition_mtx(x, L):
    n_c = x.shape[0]
    tri = _triangulate_periodic(x, L)
    n_v = tri.shape[0]
    vs = circumcenter_periodic(x[tri], L)
    neighbours = vs[get_neighbours(tri)]
    CV_matrix = get_CV_matrix(tri, n_c)
    return get_transition_mtx(n_v, n_c, neighbours, vs, CV_matrix, L)


def assign_types_random(types, type_n_c, n_c):
    """Randomly assigns indices to cell types."""
    # Generate random indices
    idx = np.arange(n_c)
    np.random.shuffle(idx)

    # Assign indices
    type_idx = np.split(idx, np.cumsum(type_n_c))[:-1]

    return type_idx


def assign_types_center(types, type_n_c, x0, center=0):
    """
    Assigns indices to cell types. The cell type given by types[center] 
    will be closest to the center of x0, the set of cell location coordinates 
    given as a (n_c x 2) Numpy array.
    """
    # Get indices of center cells
    norm_coords = x0 - np.mean(x0, axis=0)
    center_idx = np.argsort(np.linalg.norm(norm_coords, axis=1))[: type_n_c[center]]
    
    # Assign center cells
    type_idx = [None] * len(types)
    type_idx[center] = center_idx
    
    # Randomly shuffle remaining indices
    n_c = x0.shape[0]
    idx = np.arange(n_c)
    idx = np.delete(idx, center_idx)
    np.random.shuffle(idx)
    
    # Assign to remaining cell types
    cumsum = 0
    for i in range(len(types)):
        if i == center:
            type_idx[i] = center_idx
        else:
            type_idx[i] = idx[cumsum : cumsum + type_n_c[i]]
            cumsum += type_n_c[i]

    return type_idx


def assign_types(types, type_n_c, type_method="random", n_c=0, x0=[], center=0):
    """Returns a list of Numpy arrays assigning indices to each cell type in types.
    The number of indices in each array corresponds to the number of cells specified
    in type_n_c. One element of type_n_c can be -1, in which case it will be replaced
    with the appropriate number to reach the total number of cells. Either n_c or x0 
    must be provided."""
    
    # Perform checks
    types = np.array(types)
    type_n_c = np.array(type_n_c)
    
    assert ((n_c !=0) | (len(x0) != 0)), "Must supply either n_c or x0"
    
    # Get number of cells
    if len(x0) > 0:
        n_c = x0.shape[0]
    
    if sum(type_n_c) < n_c:
        assert (
            sum([x == -1 for x in type_n_c]) == 1
        ), "Number of cells in type_n_c does not match the initial number of cells."

    elif sum(type_n_c) == n_c:
        assert all(
            [x >= 0 for x in type_n_c]
        ), "Number of cells in type_n_c does not match the initial number of cells."

    else:
        assert (
            False
        ), "Number of cells in type_n_c does not match the initial number of cells."
    
    # Replace -1 with correct number of cells
    if any([x == -1 for x in type_n_c]):
        type_n_c[np.argwhere(type_n_c < 0)] = n_c - sum(type_n_c) - 1
    
    # Assign types
    if type_method == "random":
        return assign_types_random(types, type_n_c, n_c)
    elif type_method == "center":
        return assign_types_center(types, type_n_c, x0, center)


def init_GRN(n_c, type_idx, init_vals):
    E = np.empty(n_c, dtype=np.float32)
    for i, _ in enumerate(types):
        E[type_idx[i]] = init_vals[i]
    return E


def simulate_GRN_delay(
    t_span,
    rhs_delay,
    dde_params,
    x_save,
    L,
    types,
    type_n_c,
    init_vals,
    type_method="center",
    center=0,
    skip=1,
    progress_bar=False,
):
    # Get time parameters and delay
    n_t_input = x_save.shape[0]
    x_save = np.copy(x_save)[::skip]
    t_span = np.copy(t_span)[::skip]
    n_t = t_span.size
    GRN_dt = t_span[1] - t_span[0]
    
    # Get delay in steps
    step_delay = np.atleast_1d(delay) / GRN_dt
    assert (step_delay >= 1), "Step delay is too small. Consider skipping fewer steps or lowering dt."
    step_delay = math.ceil(step_delay)
    
    # Assign cell types
    type_idx = assign_types(
        types, type_n_c, x0=x_save[0], type_method="center", center=center
    )
    
    # Define integration function using RHS
    def step_GRN(E, E_past, A_past, GRN_dt):
        E_bar = A_past @ E_past
        dE_dt = rhs_delay(E, E_past, E_bar, *dde_params)
        dE_dt[type_idx[center]] = 0
        return np.maximum(0, E + dE_dt * GRN_dt)

    # Initialize expression vector
    n_c = x_save.shape[1]
    E = init_GRN(n_c, type_idx, init_vals)
    E_save = np.empty((n_t, n_c), dtype=np.float32)
    E_save[0] = E
    
    # Construct time iterator
    iterator = np.arange(1, n_t)
    if progress_bar:
        iterator = tqdm.tqdm(iterator)

    for step in iterator:
        
        # Get past expression and transition matrix
        past_step = max(0, step - step_delay)
        E_past = E_save[past_step]
        A_past = x_to_transition_mtx(x_save[past_step], L)
        
        # Integrate and set sender cells constant
        E = step_GRN(E, E_past, A_past, GRN_dt)
        E_save[step] = E
    
    # Repeat array for any skipped steps
    return np.repeat(E_save, skip, axis=0)[:n_t_input]


###########################


# Set directories
os.chdir("/home/ubuntu/git")
data_dir = "/home/ubuntu/git/evomorph/data/2020-09-09_avm_phase_sims/"

# Read metadata of all sims in batch
df = pd.read_csv(os.path.join(data_dir, "metadata_full.csv"), index_col=0)
df = df.sort_values("data_fname")
files = df.data_fname.values

# Get GRN time-span
t_span = np.linspace(df["t0"][0], df["tmax"][0], df["n_t"][0])

# Set cell types, their number of cells, and their initial expression
#  Note: assign_types() interprets -1 as "all other cells"
types     = ("sender", "transceiver")
type_n_c  = (1, -1)
init_vals = (1,  0)

# Define RHS
@numba.njit
def tc_rhs(E, E_past, E_bar, alpha, k, p, delta, lambda_):
    """
    Returns RHS of transceiver DDE.
    """
    dE_dt = lambda_ + alpha * (E_bar**p) / (k**p + (delta * E_past)**p + E_bar**p) - E
    return dE_dt

# Set DDE params
alpha = 3
k = 0.01
p = 2
delta = 3
lambda_ = 1e-5
dde_params = alpha, k, p, delta, lambda_

# Set delay
delay = 0.4

# Set directories to save DDE sims
GRN_dir = os.path.join(data_dir, "Esave")
if not os.path.exists(GRN_dir):
    os.mkdir(GRN_dir)

###########################

def simulate(f, skip=20, thresh=0.1):
    # Extract data and metadata
    x_save = np.load(os.path.join(data_dir, f))
    metadata = df.loc[df["data_fname"] == f,]
    metadata = metadata.iloc[0, :].to_dict()

    # Get GRN time-span
    t_span = np.linspace(metadata["t0"], metadata["tmax"], metadata["n_t"])
    
    # Simulate signaling using DDE
    E_save = simulate_GRN_delay(
        t_span,
        rhs_delay=tc_rhs,
        dde_params=dde_params,
        x_save=x_save,
        L=metadata["L"],
        types=types,
        type_n_c=type_n_c,
        init_vals=init_vals,
        skip=skip,
        progress_bar=False,
    )

    # Save GRN 
    np.save(os.path.join(GRN_dir, f[:-4] + "_Esave"), E_save, allow_pickle=False)

    # Compress array
    E_save = E_save[::skip]
    
    # Get additional params
    L = metadata["L"]
    n_t, n_c = E_save.shape

    #Apply threshold and calculate proportion of population
    E_thresh = E_save > thresh
    E_thresh_prop = np.sum(E_thresh, axis=1) / n_c
    
    rmss = np.empty(n_t)
    chull_vols = np.empty(n_t)
    for i, X in enumerate(x_save[::skip]):
        
        n_thresh = sum(E_thresh[i])
        if n_thresh == 0:
            rmss[i] = 0
            chull_vols[i] = 0
        elif n_thresh < 3:
            chull_vols[i] = 0
        else:
            X = X - L/2
            d = X[E_thresh[i].nonzero()[0], :]
            rmss[i] = get_rms(d)
            chull_vols[i] = ConvexHull(d).volume
    
    num_gr = opt.curve_fit(
        logistic_norm, 
        t_span[::skip], 
        E_thresh_prop,
        bounds=((0, 0), (np.inf, np.inf)),
    )[0][1]
    rms_gr = opt.curve_fit(
        logistic, 
        t_span[::skip], 
        rmss,
        bounds=((0, 0, 0,), (np.inf, np.inf, L*np.sqrt(2))),
    )[0][1]
    chull_gr = opt.curve_fit(
        logistic,
        t_span[::skip],
        chull_vols,
        bounds=((0, 0, 0,), (np.inf, np.inf, L**2,)),
    )[0][1]
    
    print(f"Thread {count()*100:.2f}% complete")
    
    return np.array([num_gr, rms_gr, chull_gr], dtype=np.float32)

def pcounter(n):
    i = 1
    while True:
        yield i/n
        i += 1

cores = 8
gen = pcounter((files.size // cores) + 1)
def count():
    return next(gen)

from multiprocessing import Pool
if __name__ == '__main__':
    with Pool(cores) as p:
        results = list(p.imap_unordered(simulate, files))

gr_df = pd.DataFrame(dict(
    data_fname        = files,
    cells_growth_rate = [x[0] for x in results], 
    RMSD_growth_rate  = [x[1] for x in results],
    CHull_growth_rate = [x[2] for x in results],
))
gr_df.to_csv(os.path.join(data_dir, "growthmetrics.csv"))

##################

print("Kul.")