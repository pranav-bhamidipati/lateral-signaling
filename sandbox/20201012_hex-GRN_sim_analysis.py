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
def get_D_eff(X0, Xmax, L, tmax, v0, Dr, n=19):
    
    X0_norm = X0 - L/2
    center_cells = np.argsort(np.sqrt(X0_norm[:, 0]**2 + X0_norm[:, 1]**2))[:n]
    
    dX = np.fmod(Xmax[center_cells] - X0[center_cells], L/2)
    dr2 = np.zeros(n, dtype=np.float32)
    for i in range(n):
        dr2[i] = dX[i, 0] ** 2 + dX[i, 1] ** 2
    
    Ds = dr2.mean() / (4*tmax)
    D0 = v0**2/(2*Dr)
    return Ds / D0

def npy_to_D_eff(fname, metadata, n=7):
    
    X = np.load(fname)
#     X = X[:X.shape[0]//2]
    n_t = X.shape[0]
    X0, Xmax = X[0], X[-1]
    kwargs = metadata.loc[metadata["filename"] == fname, :].to_dict()
    kwargs = {k: tuple(v.values())[0] for k, v in kwargs.items()}

    return get_D_eff(
        X0,
        Xmax,
        kwargs["L"],
        n_t * kwargs["dt"],
        kwargs["v0"],
        kwargs["Dr"],
        n=n
    )


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


###########################


# Set directories
# os.chdir("/home/ubuntu/git")
data_dir = "/home/ubuntu/git/evomorph/data/2020-10-01_p0v0dens_phase_sims2/"
GRN_dir = os.path.join(data_dir, "Esave")

# Read GRN metadata
Esave_metadata = pd.read_csv(
    os.path.join(data_dir, "Esave", "Esave_metadata.csv"), index_col=0
)
metadicts = [row.to_dict() for _, row in Esave_metadata.iterrows()]

# # Read metadata of all sims in batch
# df = pd.read_csv(os.path.join(data_dir, "metadata.csv"), index_col=0)
# df = df.sort_values("coords_fname")

# Get GRN time-span
GRN_tmax = 5
t0, tmax, n_t = (
    Esave_metadata["t0"][0],
    Esave_metadata["tmax"][0],
    Esave_metadata["n_t"][0],
)
t_span = np.linspace(t0, tmax, n_t)
GRN_dt = Esave_metadata["dt"][0] / Esave_metadata["f"][0]
GRN_t_span = np.linspace(0, GRN_tmax, int(GRN_tmax / GRN_dt) + 1)

# D_eff = np.empty(metadata.shape[0])
# for i, file in iterator:
#     D_eff[i] = npy_to_D_eff(file, metadata)


###########################

"""
For each sim,
- read in data
- threshold
- calc growth
- keep track of all SPV and GRN params
- save growth metrics as metadata/analytics
"""

# Read in expression data


###########################

# # Sub-sample velocity and shape
# v0s = np.unique(df.v0)[1:2]
# p0s = np.unique(df.p0)[::2]
# p0v0_bool = np.logical_and(np.isin(df.v0, v0s), np.isin(df.p0, p0s))
# metadf = df.loc[p0v0_bool, :]
# x_files = [os.path.abspath(os.path.join(data_dir, p + ".npz")) for p in metadf.coords_fname.values]
# x_saves_frozen = [np.load(x)["arr_0"][-1] for x in x_files]

# # Set parameter space
# alpha_space = np.linspace(1, 3, 3)
# k_space = np.array([1e-2, 5e-2, 25e-2])
# p_space = [2]
# delta_space = np.linspace(0, 3, 3)
# lambda_space = [1e-5]
# spaces = [alpha_space, k_space, p_space, delta_space, lambda_space,]
# param_space = np.meshgrid(*spaces)
# param_space = np.array(param_space).T.reshape(-1, len(spaces))

# # Set DDE params
# alpha = 3
# k = 0.01
# p = 2
# delta = 3
# lambda_ = 1e-5

# # Set delay
# delay = 0.4

###########################

def analyze_growth(kwargs, skip=20, thresh=0.1, count_skip=10):
    """
    """
    # Extract data
    xf = os.path.join(data_dir, kwargs["coords_fname"] + ".npz")
    Ef = os.path.join(data_dir, "Esave", kwargs["Esave_fname"] + ".npz")
    X = np.load(xf)["arr_0"][-1]
    E_save = np.load(Ef)["arr_0"]

    # Get additional params
    L = kwargs["L_norm"] / np.sqrt(kwargs["dens"])
    n_t, n_c = E_save.shape

    # Normalize cell locations
    X = X - L/2

    # Compress
    E_save = E_save[::skip]
    t = GRN_t_span[::skip]

    # Apply threshold and calculate proportion of population
    E_thresh = E_save > thresh
    E_thresh_prop = np.sum(E_thresh, axis=1) / n_c
    
    rmss = np.empty(t.size)
    chull_vols = np.empty(t.size)
    for i, et in enumerate(E_thresh):
        
        n_thresh = sum(et)
        if n_thresh == 0:
            rmss[i] = 0
            chull_vols[i] = 0
        elif n_thresh < 3:
            d = X[E_thresh[i].nonzero()[0], :]
            rmss[i] = get_rms(d)
            chull_vols[i] = 0
        else:
            d = X[E_thresh[i].nonzero()[0], :]
            rmss[i] = get_rms(d)
            chull_vols[i] = ConvexHull(d).volume
    
    num_gr = opt.curve_fit(
        logistic, 
        t, 
        E_thresh_prop,
        bounds=((0, 0, 0), (np.inf, np.inf, 1)),
    )[0][1]
    rms_gr = opt.curve_fit(
        logistic, 
        t, 
        rmss,
        bounds=((0, 0, 0,), (np.inf, np.inf, L*np.sqrt(2))),
    )[0][1]
    chull_gr = opt.curve_fit(
        logistic,
        t,
        chull_vols,
        bounds=((0, 0, 0,), (np.inf, np.inf, L**2,)),
    )[0][1]
    
    c = count()
    if (c % count_skip == 0):
        print(f"Thread completed run # {c}")

    return np.array([num_gr, rms_gr, chull_gr], dtype=np.float32)

def counter():
    i = 1
    while True:
        yield i
        i += 1

cores = 8
gen = counter()
def count():
    return next(gen)

print(f"Simulating {len(metadicts)} runs on {cores} cores ({len(metadicts)/cores:.2f} per core)")
from multiprocessing import Pool
if __name__ == '__main__':
    with Pool(cores) as p:
        results = list(p.imap_unordered(analyze_growth, metadicts))

gr_df = pd.DataFrame(dict(
    Esave_fname       = Esave_metadata.Esave_fname,
    coords_fname      = Esave_metadata.coords_fname,
    cells_growth_rate = [x[0] for x in results], 
    RMSD_growth_rate  = [x[1] for x in results],
    CHull_growth_rate = [x[2] for x in results],
))
Esave_metadata.merge(
    gr_df
).to_csv(os.path.join(data_dir,"Esave", "growthmetrics.csv"))

##################

print("Kul.")