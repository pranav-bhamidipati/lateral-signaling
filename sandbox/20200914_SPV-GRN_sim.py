import os
from glob import glob

import numpy as np
import pandas as pd
import numba
import tqdm

import scipy.optimize as opt
from scipy.spatial import ConvexHull

##############################

def normalize(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)


@numba.njit
def get_rms(y):
    """Returns root-mean-squared of `y`, a 2D Numpy array
    of displacement (n_samples x n_dim)"""
    # Squared displacement
    sqd = np.sum(y ** 2, axis=1)
    
    # Root-mean of sqd
    return np.sqrt(np.mean(sqd))


@numba.njit
def logistic(x, a, b, N):
    return N/(1 + a * np.exp(-b * x))


@numba.njit
def logistic_norm(x, a, b):
    return 1/(1 + a * np.exp(-b * x))


##############################

# Get data 
data_dir = "/hom/ubuntu/git/evomorph/data/2020-09-09_avm_phase_sims/"
# example_files = sorted([os.path.split(f)[-1] for f in glob(data_dir + "*rep?.*", )])

# Read metadata of all sims in batch
df = pd.read_csv(os.path.join(data_dir, "metadata_full.csv"), index_col = 0)
del df["D_eff_n_c"]
del df["D_eff"]
df = df.drop_duplicates()
# df = df.loc[np.isin(df["data_fname"], example_files), :].reset_index(drop=True)
df = df.sort_values("data_fname")

##############################

thresh = 0.1
num_growth = []
rms_growth = []
chull_growth = []

# iterator = example_files
iterator = np.unique(df.data_fname.values)
iterator = tqdm.tqdm(iterator)
skip = 20

for f in iterator:

    # Extract data and metadata
    x_save = np.load(os.path.join(data_dir, f))[::skip]
    E_save = np.load(os.path.join(data_dir, f[:-4] + "_Esave.npy"))[::skip]
    metadata = df.loc[df["data_fname"] == f,]
    metadata = metadata.iloc[0, :].to_dict()

    # Get GRN time-span
    t_span = np.linspace(metadata["t0"], metadata["tmax"], metadata["n_t"])[::skip]

    # Get additional params
    L = metadata["L"]
    n_c = x_save.shape[1]
    n_t = x_save.shape[0]
    
    # Apply threshold and calculate proportion of population
    E_thresh = E_save > thresh
    E_thresh_prop = np.sum(E_thresh, axis=1) / n_c
    
    rmss = np.empty(n_t)
    chull_vols = np.empty(n_t)
    
    for i, X in enumerate(x_save):
        X = X - L/2
        d = X[E_thresh[i].nonzero()[0], :]
        rmss[i] = get_rms(d)

        if sum(E_thresh[i]) < 3:
            chull_vols[i] = 0
        else:
            chull_vols[i] = ConvexHull(X[E_thresh[i].nonzero()[0]]).volume

    num_growth.append(opt.curve_fit(logistic_norm, t_span, E_thresh_prop)[0][1])
    rms_growth.append(opt.curve_fit(logistic, t_span, rmss)[0][1])
    chull_growth.append(
        opt.curve_fit(
            logistic,
            t_span,
            chull_vols,
            bounds=((-np.inf, -np.inf, 0,), (np.inf, np.inf, L**2,)),
        )[0][1]
    )

##################

gr_df["cells growth rate"] = num_growth
gr_df["RMSD growth rate"] = rms_growth
gr_df["CHull growth rate"] = chull_growth
# gr_df["_v0"] = [f"{s:.3f}" for s in gr_df.v0]
gr_df.to_csv(os.path.join(data_dir, "growthrate_full.csv"))
