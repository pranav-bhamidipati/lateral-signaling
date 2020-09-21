import os
from glob import glob

import numpy as np
import pandas as pd
import numba
import tqdm

import scipy.optimize as opt
from scipy.spatial import ConvexHull

vor_path = "/home/ubuntu/git/active_vertex"
# vor_path = 'C:\\Users\\Pranav\\git\\active_vertex'
sys.path.append(vor_path)

import voronoi_model.voronoi_model_periodic as avm

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

def simulate(f):
    p, v, rep = params
    prefix = f"p0{p:.2f}_v0{v:.2e}_rep{int(rep)}"
    
    vor2 = avm.Tissue()
    vor2.generate_cells(600)
    vor2.make_init(10)
    
    vor2.set_GRN_t_span(t0, tmax, n_t, scaling_factor=f);
    vor2.v0 = v
    vor2.n_warmup_steps = int(150 / dt)

    W = J * np.array([[1, 0], [0, 1]])
    vor2.set_interaction(W=W, pE=0)
    vor2.A0 = 0.86
    vor2.P0 = p * np.sqrt(vor2.A0)
    
    vor2.Dr = 0.01
    vor2.kappa_A = 0.2
    vor2.kappa_P = 0.1
    vor2.a = a
    vor2.k = k
    
    vor2.simulate(progress_bar=False, print_updates=False);
    
    fname = to_dir + prefix + ".npy"
    np.save(fname, vor2.x_save, allow_pickle=False)
    
    print(f"Thread {count()*100:.2f}% complete")

    return prefix, [vor_to_D_eff(vor2, n) for n in (7, 19, 37)]

##############################

# Get data 
data_dir = "/hom/ubuntu/git/evomorph/data/2020-09-09_avm_phase_sims/"
# example_files = sorted([os.path.split(f)[-1] for f in glob(data_dir + "*rep?.*", )])

# Read metadata of all sims in batch
df = pd.read_csv(os.path.join(data_dir, "metadata_full.csv"), index_col = 0)
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
