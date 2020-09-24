import sys
import os
import numpy as np
import pandas as pd

# from tqdm import tqdm
# import numba
# import datetime

# vor_path = "/home/ubuntu/git/active_vertex"
vor_path = 'C:\\Users\\Pranav\\git\\active_vertex'
# vor_path = "/home/pbhamidi/git/active_vertex"

sys.path.append(vor_path)

import voronoi_model.voronoi_model_periodic as avm

########################

# Inputs
# to_dir = "/home/pbhamidi/data/2020-09-17_avm_phase_sims_dense/"
to_dir = "C:\\Users\\Pranav\\git\\evomorph\\scratch"

########################

p_space = np.linspace(3.2, 4.0, 17)
v_space = np.array([5e-4, 1e-3, *np.linspace(2e-3, 3e-2, 15)])
replicates = np.arange(5, dtype=int)

param_space = np.meshgrid(
    p_space,
    v_space,
    replicates
)
param_space = np.array(param_space).T.reshape(-1, 3)

param_space = param_space[::1000]

########################

f = 200
t0 = 0
tmax = 2
dt = 0.025
n_t = int((tmax - t0) * f / dt) + 1  # calculates the n_t to get the desired dt

# a = 0.4
k = 2
J = 0.

common_metadata = dict(f=f, t0=t0, tmax=tmax, dt=dt, n_t=n_t, k=k, J=J)

if not os.path.exists(to_dir):
    os.mkdir(to_dir)

cores = 2
# gen = pcounter((param_space.shape[0] // cores) + 1)

# def count():
#     return next(gen)

def simulate(params, progress_bar=False, print_updates=False):
    p, v, rep = params
    rep = int(rep)
    prefix = f"p0{p:.2f}_v0{v:.2e}_rep{rep}"

    vor2 = avm.Tissue()
    vor2.make_init2(L=10, n_c=400)

    vor2.set_GRN_t_span(t0, tmax, n_t, scaling_factor=f);
    vor2.v0 = v
    vor2.n_warmup_steps = int(150 / dt)

    W = J * np.array([[1, 0], [0, 1]])
    vor2.set_interaction(W=W, pE=0)
    vor2.A0 = vor2.L**2 / vor2.n_c
    vor2.P0 = p * np.sqrt(vor2.A0)
    
    vor2.Dr = 0.01
    vor2.kappa_A = 0.2
    vor2.kappa_P = 0.1
    vor2.a = vor2.A0 / 2
    vor2.k = k
    
    vor2.simulate(progress_bar=progress_bar, print_updates=print_updates);
    
    fname = os.path.abspath(os.path.join(to_dir, prefix))
    np.save(fname, vor2.x_save, allow_pickle=False)
    
#     print(f"Thread {count()*100:.2f}% complete")
    return prefix
#     return prefix, [vor_to_D_eff(vor2, n) for n in (7, 19, 37)]

import dask
from dask.distributed import Client, progress
client = Client(threads_per_worker=1, n_workers=2)
if __name__ == '__main__':
    futures = []
    for params in param_space:
        future = client.submit(simulate, params)
        futures.append(future)
    results = client.gather(futures)

#     lazy_result = []
#     n_slurm_tasks = int(os.environ['SLURM_NTASKS'])
# #     n_slurm_tasks =2 
#     client = Client(threads_per_worker=1, n_workers=n_slurm_tasks)
#     for params in param_space:
#         lazy_result.append(dask.delayed(simulate)(params))
#     dask.compute(*lazy_result)

# from multiprocessing import Pool
# if __name__ == '__main__':
#     with Pool(cores) as p:
#         results = list(p.imap_unordered(simulate, param_space))

# results = []
# for params in param_space:
#     results.append(simulate(params))

metadata = pd.DataFrame(dict(
    p0=param_space[:, 0], 
    v0 = param_space[:, 1], 
    rep = param_space[:, 2]
))
for k, v in common_metadata.items():
    metadata[k] = v

coords_fname = []
for _, row in enumerate(metadata[["p0", "v0", "rep"]].values):
    p, v, rep = row
    coords_fname.append(f"p0{p:.2f}_v0{v:.2e}_rep{rep}.npy")
metadata["coords_fname"] = coords_fname

metadata.to_csv(os.path.join(to_dir, "metadata.csv"))

# print("COMPLETE!")
