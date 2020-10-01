#!/usr/bin/env python
import sys
import os
import numpy as np
import pandas as pd
import numba
from multiprocessing import Pool

from tqdm import tqdm
import time
import datetime
from glob import glob

git_path = "/home/ubuntu/git/"
# git_path = "C:\\Users\\Pranav\\git\\"
vor_path = os.path.join(git_path, "active_vertex")
sys.path.append(vor_path)

import voronoi_model.voronoi_model_periodic as spv

######################

f = 200
t0 = 0
tmax = 2
dt = 0.025 
n_t = int((tmax - t0) * f / dt) + 1  # calculates the n_t to get the desired dt

# a = 0.4
k = 2
J = 0.0

n_c = 235
L_norm = 10

p_space = np.linspace(3.2, 4.0, 5)
v_space = np.linspace(2e-3, 3e-2, 3)
# p_space = np.array([3.8])
# v_space = np.array([2e-2])
d_space = np.linspace(1.0, 1.6, 4)
replicates = np.arange(1, dtype=int)

cores = 3

to_dir = os.path.join(
    git_path, "evomorph", "data", str(datetime.date.today()) + "_p0v0dens_phase_sims2/",
)

######################


def counter():
    i = 1
    while True:
        yield i
        i += 1


def simulate(
    params, print_thread=True, progress_bar=False, print_sim=False,
):
    p, v, d, rep = params
    prefix = f"p0{p:.2f}_v0{v:.2e}_dens{d:.2f}_rep{int(rep)}"

    vor = spv.Tissue()
    vor.make_init2(L=L_norm / np.sqrt(d), n_c=n_c)

    vor.set_GRN_t_span(t0, tmax, n_t, scaling_factor=f)
    vor.v0 = v

    # vor.n_warmup_steps = int(150 / dt)
    vor.n_warmup_steps = 1

    W = J * np.array([[1, 0], [0, 1]])
    vor.set_interaction(W=W, pE=0)
    vor.A0 = L_norm ** 2 / n_c
    vor.P0 = p * np.sqrt(vor.A0)

    vor.Dr = 0.01
    vor.kappa_A = 0.2
    vor.kappa_P = 0.1
    vor.a = vor.A0 / 2
    vor.k = 2

    start_time = time.time()
    vor.simulate(progress_bar=progress_bar, print_updates=print_sim)
    end_time = time.time()

    secs_elapsed = int(end_time) - int(start_time)
    mins_elapsed = secs_elapsed / 60
    it_per_sec = vor.n_t / secs_elapsed

    fname = os.path.abspath(os.path.join(to_dir, prefix))
    np.savez_compressed(fname, vor.x_save.astype(np.float32))

    if print_thread:
        print(f"Thread took {mins_elapsed:.2f} mins for run {count()}")

    return prefix, mins_elapsed, it_per_sec


#     return prefix, [vor_to_D_eff(vor, n) for n in (7, 19, 37)]


gen = counter()


def count():
    return next(gen)


######################

param_space = np.meshgrid(p_space, v_space, d_space, replicates,)
param_space = np.array(param_space).T.reshape(-1, 4)

common_metadata = dict(
    f=f, t0=t0, tmax=tmax, dt=dt, n_t=n_t, k=k, J=J, L_norm=L_norm, n_c=n_c
)
metadata = pd.DataFrame(
    dict(
        p0=param_space[:, 0],
        v0=param_space[:, 1],
        dens=param_space[:, 2],
        rep=param_space[:, 3],
    )
)
for k, v in common_metadata.items():
    metadata[k] = v

coords_fname = []
cols = metadata.columns
for row in metadata.values:
    row = list(row)
    row[3] = int(row[3])
    prefix = "{}{:.2f}_{}{:.2e}_{}{:.2f}_{}{}".format(
        *[j for i in zip(cols, row) for j in i]
    )
    coords_fname.append(prefix)
metadata["coords_fname"] = coords_fname

if not os.path.exists(to_dir):
    os.mkdir(to_dir)

metadata.to_csv(os.path.join(to_dir, "metadata.csv"))

#######################

print(
    f"Simulating {param_space.shape[0]} conditions "
    + f"on {cores} threads ({param_space.shape[0] / cores:.2f} per thread)"
)

if __name__ == "__main__":
    with Pool(cores) as p:
        results = list(p.imap_unordered(simulate, param_space))

# print(
#     f"Simulating {param_space.shape[0]} conditions "
#     + f"using for-looping"
# )

# if __name__ == "__main__":
#     results = []
#     for params in param_space[:4]:
#         results.append(simulate(params))

#######################

time_df = pd.DataFrame(
    dict(
        coords_fname=[res[0] for res in results],
        mins_elapsed=[res[1] for res in results],
        it_per_sec=[res[2] for res in results],
    )
)

metadata = metadata.merge(time_df)
metadata.to_csv(os.path.join(to_dir, "metadata_time.csv"))

print("COMPLETE!")