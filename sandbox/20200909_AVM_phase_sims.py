import sys
import os
import numpy as np

from tqdm import tqdm
import numba
import datetime

# vor_path = "/home/ubuntu/git/active_vertex"
vor_path = 'C:\\Users\\Pranav\\git\\active_vertex'
sys.path.append(vor_path)

import voronoi_model.voronoi_model_periodic as avm

########################

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

def vor_to_D_eff(vor, n=7):
    return get_D_eff(
        vor.x_save[0],
        vor.x_save[-1],
        vor.L,
        vor.tfin,
        vor.v0,
        vor.Dr,
        n=n,
    )

########################

p_space = np.linspace(3.5, 3.9, 5)
v_space = np.arange(2e-3, 3e-2, 4e-3)
replicates = [0]

param_space = np.meshgrid(
    p_space,
    v_space,
    replicates
)
param_space = np.array(param_space).T.reshape(-1, 3)

########################

f = 200
t0 = 0
tmax = 4
dt = 0.05
n_t = int((tmax - t0) * f / dt) + 1  # calculates the n_t to get the desired dt

a = 0.4
k = 2
J = 0.

common_metadata = dict(f=f, t0=t0, tmax=tmax, dt=dt, n_t=n_t, a=a, k=k, J=J)

to_dir = "2020-08-19_avm_phase_sims/"

if not os.path.exists(to_dir):
    os.mkdir(to_dir)

def simulate(params):
    p, v, rep = params
    prefix = f"p0{p:.2f}_v0{v:.2e}_rep{rep}"
    
    vor2 = avm.Tissue()
    vor2.generate_cells(600)
    vor2.make_init(10)
    
    vor2.set_GRN_t_span(t0, tmax, n_t, scaling_factor=f);
    vor2.v0 = v / f
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
    
    vor2.simulate2(progress_bar=False, print_updates=False);
    
    fname = to_dir + prefix + ".npy"
    np.save(fname, vor2.x_save, allow_pickle=False)
    
    print(f"Thread {count()*100:.2f}% complete")

    return fname

from multiprocessing import Pool
if __name__ == '__main__':
    with Pool(cores) as p:
        results = list(p.imap_unordered(simulate, param_space[:2]))

print("COMPLETE!")
