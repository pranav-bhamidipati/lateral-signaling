import sys
import os

vor_path = "/home/ubuntu/git/active_vertex"
# vor_path = 'C:\\Users\\Pranav\\git\\active_vertex'
sys.path.append(vor_path)

import voronoi_model.voronoi_model_periodic as avm
from lattice_oop import *
import numpy as np
import tqdm
import datetime

########################

def pcounter(n):
    i = 1
    while True:
        yield i/n
        i += 1


p_space = np.linspace(3.2, 4.0, 9)
J_space = -np.linspace(0, 0.16, 5)
v_space = np.logspace(-2, 1, 7)[1:-1]

param_space = np.meshgrid(
    p_space, 
    J_space,
    v_space,
)
param_space = np.array(param_space).T.reshape(-1, 3)

gen = pcounter(param_space.size)
def count():
    return next(gen)

f = 200
t0 = 0
tmax = 4
dt = 0.05
n_t = int((tmax - t0) * f / dt) + 1  # calculates the n_t to get the desired dt

a = 0.4
k = 2

to_dir = "2020-08-19_avm_phase_sims/"

def simulate(params):
    p, J, v = params
    prefix = f"p0{p:.2f}_J{J:.2f}_v0{v:.2e}"
    
    vor2 = avm.Tissue()
    vor2.generate_cells(600)
    vor2.make_init(10)
    
    vor2.set_GRN_t_span(t0, tmax, n_t, scaling_factor=f);
    vor2.v0 = v / f
    vor2.n_warmup_steps = int(200 / dt)

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
    
    fname = to_dir + prefix
    
    print(f"{count()*100:.2f}% complete")

    return fname, vor2.x_save



from multiprocessing import Pool
if __name__ == '__main__':
    with Pool(8) as p:
        results = list(p.imap_unordered(simulate, param_space))


if not os.path.exists(to_dir):
    os.mkdir(to_dir)


print("Saving to", to_dir)

for fname, arr in results:
    np.save(fname, arr, allow_pickle=False)
