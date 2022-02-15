import os
import json
import numpy as np

# File containing queue of arguments for array job
q_fname = "lsig_phase_queue.json"

# Number of replicates of each param set
n_reps = 2

# Parameter values to scan
n_reps        = 5                            # Number of replicates per condition
g_space       = np.linspace(0, 2.4, 25)[1:]  # Proliferation rate
rho_0_space   = np.linspace(0,   6, 25)[1:]  # Initial density
rho_max_space = np.linspace(0,   6, 25)[1:]  # Carrying capacity
    
# Make matrix of all combinations of params
param_space = np.meshgrid(
    rho_0_space, 
    rho_max_space,
)
param_space = np.array(param_space).T.reshape(-1, len(param_space))

# JSON cannot save np arrays, but it can save lists
params = [
    dict(
        g_space = g_space.tolist(),
        rho_0   = p[0],
        rho_max = p[1],
    )
    for p in param_space
]

with open(q_fname, "w") as f:
    json.dump(params, f, indent=4)
    print(
        f"Made queue of length {len(params)} at:\n\t{q_fname}"
    )
    print()


