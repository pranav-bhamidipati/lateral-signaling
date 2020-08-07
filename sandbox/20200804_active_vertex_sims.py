#!/usr/bin/env python
# coding: utf-8

import sys
import os

vor_path = "/home/ubuntu/git/active_vertex"
# vor_path = 'C:\\Users\\Pranav\\git\\active_vertex'
sys.path.append(vor_path)

from voronoi_model.voronoi_model_periodic import *
from lattice_oop import *
import numpy as np
import tqdm
import datetime

########################

v_vals = np.logspace(-2, 0, 11)
n_replicates = 5
progress_bar = True
print_updates = False

iterator = [(rep, v) for rep in range(n_replicates) for v in v_vals][:1]

if progress_bar:
    iterator = tqdm.tqdm(iterator)

for rep, v in iterator:
    vor = Tissue()
    vor.generate_cells(600)
    vor.make_init(10)
    vor.set_interaction(W=0.08 * np.array([[0, 1], [1, 0]]), pE=0)

    # vor.P0 = 3.00
    p0 = 3.80  # 3.81
    vor.A0 = 0.86
    vor.P0 = p0 * np.sqrt(vor.A0)

    vor.v0 = v
    vor.Dr = 0.01
    vor.kappa_A = 0.2
    vor.kappa_P = 0.1
    vor.a = 0.3
    vor.k = 2

    vor.set_t_span(0.02, 20)

    vor.simulate(print_updates=print_updates)

    vor.save_all(
        f"{datetime.date.today()}_active_vor_lattice_sims", f"vel{v:.2e}_{rep}"
    )
