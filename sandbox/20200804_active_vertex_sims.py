#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('/home/ubuntu/git/active_voronoi')
# sys.path.append('C:\\Users\\Pranav\\git\\active_vertex')

from voronoi_model.voronoi_model_periodic import *
from lattice_oop import *
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import datetime

########################

v_vals = np.logspace(-2, 0, 9)
n_replicates = 3
progress_bar = True


iterator = [(rep, v) for rep in range(n_replicates) for v in v_vals][:1]

if progress_bar:
    iterator = tqdm.tqdm(iterator)

for rep, v in iterator:
    vor = Tissue()
    vor.generate_cells(600)
    vor.make_init(20)
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

    vor.set_t_span(0.02, 50)

    vor.simulate()
    
    vor = ActiveVoronoi(vor)
    vor.all_to_csv(prefix=f"v{v:.2e}_rep{rep}", to_dir=f"{datetime.today()}_active_vor_lattices", )





