import numpy as np
from math import ceil
import lateral_signaling as lsig
import os

import matplotlib.pyplot as plt

import json

## Uses a temporary directory for data, since storage is handled by Sacred
data_dir = "/tmp/work_dir"
os.makedirs(data_dir, exist_ok=True)


def do_one_simulation(
    seed,
    tmax_days,
    nt_t,
    rows,
    cols,
    r_int,
    alpha,
    k,
    p,
    delta,
    lambda_,
    beta_args,
    gamma_R,
    g,
    rho_0,
    rho_max,
    delay,
    ex=None,
    save=False,
):
    """Run a lateral signaling simulation"""
    
    # Set random seed                                                    
    rng = np.random.default_rng(seed)                                    
                                                                         
    # Set time parameters
    tmax = tmax_days / lsig.t_to_units(1)                                
    nt = int(nt_t * tmax) + 1                                            
    t = np.linspace(0, tmax, nt)
    dt = t[1] - t[0]
                                                                         
    # Make lattice                                                       
    X = lsig.hex_grid(rows, cols)                                        
                                                                         
    # Get # cells                                                        
    n = X.shape[0]                                                       
                                                                         
    # Get sender cell and center lattice on it                           
    sender_idx = lsig.get_center_cells(X)                                
    X = X - X[sender_idx]
    
    # Get index of TC1 (closest TC to sender)
    x_bias = np.array([-1e-6, 0.])
    tc1_idx = np.argsort(np.linalg.norm(X + x_bias, axis=1))[1]
    
    # Get cell-cell Adjacency                                                          
    Adj = lsig.get_weighted_Adj(X, r_int, sparse=True, row_stoch=True)  
                                                                         
    # Draw initial expression from a Half-Normal distribution with mean 
    #   `lambda` (basal expression)                                           
    S0 = np.abs(rng.normal(size=n, scale=lambda_ * np.sqrt(np.pi/2)))
                                                                         
    # Fix sender cell(s) to constant expression                          
    S0[sender_idx] = 1                                            

    # Package into args for lsig.reporter_rhs
    R_args = [S0, gamma_R, sender_idx]
    
    # Initial R expression 
    R0 = np.zeros(n, dtype=np.float32)

    # Make a mask for transceivers
    tc_mask = np.ones(n, dtype=bool)
    tc_mask[sender_idx] = False
    
    # Package args for DDe integrator
    S_args = (
        Adj, 
        sender_idx, 
        lsig.beta_rho_exp, 
        beta_args, 
        alpha, 
        k, 
        p, 
        delta, 
        lambda_, 
        g, 
        rho_0
    )
    where_rho = len(S_args) - 1
    
    # Calculate density
    rho_t = lsig.logistic(t, g, rho_0, rho_max)
    
    # Simulate
    S_t = lsig.integrate_DDE_varargs(
        t,
        rhs=lsig.signal_rhs,
        var_vals=[rho_t],
        where_vars=where_rho,
        dde_args=S_args,
        E0=S0,
        delay=delay,
        varargs_type="list",
    )
    
    # Mean fluorescence
    S_t_tcmean = S_t[:, tc_mask].mean(axis=1)
    
    # Number of activated TCs over threshold
    thresh = k
    S_t_actnum = (S_t[:, tc_mask] > thresh).sum(axis=1)
    
    # # Area of activated TCs
    # S_t_actarea = lsig.ncells_to_area(S_t_actnum, rho_t)
    
    # Intial velocity (dS/dt of the 1st TC at time t=delay)
    v_init = (
        S_t[(step_delay + 1), tc1_idx] - S_t[step_delay, tc1_idx]
    ) / dt
    
    # Number of activated TCs at end of simulation
    n_act_fin = S_t_actnum[-1]
    
    # Make version of S with delay
    step_delay = ceil(delay / dt) 
    S_t_delay = S_t[np.maximum(np.arange(nt) - step_delay, 0)]

    # Package args for DDe integrator
    R_args = (
        Adj, 
        sender_idx, 
        lsig.beta_rho_exp, 
        beta_args, 
        alpha, 
        k, 
        p, 
        delta, 
        lambda_, 
        g, 
        rho_0,
        S_t_delay[0],
        gamma_R,
    )
    where_S_delay = where_rho + 1
    
    # Simulate reporter expression
    R_t = lsig.integrate_DDE_varargs(
        t,
        lsig.reporter_rhs,
        var_vals=[rho_t, S_t_delay],
        where_vars=[where_rho, where_S_delay],
        dde_args=R_args,
        E0=R0,
        delay=delay,
        varargs_type="list",
    )

    # Mean fluorescence
    R_t_tcmean = R_t[:, tc_mask].mean(axis=1)
    
    # Number of activated TCs
    R_t_actnum = (R_t[:, tc_mask] > thresh).sum(axis=1)
    
    # # Area of activated TCs
    # R_t_actarea = lsig.ncells_to_area(R_t_actnum, rho_t)
    
    if save:
        
        if ex is not None:
            
            # Dump data to a JSON file
            data_dump_fname = os.path.join(data_dir, "results.json")
            data_dump_dict = {
                    "t": t.tolist(),               
                    "sender_idx": float(sender_idx),      
                    "rho_t": rho_t.tolist(),
                    "S_t_tcmean": S_t_tcmean.tolist(),  
                    "S_t_actnum": S_t_actnum.astype(np.float32).tolist(),  
                    "R_t_tcmean": R_t_tcmean.tolist(),  
                    "R_t_actnum": R_t_actnum.astype(np.float32).tolist(),
                    "v_init"    : float(v_init),
                    "n_act_fin" : float(n_act_fin),
            }
            
            # Save JSON to Sacred
            with open(data_dump_fname, "w") as f:
                json.dump(data_dump_dict, f, indent=4)
            ex.add_artifact(data_dump_fname)

            # Save any source code dependencies to Sacred
            source_files = [
                "lateral_signaling.py",
            ]
            for sf in source_files:
                ex.add_source_file(sf)

