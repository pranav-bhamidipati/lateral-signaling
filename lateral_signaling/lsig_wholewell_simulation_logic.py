from uuid import uuid4
import os
from math import ceil
from copy import deepcopy
import numpy as np
import lateral_signaling as lsig
import matplotlib.pyplot as plt
import colorcet as cc
import h5py
# import json

# Use a unique directory name for this run 
uid = str(uuid4())

# Write to temporary (fast read/write) directory of choice
data_dir = os.path.abspath(f"/tmp/{uid}")    # Use root temp dir (Linux/MacOS)
# data_dir = f"/home/pbhamidi/scratch/lateral_signaling/tmp/{uid}"  # Use scratch dir on compute cluster

os.makedirs(data_dir, exist_ok=True)

def do_one_simulation(
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
    animate=False,
    n_frames=100,
    fps=15,
    save_skip=5,
    save_frames=(),
    fmt="png",
    dpi=300,
    **kwargs
):
    """Run a lateral signaling simulation"""

    # Set time parameters
    nt = int(nt_t * tmax_days) + 1
    t_days = np.linspace(0, tmax_days, nt)
    
    # Convert to dimensionless units for simulation
    t = t_days / lsig.t_to_units(1)
    dt = t[1] - t[0]
                                                                         
    # Make lattice                                                       
    X = lsig.hex_grid(rows, cols)                                        
                                                                         
    # Get # cells                                                        
    n = X.shape[0]                                                       
                                                                         
    # Get sender cell and center lattice on it                           
    sender_idx = lsig.get_center_cells(X)                                
    X = X - X[sender_idx]
    
    # Get cell-cell Adjacency                                                          
    Adj = lsig.get_weighted_Adj(X, r_int, sparse=True, row_stoch=True)  
                                                                         
    # Draw initial expression from a Half-Normal distribution with mean 
    #   `lambda` (basal expression)                                           
    S0 = np.abs(np.random.normal(
        size=n, scale=lambda_ * np.sqrt(np.pi/2)
    ))
                                                                         
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
    
    if save:
        
        artifacts = []
        
        # Calculate cell positions over time
        X_t = np.multiply.outer(1 / np.sqrt(rho_t), X)
        
        # Get default plot options
        plot_kw_S = deepcopy(lsig.plot_kwargs)
        plot_kw_S["sender_idx"] = sender_idx
        
        # Scalebar length
        sbar_value = 125  # um
        sbar_units  = "um"
        plot_kw_S["scalebar"] = True
        plot_kw_S["sbar_kwargs"]["fixed_value"] = sbar_value
        plot_kw_S["sbar_kwargs"]["fixed_units"] = sbar_units
        
        # Turn off scalebar text
        plot_kw_S["sbar_kwargs"]["font_properties"]["weight"] = 0
        
        # Axis limits
        plot_kw_S["xlim"] = X_t[-1, :, 0].min(), X_t[-1, :, 1].max()
        plot_kw_S["ylim"] = X_t[-1, :, 1].min(), X_t[-1, :, 1].max() 
        
        # Change default args for reporter
        plot_kw_R = deepcopy(plot_kw_S)
        plot_kw_R["cmap"] = cc.cm["kr"]
        plot_kw_R["cbar_kwargs"]["label"] = "mCherry (AU)"
        
        # Intensity range
        vmin_S = 0.
        vmax_S = float(S_t[:, tc_mask].max())
        plot_kw_S["vmin"] = vmin_S
        plot_kw_S["vmax"] = vmax_S
        plot_kw_S["cbar_kwargs"]["ticks"] = [vmin_S, vmax_S]
        plot_kw_S["cbar_kwargs"]["format"] = "%.1f"
        
        vmin_R = 0.
        vmax_R = float(R_t[:, tc_mask].max())
        plot_kw_R["vmin"] = vmin_R
        plot_kw_R["vmax"] = vmax_R
        plot_kw_R["cbar_kwargs"]["ticks"] = [vmin_R, vmax_R]
        plot_kw_R["cbar_kwargs"]["format"] = "%.1f"
        
        # Frame titles and filenames
        frame_title = lambda i: f"{t_days[i]:.2f} days"
        
        if animate:
            
            for i in range(2):
                
                # Select data
                E       = ("GFP", "mCherry")[i]
                var_t   = (S_t, R_t)[i]
                plot_kw = (plot_kw_S, plot_kw_R)[i]

                # Make figure
                fig, ax = plt.subplots(
                    figsize=(3.5, 2),
                )
            
                # Path for video
                fpath = os.path.join(data_dir, f"simulation_{E}.mp4")
                
                # Make video
                lsig.animate_hex_sheet(
                    fpath=fpath,
                    X_t=X_t,
                    var_t=var_t,
                    rho_t=rho_t,
                    fig=fig,
                    ax=ax,
                    n_frames=n_frames,
                    fps=fps,
                    dpi=dpi,
                    title_fun=frame_title,
                    plot_kwargs=plot_kw,
                )
                plt.close()
                
                # Add to Sacred artifacts
                artifacts.append(fpath)

        else:
            for i in range(2):
                
                # Select data
                E       = ("GFP", "mCherry")[i]
                var_t   = (S_t, R_t)[i]
                plot_kw = (plot_kw_S, plot_kw_R)[i]
                
                for f in save_frames:
                    
                    # Plot frame
                    fig, ax = plt.subplots(figsize=(3, 3))
                    lsig.plot_hex_sheet(
                        ax    = ax,
                        X     = X_t[f],
                        rho   = rho_t[f],
                        var   = var_t[f],
                        title = frame_title(f),
                        **plot_kw
                    )
                    plt.tight_layout()

                    # Save frame
                    fname = f"{E}_frame_{f}" + "." + fmt
                    fpath = os.path.join(data_dir, fname)
                    print("Saving:", fpath)
                    plt.savefig(fpath, dpi=dpi)
                    plt.close()

                    # Add to Sacred artifacts
                    artifacts.append(fpath)

        if ex is not None:
            
            # Dump data to file
            # data_dump_fname = os.path.join(data_dir, "results.json")
            data_dump_fname = os.path.join(data_dir, "results.hdf5")
            
            # Dump data to an HDF5 file
            with h5py.File(data_dump_fname, "w") as f:
                f.create_dataset("t", data=t[::save_skip]);
                f.create_dataset("X", data=X);
                f.create_dataset("rho_t", data=rho_t[::save_skip]);
                f.create_dataset("sender_idx", data=sender_idx);
                f.create_dataset("S_t", data=S_t[::save_skip]);
                f.create_dataset("R_t", data=R_t[::save_skip]);

            # # Dump data to a JSON file
            # data_dump_dict = {
                # "t": t.tolist(),
                # "X": X.tolist(),
                # "sender_idx": float(sender_idx),      
                # "rho_t": rho_t.tolist(),
                # "S_t": S_t.tolist(),  
                # "R_t": R_t.tolist(),  
            # }
            # with open(data_dump_fname, "w") as f:
                # print("Saving:", data_dump_fname)
                # json.dump(data_dump_dict, f, indent=4)
            
            # Add data dump to Sacred
            artifacts.append(data_dump_fname)
            
            # Add all artifacts to Sacred
            for _a in artifacts:
                ex.add_artifact(_a)

            # Save any source code dependencies to Sacred
            source_files = [
                "lateral_signaling.py",
            ]
            for sf in source_files:
                ex.add_source_file(sf)

