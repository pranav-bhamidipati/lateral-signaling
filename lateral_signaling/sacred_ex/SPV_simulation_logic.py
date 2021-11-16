import numpy as np
import synmorph as vm
import os

import matplotlib.pyplot as plt
# %matplotlib inline

import json

## Important: Work on a temporary directory, as you now handle storage over the provenance system
data_dir = "/mnt/c/Users/Pranav/tmp/work_dir"
os.makedirs(data_dir, exist_ok=True)


def do_one_simulation(
    seed,
    n_c,
    p0,
    Waa,
    Wbb,
    Wab,
    pE=0.5,
    v0=1.0e-3,
    Dr=1e-2,
    kappa_A=1.,
    kappa_P=1.,
    a=0.2,
    k=1.,
    dt=0.01,
    tfin=50,
    ex=None,
    save=False,
    save_frames=[],
):
    """Run an SPV simulation"""
    
    # Set random seed
    vor = vm.Tissue(seed)
    
    # n_c, pE = 0.5, c_types = None, nc_per_type = (), randomize = True, 
    
    # Set cell-cell adhesion
    W = np.array([
        [Waa, Wab],
        [Wab, Wbb],
    ])
    # vor.set_interaction(W=W, pE=pE)
    
    # Set other simulation parameters
    vor.P0 = p0
    vor.v0 = v0
    vor.Dr = Dr
    vor.kappa_A = kappa_A
    vor.kappa_P = kappa_P
    vor.a = a
    vor.k = k
    
#     # Initialize cell positions and cell types
#     vor.initialize(n_c=n_c)
    
    # Time-span of simulation (includes tfin)
    vor.set_t_span(dt = dt, tfin = tfin);
    
    # Run simulation
    x_t, tri_t = vor.simulate(
        n_c=n_c,
        W=W,
        progress="bar",
        # progress="print",
        # n_print=vor.n_t,
        print_complete=True,
    )
    

    if save:
        
        # Keep track of all saved objects for Sacred
        artifacts = []
        
        # Save animation of simulation
        vid_fname = f"simulation.mp4"
        vor.animate_cell_types(dir_name=data_dir, file_name=vid_fname, n_frames=75)
        artifacts.append(os.path.join(data_dir, vid_fname))
        
        # Save specific frames (time-points)
        for f in save_frames:
            
            # Plot frame
            vor.plot_cell_types(
                x=vor.x_t[f], 
                bleed=0.5,
                scatter_kwargs=dict(s=10),
            )
            
            # Save frame
            frame_fname = f"frame={f}.png"
            frame_path = os.path.join(data_dir, frame_fname)
            plt.savefig(frame_path, dpi=80, )
            plt.close()
            
            # Save filename for Sacred
            artifacts.append(frame_path)

        if ex is not None:
            
            # Save video and frame(s)
            for _a in artifacts:
                ex.add_artifact(_a)
            
            # Dump data to a JSON file
            skip = int(0.2 / dt)
            data_dump_fname = os.path.join(data_dir, "results.json")
            data_dump_dict = {
                "t": vor.t_span[::skip].tolist(),
                "x_t": vor.x_t[::skip].tolist(),
                "tri_t": vor.tri_t[::skip].tolist(),
            }
            
            # Save JSON to Sacred
            with open(data_dump_fname, "w") as f:
                json.dump(data_dump_dict, f, indent=4)
            ex.add_artifact(data_dump_fname)

            # Save any source code dependencies to Sacred
            source_files = [
                "synmorph.py",
                "physics.py",
                "geometry.py",
                "utils.py",
                "viz.py",
            ]
            for sf in source_files:
                ex.add_source_file(sf)


# ## Important: pass the sacred experiment as a parameter
# def do_one_parameter_config(m, gamma, k, t_min=0.0, t_max=25.0, y0=1.0, dydt0=0.0, SAVE=False, ex=None):
#     T = np.linspace(t_min, t_max, 101)
#     sol = solve_ivp(
#         RHS, t_span=(t_min, t_max), y0=(y0, dydt0), args=(m, gamma, k), t_eval=T
#     )
#     plt.figure()
#     plt.plot(T, sol["y"][0])
#     plt.xlabel(r"$t$", fontsize=14)
#     plt.ylabel(r"$x$", fontsize=14)
#     plt.title(f"$m={m:.2e},~\gamma={gamma:.2e},~k={k:.2e}$")
#     if SAVE:
#         fname = f"{DATA_DIR}/very-important-figure.png"  ## Important: We no longer need to handle file names ourself
#         plt.savefig(fname)
#         if ex is not None:
#             ex.add_artifact(fname)  ## Save figure
#             data_dump_fname = f"{DATA_DIR}/solution.json"
#             data_dump_dict = {
#                 "t": T.tolist(),
#                 "x": sol["y"][0].tolist(),
#                 "dxdt": sol["y"][1].tolist(),
#             }
#             with open(data_dump_fname, "w") as f:
#                 json.dump(data_dump_dict, f, indent=4)
#             ex.add_artifact(data_dump_fname)
#         plt.close()
#     else:
#         plt.show()
