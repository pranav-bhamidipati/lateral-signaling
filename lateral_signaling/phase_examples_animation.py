import os
from glob import glob
import json
from copy import deepcopy
import h5py

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation

import lateral_signaling as lsig

example_fpath = os.path.abspath("../data/sim_data/phase_examples.json")
thresh_fpath  = os.path.abspath("../data/sim_data/phase_threshold.json")
data_dir      = os.path.abspath("../data/sim_data/20211209_phase_examples/sacred")

save_dir      = os.path.abspath("./plots")
fpath         = os.path.join(save_dir, f"phase_examples_.mp4")
dpi           = 300

def main(
    fpath=fpath,
    sacred_runs=(301, 302, 303),
    save=False,
    vmins=None,
    vmaxs=None,
    writer="ffmpeg",
    n_frames=100,
    print_frame=False,
    fps=20,
    dpi=300,
):

    # Read in data from experiments
    run_dirs = [os.path.abspath(f"./sacred/{r}/") for r in sacred_runs]
    
    # Initialize data to read from files
    names      = []
    t          = []
    X          = []
    rho_t_run  = []
    sender_idx = []
    tc_mask    = []
    S_t_run    = []
    thresh     = []
    A_t_run    = []
    
    # 
    with open(example_fpath, "r") as e:
        example_params = json.load(e)
        names = example_params["name"]
    
    # Get data from run directories
    for i, rd in enumerate(run_dirs):
        
        assert os.path.exists(rd)), f"Run directory does not exist: {rd}"
        _config_file = os.path.join(rd, "config.json")
        _results_file = os.path.join(rd, "results.hdf5")

        assert os.path.exists(_config file)),  f"Sacred config file does not exist: {_config_file}"
        assert os.path.exists(_results_file)), f"Sacred restults file does not exist: {_results_file}" 

        # Get conditions that are the same across runs
        if i == 0:
            with open(_config_file, "r") as c:
                config = json.load(c)

                # Coordinates of cell sheet
                rows       = config["rows"]
                cols       = config["cols"]
                X          = lsig.hex_grid(rows, cols)

                # Center lattice on sender
                X = X - X[sender_idx]

                # Number of cells
                n = X.shape[0]

                # Index of sender cell
                sender_idx = lsig.get_center_cells(X)

                # Promoter threshold
                thresh = config["k"]
                
                # Get mask of transceivers (no senders)
                tc_mask = np.ones(n, dtype=bool)
                tc_mask[sender_idx] = False

        # Get remaining info from run's data dump
        with h5py.File(_results_file, "r") as f:
            
            if i == 0:
                # Time-points
                t = np.asarray(f["t"])
            
            # Density vs. time
            rho_t = np.asarray(f["rho_t"])

            # Ligand expression vs. time
            S_t = np.asarray(f["S_t"])
        
        # Compute area of activated cells vs. time
        actnum_t = ((S_t[:, tc_mask]) > thresh).sum(axis=-1)
        A_t = lsig.ncells_to_area(actnum_t, rho_t)
        
        # Compute cell positions vs. time
        X_t = np.multiply.outer(1 / np.sqrt(rho_t), X)
        
        # Store data that differs between samples
        rho_t_run.append(rho_t)
        S_t_run.append(S_t)
        A_t_run.append(A_t)
        X_t_run.append(X_t)
    
    # Make data into arrays
    rho_t_run = np.asarray(rho_t_run)
    S_t_run   = np.asarray(S_t_run)
    A_t_run   = np.asarray(A_t_run)
    X_t_run   = np.asarray(X_t_run)
    
    
    if save:
        
        # Get which frames to animate
        frames = lsig.vround(np.linspace(0, nt-1, n_frames))
        
        # Make figure
        fig, axs = plt.subplots(
            nrows=2, 
            ncols=3, 
            figsize=(8.5, 6.0),
            gridspec_kw=dict(width_ratios=[1, 1, 1.2]),
        )
        
        # Get default kwargs for plotting
        plot_kwargs = deepcopy(lsig.plot_kwargs)
        plot_kwargs["sender_idx"] = sender_idx
        
        # Get axis limits
        _xmin = X_t_run[:, -1, :, 0].min(axis=1).max()
        _ymin = X_t_run[:, -1, :, 1].min(axis=1).max()
        plot_kwargs["xlim"] = _xmin, -_xmin
        plot_kwargs["ylim"] = _ymin, -_ymin

        # Set some args for colorbar
        plot_kwargs["cmap"] = lsig.kgy
        plot_kwargs["cbar_aspect"] = 8
        plot_kwargs["cbar_kwargs"]["shrink"] = 0.7
        plot_kwargs["extend"] = "neither"

        # Get colorscale limits
        ns_mask = np.ones(n, dtype=bool)
        ns_mask[sender_idx] = False
        plot_kwargs["vmin"] = tcmean_ts[:, :, ns_mask].min()
        plot_kwargs["vmax"] = tcmean_ts[:, :, ns_mask].max()

        # Add colorbar to end of first row
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(
                norm=mpl.colors.Normalize(plot_kwargs["vmin"], plot_kwargs["vmax"]), 
                cmap=plot_kwargs["cmap"]), 
            ax=axs[0][-1],
            aspect=plot_kwargs["cbar_aspect"],
            extend=_extend,
            **plot_kwargs["cbar_kwargs"]
        )

        # Turn off further colorbar plotting during animation
        plot_kwargs["colorbar"] = False
        
        # Initialize params that change over time
        var_kw = dict(
            X        = X_t_run[0, 0],
            var      = S_t_run[0, 0],
            rho      = rho_t_run[0, 0],
            t_upto   = t[:1],
            A_upto   = A_t_run[0, :1],
        )
        
        # Update which data is used for each run, in each frame
        def update_var_kw(f, run):
            var_kw.update(
                X        = X_t_run[run, frames[f]],
                var      = S_t_run[run, frames[f]],
                rho      = rho_t_run[run, frames[f]],
                t_upto   = t[:frames[f]],
                A_upto   = A_t_run[run, :frames[f]]
            )

        # Get params that don't change over time
        hex_static_kw = {
            k: plot_kwargs[k] 
            for k in plot_kwargs.keys() - var_kw.keys()
        }
        scatter_static_kw = {}

        # Plot one frame of animation
        def make_frame(f):
            
            # Set title at top of figure
            plt.suptitle(f"Time: {t[frames[f]]:.2f} days",)

            # Make a 2 x 3 layout of plots
            for idx, ax in enumerate(axs.flat):
                
                row = idx // 3
                col = idx % 3
                
                # Update plotting params
                update_var_kw(f, col)
                
                # Clear axis
                ax.clear()

                # Plot cell sheet
                if row == 0:
                    lsig.plot_hex_sheet(
                        ax=ax,
                        title=titles[col],
                        **var_kw,
                        **hex_static_kw,
                    )

                # Plot area
                elif row == 1:
                    
                    if f == 0:
                        ax.set(**scatter_static_kw)
                    
                    # Get time-series up to this point
                    _t = var_kw["t_upto"]
                    _A = var_kw["A_upto"]
                    
                    # Plot time-series and highlight current time
                    ax.scatter(_t, _A)
                    ax.scatter(_t[-1], _A[-1], alpha=0.3, c="k")


        try:
            _writer = animation.writers[writer](fps=fps, bitrate=1800)
        except RuntimeError:
            print("""
            The `ffmpeg` writer must be installed inside the runtime environment.
            Writer availability can be checked in the current enviornment by executing 
            `matplotlib.animation.writers.list()` in Python. Install location can be
            checked by running `which ffmpeg` on a command line/terminal.
            """)

        _anim_FA = animation.FuncAnimation(fig, make_frame, frames=n_frames, interval=200)

        # Get path and print to output
        _fpath = str(fpath)
        if not _fpath.endswith(".mp4"):
            _fpath += ".mp4"
        print("Writing to:", _fpath)

        # Save animation
        _anim_FA.save(_fpath, writer=_writer, dpi=dpi)


main(
    save=True, 
    n_frames=10, 
    print_frame=True,
)



#         def _anim(i):
#             """Creates one frame of animation."""

#             # Make each plot in layout
#             for r, ax in enumerate(axs):

#                 # Get changing arguments
#                 var_kw = dict(
#                     X     = X_func(frames[i]),
#                     var   = var_func(r, frames[i]),
#                     rho   = rho_func(frames[i]),
#                     title = title_fun(r),
#                     vmin  = vmins[r],
#                     vmax  = vmaxs[r],
#                 )

#                 # Plot frame of animation
#                 ax.clear()
#                 lsig.plot_hex_sheet(ax=ax, **var_kw, **static_kw)

#             # Make colorbars if they don't exist
#             if not colorbars:
#                 for r, ax in enumerate(axs):
                    
#                     _vmin = vmins[r]
#                     _vmax = vmaxs[r]
                    
#                     _cbar_kwargs = dict(
#                         ticks=[_vmin, _vmax],
#                         format="%.1f",
#                         label=("", "GFP (AU)")[r == (len(axs) - 1)],
#                     )

#                     cbar = plt.colorbar(
#                         plt.cm.ScalarMappable(
#                             norm=mpl.colors.Normalize(_vmin, _vmax), 
#                             cmap=plot_kwargs["cmap"]), 
#                         ax=ax,
#                         aspect=plot_kwargs["cbar_aspect"],
#                         extend=plot_kwargs["extend"],
#                         **_cbar_kwargs
#                     )
#                     colorbars.append(cbar)

#             # Change suptitle
#             suptitle = suptitle_fun(frames[i])
#             plt.suptitle(suptitle);
            
#             if i == 0:
#                 plt.tight_layout()
            
#             if print_frame:
#                 print(
#                     "Frame: ", 
#                     str(i + 1).rjust(len(str(n_frames))), 
#                     " / ", 
#                     str(n_frames), 
#                     sep=""
#                 )

#         try:
#             _writer = animation.writers[writer](fps=fps, bitrate=1800)
#         except RuntimeError:
#             print("""
#             The `ffmpeg` writer must be installed inside the runtime environment.
#             Writer availability can be checked in the current enviornment by executing 
#             `matplotlib.animation.writers.list()` in Python. Install location can be
#             checked by running `which ffmpeg` on a command line/terminal.
#             """)
        
#         _anim_FA = animation.FuncAnimation(fig, _anim, frames=n_frames, interval=200)
        
#         # Clean up path
#         _fpath = str(fpath)
#         if not _fpath.endswith(".mp4"):
#             _fpath += ".mp4"
#         print("Writing to:", _fpath)
        
#         # Make animation
#         _anim_FA.save(_fpath, writer=_writer, dpi=dpi)


# 0/0


# # Path to save animation
# save_dir = os.path.abspath("../plots")
# fpath = os.path.join(save_dir, f"phase_examples_.mp4")

# def main(
#     fpath=fpath,
#     save=False,
#     vmins=None,
#     vmaxs=None,
#     writer="ffmpeg",
#     n_frames=100,
#     print_frame=False,
#     fps=20,
#     dpi=300,
# ):

#     # Read in data from experiments
#     run_dirs = [f"../data/sim_data/20211209_phase_2D/sacred/{r}/" for r in sacred_runs]

#     deltas    = []
#     t         = []
#     S_t_res   = []
#     R_t_res   = []

#     for i, d in enumerate(run_dirs):
        
#         cfg_f = os.path.join(d, "config.json")
#         res_f = os.path.join(d, "results.hdf5")
        
#         # Extract time and expression
#         with h5py.File(res_f, "r") as f:
#             if i == 0:
#                 t = np.asarray(f["t"])
#                 nt = t.size
            
#             S_t_res.append(np.asarray(f["S_t"]))
#             R_t_res.append(np.asarray(f["R_t"]))

#         # Extract parameter value(s)
#         with open(cfg_f, "r") as f:

#             _config  = json.load(f)
            
#             if i == 0:

#                 g          = _config["g"]
#                 rho_0      = _config["rho_0"]
#                 rho_max    = _config["rho_max"]
#                 rho_t      = lsig.logistic(t, g, rho_0, rho_max)

#                 rows       = _config["rows"]
#                 cols       = _config["cols"]
#                 X          = lsig.hex_grid(rows, cols)
#                 sender_idx = lsig.get_center_cells(X)
#                 X          = X - X[sender_idx]
#                 n          = X.shape[0]
#                 X_t        = np.multiply.outer(1 / np.sqrt(rho_t), X)
            
#             if _config["delta"] is dict:
#                 _delta = _config["delta"]["value"]
#             else:
#                 _delta = _config["delta"]

#             deltas.append(_delta)

#     if save:
        
#         # Make figure
#         fig, axs = plt.subplots(
#             nrows=1, 
#             ncols=3, 
#             figsize=(8.5, 2.5),
# #            gridspec_kw=dict(width_ratios=[1, 1, 1.2]),
#         )

#         X_func       = lambda    i: X_t[i]
#         var_func     = lambda r, i: S_t_res[r][i]
#         rho_func     = lambda    i: rho_t[i]
#         suptitle_fun = lambda    i: f"Time: {t[i]:.2f} days"
#         title_fun    = lambda    r: ("No", "Weak", "Strong")[r] + " inhibition"
            
#         # Get default kwargs for plotting
#         plot_kwargs = deepcopy(lsig.plot_kwargs)
#         plot_kwargs["sender_idx"] = sender_idx
        
#         # Get axis limits
#         plot_kwargs["xlim"] = X_t[-1, :, 0].min(), X_t[-1, :, 0].max()
#         plot_kwargs["ylim"] = X_t[-1, :, 1].min(), X_t[-1, :, 1].max()

#         # Set some args for colorbar
#         plot_kwargs["cmap"] = lsig.kgy
#         plot_kwargs["cbar_aspect"] = 8
#         plot_kwargs["cbar_kwargs"]["shrink"] = 0.7
#         plot_kwargs["extend"] = "neither"
                    
#         # Get colorscale limits
#         ns_mask = np.ones(n, dtype=bool)
#         ns_mask[sender_idx] = False
#         if vmins is None:
#             vmins = [v[:, ns_mask].min() for v in S_t_res]
#         if vmaxs is None:
#             vmaxs = [v[:, ns_mask].max() for v in S_t_res]
        
#         # Make list to hold colorbars for easy reference
#         colorbars = []

#         # Turn off automatic colorbar plotting
#         plot_kwargs["colorbar"] = False
           
#         # Get frames to animate
#         frames = lsig.vround(np.linspace(0, nt-1, n_frames))
            
#         # Get params that don't change over time
#         static_kw = {
#             k: plot_kwargs[k] 
#             for k in plot_kwargs.keys() - set(
#                 ["X", "var", "rho", "title", "vmin", "vmax", "cbar_kwargs"]
#              )
#         }

#         def _anim(i):
#             """Creates one frame of animation."""

#             # Make each plot in layout
#             for r, ax in enumerate(axs):

#                 # Get changing arguments
#                 var_kw = dict(
#                     X     = X_func(frames[i]),
#                     var   = var_func(r, frames[i]),
#                     rho   = rho_func(frames[i]),
#                     title = title_fun(r),
#                     vmin  = vmins[r],
#                     vmax  = vmaxs[r],
#                 )
            
#                 # Plot frame of animation
#                 ax.clear()
#                 lsig.plot_hex_sheet(ax=ax, **var_kw, **static_kw)
                
#             # Make colorbars if they don't exist
#             if not colorbars:
#                 for r, ax in enumerate(axs):
                    
#                     _vmin = vmins[r]
#                     _vmax = vmaxs[r]
                    
#                     _cbar_kwargs = dict(
#                         ticks=[_vmin, _vmax],
#                         format="%.1f",
#                         label=("", "GFP (AU)")[r == (len(axs) - 1)],
#                     )

#                     cbar = plt.colorbar(
#                         plt.cm.ScalarMappable(
#                             norm=mpl.colors.Normalize(_vmin, _vmax), 
#                             cmap=plot_kwargs["cmap"]), 
#                         ax=ax,
#                         aspect=plot_kwargs["cbar_aspect"],
#                         extend=plot_kwargs["extend"],
#                         **_cbar_kwargs
#                     )
#                     colorbars.append(cbar)

#             # Change suptitle
#             suptitle = suptitle_fun(frames[i])
#             plt.suptitle(suptitle);
            
#             if i == 0:
#                 plt.tight_layout()
            
#             if print_frame:
#                 print(
#                     "Frame: ", 
#                     str(i + 1).rjust(len(str(n_frames))), 
#                     " / ", 
#                     str(n_frames), 
#                     sep=""
#                 )

#         try:
#             _writer = animation.writers[writer](fps=fps, bitrate=1800)
#         except RuntimeError:
#             print("""
#             The `ffmpeg` writer must be installed inside the runtime environment.
#             Writer availability can be checked in the current enviornment by executing 
#             `matplotlib.animation.writers.list()` in Python. Install location can be
#             checked by running `which ffmpeg` on a command line/terminal.
#             """)
        
#         _anim_FA = animation.FuncAnimation(fig, _anim, frames=n_frames, interval=200)
        
#         # Clean up path
#         _fpath = str(fpath)
#         if not _fpath.endswith(".mp4"):
#             _fpath += ".mp4"
#         print("Writing to:", _fpath)
        
#         # Make animation
#         _anim_FA.save(_fpath, writer=_writer, dpi=dpi)

# #        # Make video
# #        lsig.animate_hex_sheet(
# #            fpath=layout_vid_path,
# #            X_t=X_rho_arr[-1],
# #            var_t=var_rho_t[-1],
# #            rho_t=rhos[-1],
# #            fig=fig,
# #            ax=cbar_ax,
# #            anim=anim_func,
# #            n_frames=100,
# #            fps=15,
# #            dpi=dpi,
# #            title_fun=suptitle_fun,
# #            plot_kwargs=anim_plot_kwargs,
# #            _X_func   = lambda i: X_rho_arr,
# #            _var_func = lambda i: var_rho_t[:, i],
# #            _rho_func = lambda i: rhos,
# #        )
# #

