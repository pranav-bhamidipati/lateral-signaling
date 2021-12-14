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

save_dir      = os.path.abspath("../plots")
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
    t_days     = []
    X          = []
    rho_t_run  = []
    sender_idx = []
    ns_mask    = []
    S_t_run    = []
    thresh     = []
    X_t_run    = []
    A_t_run    = []
    R_t_run    = []
    
    # 
    with open(example_fpath, "r") as e:
        example_params = json.load(e)
        example_names = example_params["name"]
    
    # Get data from run directories
    for i, rd in enumerate(run_dirs):
        
        assert os.path.exists(rd), f"Run directory does not exist: {rd}"
        _config_file = os.path.join(rd, "config.json")
        _results_file = os.path.join(rd, "results.hdf5")

        assert os.path.exists(_config_file),  f"Sacred config file does not exist: {_config_file}"
        assert os.path.exists(_results_file), f"Sacred restults file does not exist: {_results_file}" 

        # Get conditions that are the same across runs
        if i == 0:
            with open(_config_file, "r") as c:
                config = json.load(c)

                # Coordinates of cell sheet
                rows       = config["rows"]
                cols       = config["cols"]
                X          = lsig.hex_grid(rows, cols)

                # Index of sender cell
                sender_idx = lsig.get_center_cells(X)

                # Center lattice on sender
                X = X - X[sender_idx]

                # Number of cells
                n = X.shape[0]

                # Promoter threshold
                thresh = config["k"]
                
                # Get mask of non-sender cells (transceivers)
                ns_mask = np.ones(n, dtype=bool)
                ns_mask[sender_idx] = False

        # Get remaining info from run's data dump
        with h5py.File(_results_file, "r") as f:
            
            # Time-points
            if i == 0:
                t = np.asarray(f["t"])
                t_days = lsig.t_to_units(t)
            
            # Density vs. time
            rho_t = np.asarray(f["rho_t"])

            # Ligand expression vs. time
            S_t = np.asarray(f["S_t"])
        
        # Compute area of activated cells (mm^2) vs. time
        actnum_t = ((S_t[:, ns_mask]) > thresh).sum(axis=-1)
        A_t = lsig.ncells_to_area(actnum_t, rho_t)
        R_t = np.sqrt(A_t / np.pi)

        # Compute cell positions vs. time
        X_t = np.multiply.outer(1 / np.sqrt(rho_t), X)
        
        # Store data that differs between samples
        rho_t_run.append(rho_t)
        S_t_run.append(S_t)
        A_t_run.append(A_t)
        R_t_run.append(R_t)
        X_t_run.append(X_t)
    
    # Make data into arrays
    rho_t_run = np.asarray(rho_t_run)
    S_t_run   = np.asarray(S_t_run)
    A_t_run   = np.asarray(A_t_run)
    R_t_run   = np.asarray(R_t_run)
    X_t_run   = np.asarray(X_t_run)
    
    if save:
        
        # Get which frames to animate
        nt = t.size
        frames = lsig.vround(np.linspace(0, nt-1, n_frames))
        
        # Font sizes
        SMALL_SIZE  = 12
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 16

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        # Make figure
        fig, axs = plt.subplots(
            nrows=2, 
            ncols=3, 
            figsize=(8.5, 5.0),
            gridspec_kw=dict(
                width_ratios=[1, 1, 1.2],
                height_ratios=[1, 0.8],
            ),
        )
        
        # Get default kwargs for plotting
        plot_kwargs = deepcopy(lsig.plot_kwargs)
        plot_kwargs["sender_idx"] = sender_idx
        
        # Get axis limits
        _xmin = X_t_run[:, -1, :, 0].min(axis=1).max()
        _ymin = X_t_run[:, -1, :, 1].min(axis=1).max()
        plot_kwargs["xlim"] = _xmin, -_xmin
        plot_kwargs["ylim"] = _ymin, -_ymin

        # Get colorscale limits
        plot_kwargs["vmin"] = 0
        plot_kwargs["vmax"] = S_t_run[:, :, ns_mask].max()

        # Set some args for colorbar
        plot_kwargs["cmap"] = lsig.kgy
        plot_kwargs["cbar_aspect"] = 8
        plot_kwargs["cbar_kwargs"] = dict(
            shrink = 0.7,
            ticks=[plot_kwargs["vmin"], plot_kwargs["vmax"]], 
            label="GFP (AU)",
            format="%.2f"
        )

        # Add colorbar to end of first row
        cbar1 = plt.colorbar(
            plt.cm.ScalarMappable(
                norm=mpl.colors.Normalize(plot_kwargs["vmin"], plot_kwargs["vmax"]), 
                cmap=plot_kwargs["cmap"]), 
            ax=axs[0][-1],
            aspect=plot_kwargs["cbar_aspect"],
            extend=plot_kwargs["extend"],
            **plot_kwargs["cbar_kwargs"]
        )
        
        # Make an invisible colorbar in row 2 for better spacing
        cbar2 = plt.colorbar(
            plt.cm.ScalarMappable(
                norm=mpl.colors.Normalize(plot_kwargs["vmin"], plot_kwargs["vmax"]), 
                cmap=plot_kwargs["cmap"]), 
            ax=axs[1][-1],
            aspect=plot_kwargs["cbar_aspect"],
            extend=plot_kwargs["extend"],
            ticks=[], 
            label="", 
            alpha=0, 
            shrink=1e-5,
        )

        # Turn off further colorbar plotting during animation
        plot_kwargs["colorbar"] = False
        
        # Initialize params that change over time
        var_kw = dict(
            X   = X_t_run[0, 0],
            var = S_t_run[0, 0],
            rho = rho_t_run[0, 0],
        )
        
        # Update which data is used for each run, in each frame
        def update_var_kw(f, run):
            var_kw.update(
                X   = X_t_run[run, frames[f]],
                var = S_t_run[run, frames[f]],
                rho = rho_t_run[run, frames[f]],
            )

        # Get params that don't change over time
        hex_static_kw = {
            k: plot_kwargs[k] 
            for k in plot_kwargs.keys() - var_kw.keys()
        }
        
        # Set scatterplot kwargs
        scatter_kw = dict(
            xlabel="Days",
            xticks=(0,2,4,6,8),
            xlim=(t_days.min() - 0.5, t_days.max() + 0.5),
            ylabel=r"Radius ($mm$)",
            yticks=(0, 0.1, 0.2, 0.3, 0.4,),
            ylim=(-0.01, R_t_run.max()), 
        )

        def update_scatter_kw(run):
            scatter_kw.update(
                ylabel=[r"Radius ($mm$)", "", ""][run],
                yticks=[(0, 0.1, 0.2, 0.3, 0.4,), (), ()][run],
            )

        # Plot one frame of animation
        def make_frame(f):
            
#            print(f"Frame {f+1} / {n_frames}")
            
            # Set title at top of figure
            plt.suptitle(f"Time: {t_days[frames[f]]:.2f} days",)

            # Make a 2 x 3 layout of plots
            for idx, ax in enumerate(axs.flat):
                
                row = idx // 3
                col = idx % 3
                
                # Update plotting params
                update_var_kw(f, col)
                update_scatter_kw(col)
                
                # Clear axis
                ax.clear()

                # Plot cell sheet
                if row == 0:
                    lsig.plot_hex_sheet(
                        ax=ax,
                        title=example_names[col],
                        **var_kw,
                        **hex_static_kw,
                    )

                # Plot area
                elif row == 1:
                    
                    # Get time-series up to this point
                    ffin = int(np.maximum(1, frames[f]))
                    _t = t_days[:ffin]
                    _R = R_t_run[col, :ffin]
                    
                    # Plot time-series and highlight current time
                    ax.plot(_t, _R, "b-")
                    ax.plot(_t[-1], _R[-1], color="k", marker="o", alpha=0.2)

                    ax.set(**scatter_kw)
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
#                    plt.tight_layout()

                    
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
        _anim_FA.save(
            _fpath, 
            writer=_writer, 
            dpi=dpi, 
            progress_callback=lambda i, n: print(f"Frame {i+1} / {n}"),
        )


main(
    save=True, 
    n_frames=100,
    print_frame=True,
)
