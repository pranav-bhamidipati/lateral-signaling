import os
from glob import glob
import json
import h5py

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation

import lateral_signaling as lsig



data_dir      = os.path.abspath("../data/sim_data/20211201_singlespotphase/sacred")
save_dir      = os.path.abspath("../plots")
example_fpath = os.path.abspath("../data/sim_data/phase_examples.json")

save_figs = True
fig_fmt   = "png"
dpi       = 300

# Get threshold for v_init
with open(example_fpath, "r") as f:
    examples = json.load(f)
    
    example_names  = examples["name"]
    example_gs     = np.asarray(examples["g"])
    example_rho_0s = np.asarray(examples["rho_0"])

# Get directory for each run in parameter sweep
run_dirs = glob(os.path.join(data_dir, "[0-9]*"))

# Store each run's data in a DataFrame
dfs = []
for g, rho_0 in zip(example_gs, example_rho_0s):
    for rd in run_dirs:    

        _config_file = os.path.join(rd, "config.json")
        _results_file = os.path.join(rd, "results.hdf5")

        if (not os.path.exists(_config_file)) or (not os.path.exists(_results_file)):
            continue

        # Find the correct run
        with open(_config_file, "r") as c:
            config = json.load(c)

            # Initial density, carrying capacity
            rho_0  = config["rho_0"]

            if rho_0 not in example_rho_0s:
                continue
            else:
                print(f"Found example {np.where(example_rho_0s == rho_0)[0][0]} - rho_0")

            rho_max  = config["rho_max"]

        # Get remaining info from run's data dump
        with h5py.File(_results_file, "r") as f:

            # Phase metrics
            v_init    = np.asarray(f["v_init_g"])
            n_act_fin = np.asarray(f["n_act_fin_g"])

            # Expression and density information
            S_t_g_tcmean = np.asarray(f["S_t_g_tcmean"])
            S_t_g_actnum = np.asarray(f["S_t_g_actnum"])
            rho_t_g      = np.asarray(f["rho_t_g"])

            # Proliferation rates and time-points
            g = list(f["g_space"])
            t = np.asarray(f["t"])

        # Store things that are functions of time
        tcmean_ts.append(S_t_g_tcmean)
        actnum_ts.append(S_t_g_actnum
        rho_ts.append(rho_t_g)

# Concatenate into one dataset and isolate rows with the right parameters
df = pd.concat(dfs).reset_index(drop=True)
df = df.loc[np.isin(df.g, example_gs), :]

print(df)

0 / 0

# Path to save animation
save_dir = os.path.abspath("../plots")
fpath = os.path.join(save_dir, f"phase_examples_.mp4")

def main(
    fpath=fpath,
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
    run_dirs = [f"./sacred/{r}/" for r in sacred_runs]

    deltas    = []
    t         = []
    S_t_res   = []
    R_t_res   = []

    for i, d in enumerate(run_dirs):
        
        cfg_f = os.path.join(d, "config.json")
        res_f = os.path.join(d, "results.hdf5")
        
        # Extract time and expression
        with h5py.File(res_f, "r") as f:
            if i == 0:
                t = np.asarray(f["t"])
                nt = t.size
            
            S_t_res.append(np.asarray(f["S_t"]))
            R_t_res.append(np.asarray(f["R_t"]))

        # Extract parameter value(s)
        with open(cfg_f, "r") as f:

            _config  = json.load(f)
            
            if i == 0:

                g          = _config["g"]
                rho_0      = _config["rho_0"]
                rho_max    = _config["rho_max"]
                rho_t      = lsig.logistic(t, g, rho_0, rho_max)

                rows       = _config["rows"]
                cols       = _config["cols"]
                X          = lsig.hex_grid(rows, cols)
                sender_idx = lsig.get_center_cells(X)
                X          = X - X[sender_idx]
                n          = X.shape[0]
                X_t        = np.multiply.outer(1 / np.sqrt(rho_t), X)
            
            if _config["delta"] is dict:
                _delta = _config["delta"]["value"]
            else:
                _delta = _config["delta"]

            deltas.append(_delta)

    if save:
        
        # Make figure
        fig, axs = plt.subplots(
            nrows=1, 
            ncols=3, 
            figsize=(8.5, 2.5),
#            gridspec_kw=dict(width_ratios=[1, 1, 1.2]),
        )

        X_func       = lambda    i: X_t[i]
        var_func     = lambda r, i: S_t_res[r][i]
        rho_func     = lambda    i: rho_t[i]
        suptitle_fun = lambda    i: f"Time: {t[i]:.2f} days"
        title_fun    = lambda    r: ("No", "Weak", "Strong")[r] + " inhibition"
            
        # Get default kwargs for plotting
        plot_kwargs = deepcopy(lsig.plot_kwargs)
        plot_kwargs["sender_idx"] = sender_idx
        
        # Get axis limits
        plot_kwargs["xlim"] = X_t[-1, :, 0].min(), X_t[-1, :, 0].max()
        plot_kwargs["ylim"] = X_t[-1, :, 1].min(), X_t[-1, :, 1].max()

        # Set some args for colorbar
        plot_kwargs["cmap"] = lsig.kgy
        plot_kwargs["cbar_aspect"] = 8
        plot_kwargs["cbar_kwargs"]["shrink"] = 0.7
        plot_kwargs["extend"] = "neither"
                    
        # Get colorscale limits
        ns_mask = np.ones(n, dtype=bool)
        ns_mask[sender_idx] = False
        if vmins is None:
            vmins = [v[:, ns_mask].min() for v in S_t_res]
        if vmaxs is None:
            vmaxs = [v[:, ns_mask].max() for v in S_t_res]
        
        # Make list to hold colorbars for easy reference
        colorbars = []

        # Turn off automatic colorbar plotting
        plot_kwargs["colorbar"] = False
           
        # Get frames to animate
        frames = lsig.vround(np.linspace(0, nt-1, n_frames))
            
        # Get params that don't change over time
        static_kw = {
            k: plot_kwargs[k] 
            for k in plot_kwargs.keys() - set(
                ["X", "var", "rho", "title", "vmin", "vmax", "cbar_kwargs"]
             )
        }

        def _anim(i):
            """Creates one frame of animation."""

            # Make each plot in layout
            for r, ax in enumerate(axs):

                # Get changing arguments
                var_kw = dict(
                    X     = X_func(frames[i]),
                    var   = var_func(r, frames[i]),
                    rho   = rho_func(frames[i]),
                    title = title_fun(r),
                    vmin  = vmins[r],
                    vmax  = vmaxs[r],
                )
            
                # Plot frame of animation
                ax.clear()
                lsig.plot_hex_sheet(ax=ax, **var_kw, **static_kw)
                
            # Make colorbars if they don't exist
            if not colorbars:
                for r, ax in enumerate(axs):
                    
                    _vmin = vmins[r]
                    _vmax = vmaxs[r]
                    
                    _cbar_kwargs = dict(
                        ticks=[_vmin, _vmax],
                        format="%.1f",
                        label=("", "GFP (AU)")[r == (len(axs) - 1)],
                    )

                    cbar = plt.colorbar(
                        plt.cm.ScalarMappable(
                            norm=mpl.colors.Normalize(_vmin, _vmax), 
                            cmap=plot_kwargs["cmap"]), 
                        ax=ax,
                        aspect=plot_kwargs["cbar_aspect"],
                        extend=plot_kwargs["extend"],
                        **_cbar_kwargs
                    )
                    colorbars.append(cbar)

            # Change suptitle
            suptitle = suptitle_fun(frames[i])
            plt.suptitle(suptitle);
            
            if i == 0:
                plt.tight_layout()
            
            if print_frame:
                print(
                    "Frame: ", 
                    str(i + 1).rjust(len(str(n_frames))), 
                    " / ", 
                    str(n_frames), 
                    sep=""
                )

        try:
            _writer = animation.writers[writer](fps=fps, bitrate=1800)
        except RuntimeError:
            print("""
            The `ffmpeg` writer must be installed inside the runtime environment.
            Writer availability can be checked in the current enviornment by executing 
            `matplotlib.animation.writers.list()` in Python. Install location can be
            checked by running `which ffmpeg` on a command line/terminal.
            """)
        
        _anim_FA = animation.FuncAnimation(fig, _anim, frames=n_frames, interval=200)
        
        # Clean up path
        _fpath = str(fpath)
        if not _fpath.endswith(".mp4"):
            _fpath += ".mp4"
        print("Writing to:", _fpath)
        
        # Make animation
        _anim_FA.save(_fpath, writer=_writer, dpi=dpi)

#        # Make video
#        lsig.animate_hex_sheet(
#            fpath=layout_vid_path,
#            X_t=X_rho_arr[-1],
#            var_t=var_rho_t[-1],
#            rho_t=rhos[-1],
#            fig=fig,
#            ax=cbar_ax,
#            anim=anim_func,
#            n_frames=100,
#            fps=15,
#            dpi=dpi,
#            title_fun=suptitle_fun,
#            plot_kwargs=anim_plot_kwargs,
#            _X_func   = lambda i: X_rho_arr,
#            _var_func = lambda i: var_rho_t[:, i],
#            _rho_func = lambda i: rhos,
#        )
#

# # Run animation function
# vmin = 0.
# vmax = 0.75

main(
    save=True, 
    n_frames=100, 
    print_frame=True,
)


