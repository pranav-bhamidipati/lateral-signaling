# Set up environment
import json
from copy import deepcopy
import lateral_signaling as lsig
import numpy as np
import pandas as pd
from tqdm import tqdm

import scipy.stats as st

import colorcet as cc

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib_scalebar.scalebar import ScaleBar

import os


# Set parameters to save figures
data_dir = os.path.abspath("../data/sim_data")
save_dir = os.path.abspath("../plots")
params_json_path = os.path.join(data_dir, "sim_parameters.json")

save_figs = True
save_vids = True
dpi = 300

fmt = "png"

# Set seed for RNG
seed = 2021

# Read from JSON file
with open(params_json_path, "r") as f:
    params = json.load(f)
 
# Unpack
alpha     = float(params["alpha"])
k         = float(params["k"])
p         = float(params["p"])
delta     = float(params["delta"])
lambda_   = float(params["lambda_"])
g         = float(params["g"])
rho_0     = float(params["rho_0"])
delay     = float(params["delay"])
r_int     = float(params["r_int"])
gamma_R   = float(params["gamma_R"])
beta_args = tuple([float(params[k]) for k in params.keys() if k.startswith("beta_")])
dde_args  = tuple([float(params[k]) for k in params["dde_args"]])

# Set time parameters (dimensionless units)
tmax_tau = 6
nt_tau   = 200
tmax     = tmax_tau * delay
nt       = int(nt_tau * tmax_tau) + 1
t        = np.linspace(0, tmax, nt)

# Make square lattice centered on a sender cell
rows = cols = 50
X = lsig.hex_grid(rows, cols)
sender_idx = lsig.get_center_cells(X)
X = X - X[sender_idx]
n = X.shape[0]

# Get adjacency and normalize
Adj = lsig.get_weighted_Adj(X, r_int, sparse=True, row_stoch=True)

# Use basal promoter activity as mean of distribution
lambda_ = dde_args[4]

# Seed random initial expression
## Values are drawn from a HalfNormal distribution with mean `lambda`
S0 = st.halfnorm.rvs(
    size=n, 
    scale=lambda_ * np.sqrt(np.pi/2), 
    random_state=seed,
).astype(np.float32)

# Fix sender cell(s) to expression of 1
S0[sender_idx] = 1

# Get RHS of DDE equation to pass to integrator
rhs = lsig.get_DDE_rhs(lsig.signal_rhs, Adj, sender_idx, lsig.beta_rho_exp, beta_args,)

# Make args for each density 
rhos = np.array([1, 2, 4])
args_rho = [(*dde_args[:6], r) for r in rhos]

# Integrate
S_t_rho = np.array([
    lsig.integrate_DDE(
        t,
        rhs,
        dde_args=_args,
        E0=S0,
        delay=delay,
        progress_bar=False,
    )
    for _args in args_rho
])

# Get cell locations at each density 
X_rho = np.array([
    X / np.sqrt(rho)
    for rho in rhos
])


############# Plotting

# Get cell diameter for plotting
ref_density =  1250  # cells / mm^2
ref_density /= 1e6   # cells / um^2
cell_diam = np.sqrt(
    2 / (np.sqrt(3) * ref_density)
)
# print(f"Cell diameter at reference density: {cell_diam:.2f} μm")

# Scalebar
sbar_kwargs = dict(
    dx=cell_diam,
    units="um", 
    color="w", 
    box_color="w", 
    box_alpha=0, 
    font_properties=dict(weight=1000, size=10), 
    width_fraction=0.03,
    location="lower right",
)

# Get plotting boundaries, centered on sender cell (0, 0)
bounds = np.array([X.min(axis=0), X.max(axis=0)]).T / np.sqrt(max(rhos))
halfwidth, halfheight = np.min(np.abs(bounds), axis=1)
xlim = ( -halfwidth,  halfwidth)
ylim = (-halfheight, halfheight)

# Set colorscale limits (for colorbar)
vmin = 0.0
vmax = S_t_rho.max()

# Plotting
plot_kwargs = dict(
    sender_idx=sender_idx,
    xlim=xlim,
    ylim=ylim,
    vmin=vmin,
    vmax=vmax,
    cmap=lsig.kgy,
    colorbar=False,
    scalebar=False,
)

# Plot a (3 x 3) layout of frames
rows = 3
cols = 3
fig, axs = plt.subplots(
    rows, 
    cols, 
    figsize=(6.25, 5),
    gridspec_kw=dict(width_ratios=[1, 1, 1.2]),
)

for i, ax in enumerate(axs.flat):
    
    # Select current frame
    row = i // rows
    col = i % rows
    time_tau = (tmax_tau // cols) * (1 + col)
    frame = time_tau * nt_tau
    
    # Make colorbars invisible except in first row
    if row == 0:
#        cbar_kwargs = dict(ticks=[vmin, vmax], label="GFP (AU)", shrink=0.9, format="%.2f")
        cbar_kwargs = dict(ticks=[], label="", shrink=1.0)
    else: 
        cbar_kwargs = dict(ticks=[], label="", alpha=0, shrink=1e-5)
    
    if row == 2:
        mpl.rcParams["axes.titlesize"] = 14
        ax.set_title(fr"{time_tau:.0f} $\tau$", y=-0.2)

    # Plot frame
    lsig.plot_hex_sheet(
        ax=ax,
        X=X_rho[row],
        var=S_t_rho[row, frame],
        rho=rhos[row],
        **plot_kwargs
    )
    
    # Plot colorbars in last column
    if col == 2:
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin, vmax), 
                cmap=lsig.kgy,
            ), 
            ax=ax,
            aspect=10,
            **cbar_kwargs
        )
    
    # Hide scalebar text except first image
    font_size = (i == 0) * 10
    sbar_kwargs["font_properties"].update(
        size = font_size,
    )
    
    # Make scalebar
    scalebar = ScaleBar(**sbar_kwargs)
    ax.add_artist(scalebar)

plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05)

if save_figs:
    fig_fname = f"simulation_constant_density_rho_1-4x_{tmax_tau}tau"
    fig_path = os.path.join(save_dir, fig_fname + "." + fmt)
    plt.savefig(fig_path, dpi=dpi, format=fmt,)
    print(f"Saved figure to {fig_path}")

########### Animation    

# Make title of animation
suptitle_fun = lambda i: fr"Simulation time: {t[i]:.2f} ({t[i]:.2f} $\tau$)"

# Change args for animation
anim_plot_kwargs = deepcopy(plot_kwargs)

sbar_kwargs = dict(
    dx=cell_diam,
    units="um", 
    color="w", 
    box_color="w", 
    box_alpha=0, 
    font_properties=dict(weight=1000, size=10), 
    width_fraction=0.03,
    location="lower right",
)

anim_plot_kwargs.update(
    colorbar=True,
    cbar_aspect=8,
    cbar_kwargs=dict(
        ticks=[vmin, vmax], label="GFP (AU)", shrink=0.7, format="%.2f"
    ),
    scalebar=True,
    sbar_kwargs=sbar_kwargs,
)


# Make figure
fig, axs = plt.subplots(
    nrows=1, 
    ncols=3, 
    figsize=(6.5, 2.5),
    gridspec_kw=dict(width_ratios=[1, 1, 1.2]),
)

def anim_func(**kw):

    # Unpack params into plotting params and variables
    plot_kw = {k: kw[k] for k in kw.keys() - set(["X", "var", "rho"])}
    _X_rho = kw["X"]
    var_rho = kw["var"]
    rhos = kw["rho"]

    # Extract title and plot as suptitle
    plt.suptitle(plot_kw["title"])

    for j in range(len(axs)):

        # Wipe last frame
        axs[j].clear()

        # Get axis title
        plot_kw["title"] = ("1x", "2x", "4x")[j] + " density"

        # Plot new frame
        lsig.plot_hex_sheet(
            ax=axs[j], 
            X=_X_rho[j],
            var=var_rho[j],
            rho=rhos[j],
            **plot_kw
        )

# Specify axis for colorbar
cbar_ax = axs[-1]

if save_vids:
    
    # Path for video
    vid_fname = f"simulation_constant_density_{int(rhos[0])}x-{int(rhos[-1])}x_.mp4"
    vid_fpath = os.path.join(save_dir, vid_fname)

    # Make animation
    lsig.animate_hex_sheet(
        fpath=vid_fpath,
        X_t=X_rho[-1],       # These args are 
        var_t=S_t_rho[-1],     # overrided by the 
        rho_t=rhos[-1],          # "_func" args below
        fig=fig, 
        ax=cbar_ax,
        anim=anim_func,
        n_frames=100,
        fps=15,
        dpi=dpi,
        title_fun=suptitle_fun,
        plot_kwargs=anim_plot_kwargs,
        _X_func   = lambda i: X_rho,
        _var_func = lambda i: S_t_rho[:, i],
        _rho_func = lambda i: rhos,
    )
    print(f"Saved animation to {vid_fpath}")

