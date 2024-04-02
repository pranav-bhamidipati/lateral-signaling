from pathlib import Path
from matplotlib import animation

import numpy as np
from numba import njit
import matplotlib as mpl
import matplotlib.pyplot as plt

import lateral_signaling as lsig

lsig.set_simulation_params()
lsig.set_growth_params()
lsig.set_steady_state_data()
lsig.viz.default_rcParams()


@njit
def get_rho_x_0(x, psi, rho_bar):
    """Number density of cells at time zero."""
    return np.log(psi) / (psi - 1) * rho_bar * psi**x


@njit
def get_rho_x_t(x, t, psi, rho_bar, rho_max):
    """
    Number density of cells over time.
    Initial condition is an exponential gradient, and growth
    follows the logistic equation.
    Diffusion and advection are assumed to be negligible.
    """
    rho_x_0 = get_rho_x_0(x, psi, rho_bar)
    return rho_max * rho_x_0 * np.exp(t) / (rho_max + rho_x_0 * (np.exp(t) - 1))


def main(
    rho_max=None,
    figsize=(4.5, 4.5),
    scale=2.0,
    rho_bar=1.0,
    psi=0.01,
    tmax_days=5,
    n_frames=101,
    cmap=lsig.viz.kgy,
    nx=201,
    bg_clr="k",
    save=True,
    save_dir=None,
    writer="ffmpeg",
    fps=20,
    dpi=300,
):
    # Compute/get parameters
    if rho_max is None:
        rho_max = lsig.mle_params.rho_max_ratio

    # Set the density gradient parameters
    # Psi is the ratio of densities at the top and bottom of the gradient
    # rho_bar is the mean density of cells in the gradient
    rho_bar = (rho_max - 1) / np.log(rho_max)

    tmax_nondim = tmax_days / lsig.t_to_units(1)
    t_nondim = np.linspace(0, tmax_nondim, n_frames)
    t_days = lsig.t_to_units(t_nondim)

    x = np.linspace(0, 1, nx)
    rho_x_t = np.zeros((n_frames, nx))
    SS_xs = np.zeros((n_frames, nx))
    for i, t in enumerate(t_nondim):
        rho_x = get_rho_x_t(x, t, psi, rho_bar, rho_max)
        rho_x_t[i] = rho_x
        SS_xs[i] = lsig.get_steady_state_mean(rho_x)

    # SS_xs = lsig.normalize(SS_xs, SS_xs.min(), SS_xs.max())
    norm = mpl.colors.Normalize(vmin=SS_xs.min(), vmax=SS_xs.max())

    # Make a circle to clip the gradient into a circle
    xx = np.linspace(-1, 1, nx * 2)
    yy = np.sqrt(1 - xx**2)
    xx *= scale
    yy *= scale
    circle = plt.fill_between(xx, yy, -yy, lw=0, color="none")
    circle_clip = circle.get_paths()[0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(bg_clr)
    ax.axis("off")
    ax.set_aspect(1)
    extent = -scale, scale, -scale, scale

    # Every frame, clear the axis and redraw the gradient
    def render_frame(frame):
        ax.cla()
        # ax.set_facecolor(bg_clr)
        # ax.set_aspect(1)
        gradient = ax.imshow(
            np.ones((nx, nx)) * SS_xs[frame][:, np.newaxis],
            cmap=cmap,
            norm=norm,
            extent=extent,
        )
        gradient.set_clip_path(
            circle_clip,
            # transform=ax.transData,
            transform=gradient.get_transform(),
        )
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        # ax.axis("off")
        ax.set_title(f"$t = {t_days[frame]:.1f}$ days", fontsize=16)

    # Make animation
    try:
        _writer = animation.writers[writer](fps=fps, bitrate=1800)
    except RuntimeError:
        print(
            "The `ffmpeg` writer must be installed inside the runtime environment. \n"
            "Writer availability can be checked in the current enviornment by executing  \n"
            "`matplotlib.animation.writers.list()` in Python. Install location can be \n"
            "checked by running `which ffmpeg` on a command line/terminal."
        )
        raise

    _anim_FA = animation.FuncAnimation(fig, render_frame, frames=n_frames, interval=200)

    if save:
        fpath = save_dir.joinpath("kinematic_wave_steady_state_simulation.mp4")
        print("Animating...")
        _anim_FA.save(
            fpath,
            writer=_writer,
            dpi=dpi,
            progress_callback=lambda i, n: print(f"Frame {i+1} / {n}"),
        )
        print("Writing to:", fpath.resolve().absolute())


if __name__ == "__main__":
    save_dir = lsig.temp_plot_dir
    main(
        save=True,
        save_dir=save_dir,
    )
