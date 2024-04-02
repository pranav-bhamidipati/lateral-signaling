from copy import deepcopy
import numpy as np
import pandas as pd

from matplotlib import animation
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.colors import ListedColormap

import colorcet as cc
import matplotlib as mpl
import matplotlib.pyplot as plt

from lateral_signaling import ceiling, vround, normalize, ref_cell_diam_um, plot_dir


####### Color utils

# Colors for specific uses
_sender_clr = "#e330ff"
_sender_clr2 = "#C43EA8BF"
_gfp_green = "#0B7E18"
_receiver_red = "#C25656E6"

# Color swatches

pinks = [
    "#fde0dd",
    "#fa9fb5",
    "#c51b8a",
]

greens = [
    "#e5f5e0",
    "#a1d99b",
    "#31a354",
    "#006d2c",
]

yob = [
    "#fff7bc",
    "#fec44f",
    "#d95f0e",
    "#9a3403",
]

cols_blue = [
    "#2e5a89",
    "#609acf",
    "#afc4cd",
]

cols_teal = [
    "#37698a",
    "#1dab99",
    "#5c6c74",
]

cols_green = [
    "#37698a",
    "#80c343",
    "#9cb6c1",
]

cols_red = [
    "#e16566",
    "#efc069",
    "#b0c5cd",
]

dark_blue = "#173b84"
purple = "#8856a7"

white = "#ffffff"
light_gray = "#eeeeee"
gray = "#aeaeae"
dark_gray = "#5a5a5a"
darker_gray = "#303030"
black = "#060605"

# Custom version of the "KGY" colormap
kgy = ListedColormap(cc.cm["kgy"](np.linspace(0, 0.92, 256)))

# Custom categorical color list(s)
growthrate_colors = [purple, greens[3], yob[1]]


def rgb_as_int(rgb):
    """Coerce RGB iterable to a tuple of integers"""
    if any([v >= 1.0 for v in rgb]):
        _rgb = tuple(rgb)
    else:
        _rgb = tuple((round(255 * c) for c in rgb))

    return _rgb


def rgb_as_float(rgb):
    """Coerce RGB iterable to an ndarray of floats"""
    if any([v >= 1.0 for v in rgb]) or any([type(v) is int for v in rgb]):
        _rgb = (np.asarray(rgb) / 255).astype(float)
    else:
        _rgb = np.asarray(rgb).astype(float)

    return _rgb


def sample_cycle(cycle, size):
    """Sample a continuous colormap at regular intervals to get a linearly segmented map"""
    from holoviews import Cycle

    return Cycle([cycle[i] for i in ceiling(np.linspace(0, len(cycle) - 1, size))])


def blend_hex(hex1, hex2, c=0.5):
    """Returns a blend of two colors."""
    blend_rgb = c * np.asfarray(hex2rgb(hex1)) + (1 - c) * np.asfarray(hex2rgb(hex2))
    return rgb2hex(tuple(round(i) for i in blend_rgb))


def hex2rgb(h):
    """Convert 6-digit hex code to RGB values (0, 255)"""
    h = h.lstrip("#")
    return tuple(int(h[(2 * i) : (2 * (i + 1))], base=16) for i in range(3))


def rgb2hex(rgb):
    """Converts rgb colors to hex"""

    RGB = np.zeros((3,), dtype=np.uint8)
    for i, _c in enumerate(rgb):

        # Convert vals in [0., 1.] to [0, 255]
        if _c <= 1.0:
            c = int(_c * 255)
        else:
            c = _c

        # Calculate new values
        RGB[i] = round(c)

    return "#{:02x}{:02x}{:02x}".format(*RGB)


def rgba2hex(rgba, background=(255, 255, 255)):
    """
    Adapted from StackOverflow
    ------------------

    Question: Convert RGBA to RGB in Python
    Link: https://stackoverflow.com/questions/50331463/convert-rgba-to-rgb-in-python/50332356
    Asked: May 14 '18 at 13:25
    Answered: Nov 7 '19 at 12:40
    User: Feng Wang
    """

    rgb = np.zeros((3,), dtype=np.uint8)
    *_rgb, a = rgba

    for i, _c in enumerate(_rgb):

        # Convert vals in [0., 1.] to [0, 255]
        if _c <= 1.0:
            c = int(_c * 255)
        else:
            c = _c

        # Calculate new values
        rgb[i] = round(a * c + (1 - a) * background[i])

    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _hexa2hex(h, alpha, background="#ffffff"):
    """
    Returns hex code of the color observed when displaying
    color `h` (in hex code) with transparency `alpha` on a
    background color `background` (default white)
    """

    # Convert background to RGB
    bg = hex2rgb(background)

    # Get color in RGBA
    rgba = *hex2rgb(h), alpha

    # Convert to HEX without transparency
    return rgba2hex(rgba, bg)


# Vectorized function to convert HEX and alpha to HEX given background
hexa2hex = np.vectorize(_hexa2hex)


####### Lattice coordinates

# Vertices of a regular hexagon centered at (0,0) with width 1.
_hex_vertices = np.array(
    [
        np.cos(np.arange(0, 2 * np.pi, np.pi / 3) + np.pi / 6),
        np.sin(np.arange(0, 2 * np.pi, np.pi / 3) + np.pi / 6),
    ]
).T / np.sqrt(3)

_hex_x, _hex_y = _hex_vertices.T


##### Visualization options

# Colorbar
_vmin = 0.0
_vmax = 0.3
cbar_kwargs = dict(ticks=[_vmin, _vmax], label="GFP (AU)")

# Scalebar
sbar_kwargs = dict(
    dx=ref_cell_diam_um,
    units="um",
    color="w",
    box_color="w",
    box_alpha=0,
    scale_loc="none",
    width_fraction=0.03,
    location="lower right",
)

# Container for plotting kwargs
plot_kwargs = dict(
    # sender_idx=sender_idx,
    # xlim=xlim,
    # ylim=ylim,
    vmin=_vmin,
    vmax=_vmax,
    cmap=kgy,
    colorbar=True,
    cbar_aspect=12,
    extend=None,
    cbar_kwargs=cbar_kwargs,
    scalebar=False,
    sbar_kwargs=sbar_kwargs,
)


def predictive_regression(
    samples,
    samples_x,
    percentiles=[68, 90],
    key_dim="__x",
):
    """
    Compute a predictive regression plot from samples.

    Heavily inspired by the `beb103` package by Justin Bois. The
    main difference is using matplotlib instead of Bokeh.

    See `bebi103.viz.predictive_regression` for documentation.
    """

    if not isinstance(samples, np.ndarray):
        raise RuntimeError("Samples can only be Numpy arrays.")

    if not isinstance(samples_x, np.ndarray):
        raise RuntimeError("samples_x can only be a Numpy array.")

    if len(percentiles) > 4:
        raise RuntimeError("Can specify maximally four percentiles.")

    if samples.shape[1] != len(samples_x):
        raise ValueError(
            "`samples_x must have the same number of entries as `samples` does columns."
        )

    # Build ptiles
    percentiles = np.sort(percentiles)[::-1]
    ptiles = [pt for pt in percentiles if pt > 0]
    ptiles = (
        [50 - pt / 2 for pt in percentiles]
        + [50]
        + [50 + pt / 2 for pt in percentiles[::-1]]
    )
    ptiles_str = [str(pt) for pt in ptiles]

    df_pred = pd.DataFrame(
        data=np.percentile(samples, ptiles, axis=0).transpose(),
        columns=ptiles_str,
    )
    df_pred[key_dim] = samples_x
    df_pred = df_pred.sort_values(by=key_dim)

    return df_pred


def default_rcParams(
    SMALL_SIZE=12,
    MEDIUM_SIZE=14,
    BIGGER_SIZE=16,
):
    """Set default parameters for Matplotlib"""

    # Set font sizes
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_predictive_regression(
    df_pred=None,
    data=None,
    key_dim="__x",
    ax=None,
    figsize=(8, 8),
    colors=cc.cm["gray"](np.linspace(0.2, 0.85, 7))[::-1],
    median_lw=2,
    ci_kwargs=dict(),
    median_kwargs=dict(),
    data_kwargs=dict(),
):
    """
    Compute a predictive regression plot from samples.

    Heavily inspired by the `beb103` package by Justin Bois. The
    main difference is using matplotlib instead of Bokeh.

    See `bebi103.viz.predictive_regression` for documentation.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    plt.sca(ax)

    # Confidence regions
    ptiles_str = df_pred.columns.tolist()
    ptiles_str.remove(key_dim)
    n = (len(ptiles_str) - 1) // 2
    for i in range(n):
        plt.fill_between(
            key_dim,
            ptiles_str[i],
            ptiles_str[2 * n - i],
            data=df_pred,
            color=colors[i],
            edgecolor=(0, 0, 0, 0),
            **ci_kwargs,
        )

    # Median as a line
    plt.plot(
        key_dim,
        ptiles_str[n],
        data=df_pred,
        linewidth=median_lw,
        color=colors[-1],
        **median_kwargs,
    )

    # It's useful to have data as a data frame
    if data is not None:
        if type(data) == tuple and len(data) == 2 and len(data[0]) == len(data[1]):
            data = np.vstack(data).transpose()
        df_data = pd.DataFrame(data=data, columns=["__data_x", "__data_y"])
        df_data = df_data.sort_values(by="__data_x")

        # Plot data points
        data_color = data_kwargs.pop("c", "k")
        data_size = data_kwargs.pop("s", 15)
        data_marker = data_kwargs.pop("marker", "o")
        data_alpha = data_kwargs.pop("alpha", 1.0)
        plt.scatter(
            "__data_x",
            "__data_y",
            data=df_data,
            c=data_color,
            s=data_size,
            marker=data_marker,
            alpha=data_alpha,
            **data_kwargs,
        )

    return plt.gca()


def remove_RT_spines(plot, element):
    """Hook to remove right and top spines from Holoviews plot"""
    plot.state.axes[0].spines["right"].set_visible(False)
    plot.state.axes[0].spines["top"].set_visible(False)


##### Plotting and animating


def plot_hex_sheet(
    ax,
    X,
    var,
    rho=1.0,
    vmin=None,
    vmax=None,
    cmap="CET_L8",
    ec=None,
    title=None,
    xlim=(),
    ylim=(),
    axis_off=True,
    aspect=None,
    sender_idx=np.array([], dtype=int),
    sender_clr=_sender_clr,
    colorbar=False,
    cbar_aspect=20,
    extend=None,
    cbar_kwargs=dict(),
    poly_padding=0.0,
    scalebar=False,
    sbar_kwargs=dict(),
    **kwargs,
):

    # Clear axis (allows you to reuse axis when animating)
    # ax.clear()
    if axis_off:
        ax.axis("off")

    # Get min/max values in color space
    if vmin is None:
        vmin = var.min()
    if vmax is None:
        vmax = var.max()

    # Get colors based on supplied values of variable
    if type(cmap) is str:
        _cmap = cc.cm[cmap]
    else:
        _cmap = cmap
    colors = np.asarray(_cmap(normalize(var, vmin, vmax)))

    # Replace sender(s) with appropriate color
    if isinstance(sender_clr, str):
        _sender_clr = hex2rgb(sender_clr)
    else:
        _sender_clr = list(sender_clr)

    if len(_sender_clr) == 3:
        _sender_clr = [*rgb_as_float(_sender_clr), 1.0]
    else:
        _sender_clr = [*rgb_as_float(_sender_clr[:3]), _sender_clr[3]]

    colors[sender_idx] = _sender_clr

    # Get polygon size. Optionally increase size
    #  so there's no visual gaps between cells
    _r = (1 + poly_padding) / np.sqrt(rho)

    # Plot cells as polygons
    for i, (x, y) in enumerate(X):

        ax.fill(_r * _hex_x + x, _r * _hex_y + y, fc=colors[i], ec=ec, **kwargs)

    # Set figure args, accounting for defaults
    if title is not None:
        ax.set_title(title)
    if not xlim:
        xlim = [X[:, 0].min(), X[:, 0].max()]
    if not ylim:
        ylim = [X[:, 1].min(), X[:, 1].max()]
    if aspect is None:
        aspect = 1
    ax.set(
        xlim=xlim,
        ylim=ylim,
        aspect=aspect,
    )

    if colorbar:

        # Calculate colorbar extension if necessary
        if extend is None:
            n = var.shape[0]
            ns_mask = ~np.isin(np.arange(n), sender_idx)
            is_under_min = var.min(initial=0.0, where=ns_mask) < vmin
            is_over_max = var.max(initial=0.0, where=ns_mask) > vmax
            _extend = ("neither", "min", "max", "both")[is_under_min + 2 * is_over_max]
        else:
            _extend = extend

        # Construct colorbar
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin, vmax), cmap=_cmap),
            ax=ax,
            aspect=cbar_aspect,
            extend=_extend,
            **cbar_kwargs,
        )

    if scalebar:
        sb_kw = plot_kwargs["sbar_kwargs"].copy()
        sb_kw.update(sbar_kwargs)
        _scalebar = ScaleBar(**sb_kw)
        ax.add_artist(_scalebar)


def animate_hex_sheet(
    fname,
    X_t,
    var_t,
    rho_t=1.0,
    fig=None,
    ax=None,
    anim=None,
    n_frames=100,
    fps=20,
    dpi=300,
    title_fun=None,
    save_dir=plot_dir,
    writer="ffmpeg",
    fig_kwargs=dict(),
    plot_kwargs=dict(),
    _X_func=None,
    _var_func=None,
    _rho_func=None,
    **kwargs,
):

    nt = var_t.shape[0]
    n = X_t.shape[-2]

    if _X_func is None:

        if X_t.ndim == 2:
            _X_func = lambda i: X_t
        elif X_t.ndim == 3:
            _X_func = lambda i: X_t[i]

    if _var_func is None:

        if var_t.ndim == 1:
            _var_func = lambda i: var_t
        elif var_t.ndim == 2:
            _var_func = lambda i: var_t[i]

    if _rho_func is None:

        _rho_t = np.asarray(rho_t)
        if rho_t.ndim == 0:
            _rho_func = lambda i: rho_t
        elif rho_t.ndim == 1:
            _rho_func = lambda i: rho_t[i]

    if title_fun is None:
        tf = lambda i: None

    # Generate figure and axes if necessary
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(**fig_kwargs)

    # If colorbar is specified, plot only once
    if "colorbar" in plot_kwargs:
        if plot_kwargs["colorbar"]:

            # Unpack args for colorbar
            _var = var_t[0]
            _sender_idx = plot_kwargs["sender_idx"]
            _vmin = plot_kwargs["vmin"]
            _vmax = plot_kwargs["vmax"]
            _cmap = plot_kwargs["cmap"]
            _cbar_aspect = plot_kwargs["cbar_aspect"]
            _cbar_kwargs = plot_kwargs["cbar_kwargs"]

            # Calculate colorbar extension if necessary
            if "extend" in plot_kwargs:
                _extend = plot_kwargs["extend"]
            else:
                _extend = None

            if _extend is None:
                n = _var.shape[0]
                ns_mask = ~np.isin(np.arange(n), _sender_idx)
                is_under_min = _var.min(initial=0.0, where=ns_mask) < _vmin
                is_over_max = _var.max(initial=0.0, where=ns_mask) > _vmax
                _extend = ("neither", "min", "max", "both")[
                    is_under_min + 2 * is_over_max
                ]

            # Construct colorbar
            cbar = plt.colorbar(
                plt.cm.ScalarMappable(
                    norm=mpl.colors.Normalize(_vmin, _vmax), cmap=_cmap
                ),
                ax=ax,
                aspect=_cbar_aspect,
                extend=_extend,
                **_cbar_kwargs,
            )

    # Turn off further colorbar plotting during animation
    _plot_kwargs = deepcopy(plot_kwargs)
    _plot_kwargs["colorbar"] = False

    frames = vround(np.linspace(0, nt - 1, n_frames))

    # Animate using plot_hex_sheet() if no animation func supplied
    if anim is None:

        def anim(**kw):
            ax.clear()
            plot_hex_sheet(ax=ax, **kw)

    # Make wrapper function that changes arguments with each frame
    def _anim(i):

        # Get changing arguments
        var_kw = dict(
            X=_X_func(frames[i]),
            var=_var_func(frames[i]),
            rho=_rho_func(frames[i]),
            title=title_fun(frames[i]),
        )

        # Plot frame of animation
        anim(**var_kw, **_plot_kwargs)

    try:
        _writer = animation.writers[writer](fps=fps, bitrate=1800)
    except RuntimeError:
        print(
            """
        The `ffmpeg` writer must be installed inside the runtime environment.
        Writer availability can be checked in the current enviornment by executing 
        `matplotlib.animation.writers.list()` in Python. Install location can be
        checked by running `which ffmpeg` on a command line/terminal.
        """
        )

    _anim_FA = animation.FuncAnimation(fig, _anim, frames=n_frames, interval=200)

    # Get path and print to output
    fpath = save_dir.joinpath(fname).with_suffix(".mp4")
    print("Writing to:", fpath.resolve().absolute())

    # Save animation
    _anim_FA.save(
        fpath,
        writer=_writer,
        dpi=dpi,
        progress_callback=lambda i, n: print(f"Frame {i+1} / {n}"),
    )
