import json
import h5py

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as rotation
from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

import lateral_signaling as lsig
from itertools import islice


# Reading
sacred_dir = lsig.simulation_dir.joinpath("20211201_singlespotphase/sacred")
thresh_fpath = lsig.simulation_dir.joinpath("phase_threshold.json")


def main(
    figsize=(8, 8),
    xyz=["g_inv_days", "rho_max", "rho_0"],
    save_dir=lsig.plot_dir,
    save=False,
    fmt="png",
    dpi=300,
):

    ## Read in and assemble data
    # Get threshold for v_init
    v_init_thresh = float(json.load(thresh_fpath.open("r"))["v_init_thresh"])

    # Read in phase metric data
    run_dirs = [d for d in sacred_dir.glob("*") if d.joinpath("config.json").exists()]
    dfs = []
    for rd_idx, rd in enumerate(tqdm(run_dirs)):

        # Get some info from the run configuration
        with rd.joinpath("config.json").open("r") as c:
            config = json.load(c)

            # Initial density, carrying capacity
            rho_0 = config["rho_0"]
            rho_max = config["rho_max"]

        if rho_0 < 1.0:
            continue

        # Get remaining info from run's data dump
        with h5py.File(rd.joinpath("results.hdf5"), "r") as f:

            # Phase metrics
            v_init = np.asarray(f["v_init_g"])
            n_act_fin = np.asarray(f["n_act_fin_g"])

            # Proliferation rates
            g = list(f["g_space"])

        # Assemble dataframe
        _df = pd.DataFrame(
            dict(
                v_init=v_init,
                n_act_fin=n_act_fin,
                g=g,
                rho_0=rho_0,
                rho_max=rho_max,
            )
        )
        dfs.append(_df)

    # Concatenate into one dataset
    df = pd.concat(dfs).reset_index(drop=True)
    df["g_inv_days"] = lsig.g_to_units(df["g"].values)

    # Assign phases and corresponding plot colors
    df["phase"] = (df.v_init > v_init_thresh).astype(int) * (
        1 + (df.n_act_fin > 0).astype(int)
    )
    df["color"] = np.array(lsig.cols_blue)[df.phase]

    ## Plot phase boundaries in 3D
    # Phase pairs to plot - (X, Y, Z) correspond to (0,1,2)
    phase_pairs = [
        (0, 1),
        (1, 2),
        (0, 2),
    ]

    # Colors for phase regions
    phase_colors = lsig.cols_blue[::-1]

    # Rotation vectors - optionally used for better triangulation
    #   when a part of the phase boundary is orthogonal to XY plane
    rot_idx = (1,)
    rot_vecs = ([1.3, -0.5, 0.2],)

    # Maximum allowed edge length in a triangulation
    max_edge_length = 0.9

    # Text options
    mpl.rcParams["axes.labelsize"] = 18
    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12

    # axis options

    xticks = lsig.g_to_units(np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5]))
    xtlabs = tuple([f"{v:.1f}" for v in xticks])
    axis_kw = dict(
        xlim3d=[0, lsig.g_to_units(2.5)],
        ylim3d=[0, 6.25],
        zlim3d=[0, 6.25],
        xlabel=r"$g$ ($days^{-1}$)",
        ylabel=r"$\rho_{max}$",
        zlabel=r"$\rho_0$",
        xticks=xticks,
        xtlabs=xtlabs,
        yticks=[0, 2, 4, 6],
        zticks=[0, 2, 4, 6],
    )

    # Set up plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="3d"))
    ax.zaxis.set_rotate_label(False)

    # Plot phase boundaries as surfaces
    plot_phase_boundaries_3D(
        ax3d=ax,
        data=df,
        xyz=xyz,
        rot_idx=rot_idx,
        rot_vecs=rot_vecs,
        phase_colors=phase_colors,
        phase_pairs=phase_pairs,
        max_edge_length=max_edge_length,
        **axis_kw,
    )

    text_kw = dict(
        ha="center",
        fontsize=20,
        zorder=1000,
    )

    ax.text(1.5, 4.0, 5.5, "Attenuated", "y", c="k", **text_kw)
    ax.text(1.5, 2.5, 1.0, "Unlimited", "y", c="k", **text_kw)
    ax.text(1.5, 5.3, 1.5, "Limited", "y", c="w", **text_kw)

    if save:
        _fpath = save_dir.joinpath(f"phase_boundaries_3D.{fmt}")
        print(f"Writing to: {_fpath.resolve().absolute()}")
        plt.savefig(_fpath, dpi=dpi)


## Define functions to draw phase boundaries
##   Using a rudimentary ray tracing technique
def window(seq, n):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def get_phase_boundary_idx(
    phase_grid,
    phase1,
    phase2,
):
    """
    Given a `(n1 x n2 x n3 `Numpy array containing the phase at each point in a gridded
    3D parameter space, find the points at which there's a boundary between `phase1`
    and `phase2`. Uses a ray tracing-esque method that starts with the first axis, iterates
    over each grid line parallel to that axis, finds the first boundary (`phase1`->`phase2`
    or `phase2`->`phase1`), and repeats this for each spatial axis.

    Returns a 3-tuple of lists `bound`. Each item in `bound[axis]` is the index of the
    boundary found in `griddata`.
    """

    # Get number of sampled points along each axis
    shape = phase_grid.shape

    # Initialize output
    bounds = ([], [], [])

    # Iterate over axes
    for axis in range(3):

        # Get the i and j axes (indices on the plane
        #    perpendicular to the current axis)
        _i_axis = (axis + 1) % 3
        _j_axis = (axis + 2) % 3

        # Iterate over all rays parallel to the axis
        _slice = list(np.s_[:, :, :])
        for i in range(shape[_i_axis]):
            _slice[_i_axis] = i
            for j in range(shape[_j_axis]):
                _slice[_j_axis] = j
                _slice[axis] = slice(None, None, None)

                # Make a generator object to iterate along the ray
                gen = window(phase_grid[tuple(_slice)], 2)

                # Search for the first phase boundary
                try:
                    idx = next(
                        i
                        for i, el in enumerate(gen)
                        if (el == (phase1, phase2)) or (el == (phase2, phase1))
                    )

                # If not found, no worries
                except StopIteration:
                    continue

                # If found, add the indices to bounds in correct order
                else:
                    b = tuple(np.roll((idx, i, j), shift=axis))
                    bounds[axis].append(b)

    return bounds


def get_phase_boundary_pts(bounds, *grid_axes):

    # Make a new bounds object on the other side of the boundary
    bounds1 = bounds
    bounds2 = ([], [], [])
    for axis, (b1, b2) in enumerate(zip(bounds1, bounds2)):
        if b1:
            b2 = np.asarray(b1, dtype=int)
            b2[:, axis] += 1
            bounds2[axis][:] = [tuple(row) for row in b2]

    # Concatenate indices
    b1_cat = np.concatenate([b for b in bounds1 if b]).T
    b2_cat = np.concatenate([b for b in bounds2 if b]).T

    # Get points on either side of boundary
    pts1 = np.array([grid_axes[i][b] for i, b in enumerate(b1_cat)]).T
    pts2 = np.array([grid_axes[i][b] for i, b in enumerate(b2_cat)]).T

    # Return midpoints as estimates of boundary
    return (pts1 + pts2) / 2


def plot_phase_boundaries_3D(
    ax3d,
    data,
    xyz,
    rot_idx,
    rot_vecs,
    phase_pairs,
    phase_colors,
    phase_col="phase",
    alpha=0.9,
    azim=15,
    elev=18,
    max_edge_length=0.9,
    zbias=1e-3,
    xticks=(),
    xtlabs=(),
    **kw,
):

    # Unpack axis variables
    x, y, z = xyz

    # Get which coordinates were sampled along each axis
    grid_axes = (
        np.unique(data[x]),
        np.unique(data[y]),
        np.unique(data[z]),
    )

    # Get phase values in terms of x/y/z axes
    dgrid = data[[x, y, z, phase_col]].pivot(index=[x, y], columns=z, values=phase_col)

    # Turn into (n_x x n_y x n_z) array containing the
    #   values at each point in grid.
    dgrid = np.array(list(dgrid.groupby(x).apply(pd.DataFrame.to_numpy)))

    # Set up plot
    ax3d.set(**kw)
    plt.xticks(xticks, xtlabs)
    ax3d.azim = azim
    ax3d.elev = elev

    for i, (p1, p2) in enumerate(phase_pairs):

        # Get boundary indices from gridded data
        bounds = get_phase_boundary_idx(dgrid, p1, p2)

        # Get boundary points
        bound_pts = get_phase_boundary_pts(bounds, *grid_axes)

        # Optionally, rotate boundary points before triangulation.
        # NOTE:
        #   Under the hood, MPL does a projection down to 2D to make triangulations.
        #   This produces problems with manifolds perpendicular to XY plane. Hence,
        #   rotation before triangulation can improve the quality of the projection.
        #   I tried random rotations until one produced a good triangulation.
        if i in rot_idx:

            # Find which rotation vector to use
            j = np.nonzero(np.asarray(rot_idx) == i)[0][0]

            # Rotate points
            R = rotation.from_rotvec(rot_vecs[j])
            bp = R.apply(bound_pts)

        else:
            bp = bound_pts

        # Get triangulation
        _triangles = mpl.tri.Triangulation(bp[:, 0], bp[:, 1]).triangles

        # Remove triangles with long edges
        edges = bound_pts[_triangles] - bound_pts[np.roll(_triangles, 1, axis=1)]
        edge_lengths = np.linalg.norm(edges, axis=2)
        _mask = np.any(edge_lengths > max_edge_length, axis=1)

        # Use this triangulation and mask and supply points
        _triangulation = mpl.tri.Triangulation(
            bound_pts[:, 0],
            bound_pts[:, 1],
            triangles=_triangles,
            mask=_mask,
        )

        # Plot two surfaces, one for each side of the boundary, colored appropriately
        # Z must be provided because the triangulation object only preserves X and Y
        ax3d.plot_trisurf(
            _triangulation,
            Z=bound_pts[:, 2],
            alpha=alpha,
            color=phase_colors[p1],
        )

        ax3d.plot_trisurf(
            _triangulation,
            Z=bound_pts[:, 2] - zbias,
            alpha=alpha,
            color=phase_colors[p2],
        )


if __name__ == "__main__":

    main(
        save=True,
    )
