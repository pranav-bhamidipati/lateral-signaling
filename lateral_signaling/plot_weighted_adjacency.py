import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import holoviews as hv

hv.extension("matplotlib")

import lateral_signaling as lsig


def main(
    rows=12,
    boxsize=8,
    imsize=101,
    cmap="viridis",
    figsize=(6.2, 6.5),
    save_dir=lsig.plot_dir,
    save=False,
    fmt="png",
    dpi=300,
):

    # Set r_int
    ## NOTE: This is hard-coded because r_int is used in a MathTex
    ##        formula below - if it changes, the formula must be
    ##        manually changed.
    r_int = 3.0

    # Make hexagonal sheet and get cell at center
    X = lsig.hex_grid(rows, rows)
    center = lsig.get_center_cells(X)
    X = X - X[center]

    # Get adjacency of center cell to other cells
    Adj = lsig.get_weighted_Adj(X, r_int, row_stoch=True)
    center_Adj_weights = Adj[:, center].ravel()

    # Set plot options
    cbar_kw = dict(
        aspect=10, label=r"BiNormal( $X_i$ , $\frac{1}{2}\, r_{int}$ )", ticks=[]
    )
    text_kw = dict(
        ha="center",
        va="center",
        fontsize=14,
    )

    # Discretize space
    extent = boxsize / 2
    x = y = np.linspace(-extent, extent, imsize)
    xy = np.zeros((imsize, imsize, 2))
    xy[:, :, 0], xy[:, :, 1] = np.meshgrid(x, y)

    # Sample bivariate normal distribution over space
    scale = r_int / 2
    kernel = stats.multivariate_normal(mean=np.zeros(2), cov=scale * np.eye(2))
    z_mesh = kernel.pdf(xy)

    # Make figure
    fig, ax = plt.subplots(figsize=figsize)
    plt.suptitle(r"Cell adjacency weight ($w_{ij}$)", fontsize=24, y=0.95)
    #    plt.title(r"$\sum_j w_{ij} = 1 \quad \forall i$", fontsize=24, y=-0.11)
    plt.title(r"$\sum_j w_{ij} = \mathbf{1}$", fontsize=24, y=-0.11)

    # Plot multivariate Gaussian in background
    plt.pcolor(*xy.T, z_mesh, shading="auto", cmap=cmap)

    for i, (x, y) in enumerate(X):

        # Plot cell as polygons
        ax.fill(
            lsig.viz._hex_x + x,
            lsig.viz._hex_y + y,
            fc=(0, 0, 0, 0),
            ec="k",
        )

        # Plot weighted adjacency of cell i and center cell
        weight = center_Adj_weights[i]

        # Center cell is not adjacent to itself
        if i == center:
            ax.text(x, y, "--", c="k", **text_kw)

        # Cells inside the plot
        elif (-extent <= x <= extent) and (-extent <= y <= extent):

            # Non-adjacent to center cell
            if weight == 0:
                ax.text(x, y, "--", c="w", **text_kw)

            # Adjacent to center cell
            else:
                ax.text(x, y, f"{weight:.3f}"[1:], c="w", **text_kw)

    # Set plot options
    ax.axis("off")
    ax.set(
        xlim=(-extent, extent),
        ylim=(-extent, extent),
        aspect="equal",
    )

    if save:
        fpath = save_dir.joinpath(f"weighted_adjacency.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi)


main(
    save=True,
)
