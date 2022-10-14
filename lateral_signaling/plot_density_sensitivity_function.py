import numpy as np
import holoviews as hv

hv.extension("matplotlib")

import matplotlib.pyplot as plt

import lateral_signaling as lsig


def main(
    rows=12,
    m=1.0,
    rhomin=1,
    rhomax=7,
    nrho=101,
    figsize=(6, 4),
    save_dir=lsig.plot_dir,
    save=False,
    fmt="png",
    dpi=300,
):

    ## Construct data for example
    # make cell sheet
    cols = rows
    X = lsig.hex_grid(rows, cols)
    center = lsig.get_center_cells(X)
    X = X - X[center]

    # Store cell type
    var = np.zeros(X.shape[0], dtype=np.float32)
    var[center] = 1.0

    # Sample beta(rho, m) in the defined space
    rho_space = np.linspace(rhomin, rhomax, nrho)
    beta_space = lsig.beta_rho_exp(rho_space, m)

    # Set which rhos to use as examples
    rhos = np.array([1.0, 2.0, 4.0])
    betas = lsig.beta_rho_exp(rhos, m)
    X_rho = np.multiply.outer(1 / np.sqrt(rhos), X)

    ## Plotting options
    # Set axis limits for inset plots
    _extent = np.min([-X_rho[-1].min(axis=0).max(), X_rho[-1].max(axis=0).min()])
    xlim = -_extent, _extent
    ylim = -_extent, _extent

    # Set locations for insets
    inset_locs = np.array(
        [
            [1.34, 0.72],
            [2.59, 0.335],
            [4.85, 0.195],
        ]
    )
    ins_width, ins_height = 1.75, 0.275

    # Set axis limits for larger plot
    plot_xlim = rhomin - 0.5, rhomax + 0.5
    plot_ylim = -0.05, 1.05

    ## Make plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot curve
    ax.plot(rho_space, beta_space)

    # Plot lattices as inset plots
    for i, _ in enumerate(rhos):

        # Make inset axis
        x_ins, y_ins = inset_locs[i]
        axins = ax.inset_axes(
            [x_ins, y_ins, ins_width, ins_height],
            transform=ax.transData,
        )

        # Draw line from curve to inset ensuring line isn't touching either
        src = np.array([rhos[i], betas[i]])
        dst = np.array([x_ins + ins_width / 6, y_ins + ins_height / 2])
        pad = 0.1 * (dst - src) / np.linalg.norm(dst - src)
        src = src + pad
        dst = dst - pad
        ax.plot(
            (src[0], dst[0]),
            (src[1], dst[1]),
            c="k",
        )

        # Plot cells
        lsig.plot_hex_sheet(
            ax=axins,
            X=X_rho[i],
            var=var,
            rho=rhos[i],
            sender_idx=center,
            xlim=xlim,
            ylim=ylim,
            ec="w",
            lw=0.2,
            scalebar=True,
            sbar_kwargs=dict(font_properties=dict(size=0)),
            # axis_off=False,
        )

    # Add titles/labels
    ax.set_title(r"Signaling coefficient ($\beta$) vs density ($\rho$)", fontsize=18)
    ax.text(
        rho_space.max(),
        beta_space.max() - 0.125,
        #        beta_space.max(),
        r"$\beta(\rho, m)=e^{-m(\rho-1)}$",
        ha="right",
        va="top",
        fontsize=14,
    )
    #    ax.text(
    #        rho_space.max(),
    #        beta_space.max() - 0.125,
    #        f"$m={m:.2f}$",
    #        ha="right",
    #        va="top",
    #        fontsize=14,
    #    )

    # Set other options
    ax.set_xlim(plot_xlim)
    ax.set_ylim(plot_ylim)
    ax.set_xlabel(r"$\rho$", fontsize=16)
    ax.set_ylabel(r"$\beta$", fontsize=16)
    #    ax.set_ylabel("Signaling coefficient", fontsize=16)
    plt.tick_params(labelsize=12)
    # ax.set_aspect(4)

    plt.tight_layout()

    if save:
        fpath = save_dir.joinpath(f"signaling_vs_density_curve.{fmt}")
        print(f"Writing to: {fpath.resolve().absolute()}")
        plt.savefig(fpath, dpi=dpi, format=fmt)


if __name__ == "__main__":
    main(
        save=True,
    )
