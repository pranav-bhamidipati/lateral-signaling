import numpy as np
import matplotlib.pyplot as plt

import lateral_signaling as lsig


def main(
    rhos=[1, 2, 3, 4],
    Xrows=12,
    prows=2,
    pcols=2,
    figsize=(4, 4),
    save_dir=lsig.plot_dir,
    save=False,
    fmt="png",
    dpi=300,
):

    ## Construct data for example
    # make cell sheet
    Xcols = Xrows
    X = lsig.hex_grid(Xrows, Xcols)
    center = lsig.get_center_cells(X)
    X = X - X[center]

    # Store cell type
    var = np.zeros(X.shape[0], dtype=np.float32)
    var[center] = 1.0

    # Set which rhos to use as examples
    rhos = np.array(rhos)
    X_rho = np.multiply.outer(1 / np.sqrt(rhos), X)

    ## Plotting options
    # Set axis limits for inset plots
    _extent = np.min([-X_rho[-1].min(axis=0).max(), X_rho[-1].max(axis=0).min()])
    xlim = -_extent, _extent
    ylim = -_extent, _extent

    fig = plt.figure(figsize=figsize)
    for i, (_rho, _x) in enumerate(zip(rhos, X_rho)):
        _ax = fig.add_subplot(prows, pcols, i + 1)

        lsig.viz.plot_hex_sheet(
            ax=_ax,
            X=_x,
            var=var,
            rho=_rho,
            sender_idx=center,
            xlim=xlim,
            ylim=ylim,
            title=fr"$\rho={{{_rho:.1f}}}$",
            ec="w",
            lw=0.2,
            scalebar=True,
            # sbar_kwargs=dict(font_properties=dict(size=0)),
            # axis_off=False,
        )

    if save:
        fpath = save_dir.joinpath(f"lattice_examples_at_diff_densities.{fmt}")
        print("Writing to:", fpath.resolve().absolute())
        plt.savefig(fpath, dpi=dpi)


if __name__ == "__main__":
    main(
        save_dir=lsig.temp_plot_dir,
        save=True,
    )
