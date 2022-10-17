import json
import numpy as np
import matplotlib.pyplot as plt
import holoviews as hv

hv.extension("matplotlib")

import lateral_signaling as lsig

lsig.viz.default_rcParams()


mle_params_csv = lsig.analysis_dir.joinpath("growth_parameters_MLE.csv")
phase_examples_json = lsig.simulation_dir.joinpath("phase_examples.json")


def main(
    tmin=0.0,
    tmax=3.25,
    rho_crit=3.0,
    prefix="logistic_equation",
    phase_examples_json=phase_examples_json,
    figsize=(2.3, 2.0),
    save_dir=lsig.plot_dir,
    save=False,
    fmt="png",
    dpi=300,
):

    # Make an illustrative example of logistic growth with exemplary parameters
    _tmin = 0
    _tmax = 5
    _t = np.linspace(_tmin, _tmax, 201)
    logistic_example = lsig.logistic(_t, 1, 1, 10)

    fig1 = plt.figure(1, figsize=figsize)
    ax = plt.gca()
    ax.set(
        xlabel="Time",
        xlim=(_tmin, _tmax),
        xticks=(),
        ylabel="Density",
        ylim=(0, 11),
        yticks=(1.0,),
        yticklabels=(r"$\rho_0$",),
    )
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.plot(
        _t,
        logistic_example,
        linewidth=4,
        color="k",
    )
    plt.hlines(
        10,
        _tmin,
        _tmax,
        linestyles="dashed",
        colors="k",
        linewidth=2,
    )
    # plt.text(-0.25, 1, r"$\rho_0$", ha="right")
    plt.text(4.0, 10.75, r"$\rho_\mathrm{max}$", ha="right", va="baseline")

    plt.tight_layout()

    if save:
        fname = save_dir.joinpath(f"{prefix}_example.{fmt}")
        print(f"Writing to: {fname.resolve().absolute()}")
        plt.savefig(fname, format=fmt, dpi=dpi)

    # Get parameters for an example from each phase of the signaling phase diagram (Figure 4)
    with phase_examples_json.open("r") as f:
        j = json.load(f)
        rho_0s = j["rho_0"]
        gs = j["g"]

    rho_max = lsig.mle_params.rho_max_ratio

    t = np.linspace(tmin, tmax, 201)
    curve_data = np.array(
        [lsig.logistic(t, g, rho_0, rho_max) for g, rho_0 in zip(gs, rho_0s)]
    )

    colors = lsig.viz.cols_blue[::-1]
    opts = dict(
        xlabel="Time",
        xlim=(0, tmax),
        xticks=(),
        ylabel="Density",
        ylim=(0.5, 6.5),
        yticks=(),
    )

    fig2 = plt.figure(2, figsize=figsize)
    ax = plt.gca()
    ax.set(**opts)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    for data, clr in zip(curve_data, colors):
        plt.plot(
            t,
            data,
            color=clr,
            linewidth=3.0,
        )

    plt.hlines(
        rho_crit,
        t.min(),
        t.max(),
        linestyles="dotted",
        colors=lsig.viz.gray,
        linewidth=3.0,
    )
    plt.text(0.8 * (tmax - tmin), rho_crit + 0.5, r"$\rho_\mathrm{crit}$")

    plt.tight_layout()

    if save:
        fname = save_dir.joinpath(f"{prefix}_phase_examples.{fmt}")
        print(f"Writing to: {fname.resolve().absolute()}")
        plt.savefig(fname, dpi=dpi)


if __name__ == "__main__":
    main(
        save=True,
    )
