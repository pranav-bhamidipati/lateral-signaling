import lateral_signaling as lsig

import os
import json

import numpy as np
import os

import holoviews as hv

hv.extension("matplotlib")

import matplotlib.pyplot as plt

lsig.default_rcParams()


mle_params_csv = os.path.abspath("../data/analysis/growth_parameters_MLE.csv")
phase_examples_json = os.path.abspath("../data/simulations/phase_examples.json")

save_dir = os.path.abspath("../plots/tmp")
plot_prefix = os.path.join(save_dir, "logistic_equation")


def main(
    tmin=0.0,
    tmax=3.25,
    rho_crit=3.0,
    plot_prefix=plot_prefix,
    phase_examples_json=phase_examples_json,
    figsize=(2.3, 2.0),
    save=False,
    fmt="png",
    dpi=300,
):

    for fname in (mle_params_csv, phase_examples_json):
        assert os.path.exists(fname), f"File does not exist: {fname}"

    # Make an illustrative example of logistic growth with made-up parameters
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

    # example_curve = hv.Curve((t_, logistic_example)).opts(
    #     linewidth=6,
    #     c="k",
    #     xticks=0,
    #     xlabel="Time",
    #     ylim=(0, None),
    #     yticks=0,
    #     ylabel="Density",
    #     fontscale=2,
    #     aspect=1.2,
    # )
    # example_rhomax = hv.HLine(10).opts(
    #     linestyle="dashed",
    #     c="k",
    #     linewidth=3,
    # )
    # example_plot = example_curve * example_rhomax

    if save:
        example_fname = f"{plot_prefix}_example.{fmt}"
        print(f"Writing to: {example_fname}")

        # hv.save(example_plot, example_fname, dpi=dpi, fmt=fmt)
        plt.savefig(example_fname, format=fmt, dpi=dpi)

    # Get parameters for an example from each phase of the signaling phase diagram (Figure 4)
    with open(phase_examples_json, "r") as f:
        j = json.load(f)
        phase_names = j["name"]
        rho_0s = j["rho_0"]
        gs = j["g"]

    rho_max = lsig.mle_params.rho_max_ratio

    t = np.linspace(tmin, tmax, 201)
    curve_data = np.array(
        [lsig.logistic(t, g, rho_0, rho_max) for g, rho_0 in zip(gs, rho_0s)]
    )

    # t_crit = t[np.searchsorted(curve_data[phase_names.index("Limited")], rho_crit)]

    colors = lsig.cols_blue[::-1]
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
        colors=lsig.gray,
        linewidth=3.0,
    )
    plt.text(0.8 * (tmax - tmin), rho_crit + 0.5, r"$\rho_\mathrm{crit}$")

    plt.tight_layout()

    # color_cycle = hv.Cycle(colors)
    # hv_opts = dict(
    #     xlabel="time",
    #     xlim=(0, tmax),
    #     xticks=0,
    #     ylabel="density",
    #     ylim=(0.5, 6.5),
    #     yticks=0,
    #     fontscale=2,
    #     show_legend=False,
    #     aspect=1.2,
    #     color=color_cycle,
    # )
    #
    # growth_curves = (
    #     hv.Curve(data, kdims=["t"], vdims=["density", "case"])
    #     .groupby("case")
    #     .opts(
    #         color=hv.Cycle(lsig.cols_blue[::-1]),
    #         linewidth=6,
    #     )
    #     .overlay(
    #         # ).options(
    #         #     {"Curve": dict(color=cycle)}
    #     )
    #     .opts(
    #     )
    # )

    if save:
        phase_examples_fname = f"{plot_prefix}_phase_examples.{fmt}"
        print(f"Writing to: {phase_examples_fname}")

        # hv.save(phase_examples_plot, phase_examples_fname, dpi=dpi, fmt=fmt)
        plt.savefig(phase_examples_fname, format=fmt, dpi=dpi)


if __name__ == "__main__":
    main(
        save=True,
    )
