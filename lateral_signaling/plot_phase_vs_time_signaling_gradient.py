import json
import h5py
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import lateral_signaling as lsig

lsig.default_rcParams()

sim_dir = Path("../data/simulations")
sacred_dir = sim_dir.joinpath("20220818_phase_linrho/sacred")
# sacred_dir = Path("./sacred")
# sacred_dir = sim_dir.joinpath("20211209_phase_2D/sacred")
save_dir = Path("../plots/tmp")


def main(
    grad_t=3.7,
    grad_lo=2.0,
    grad_hi=5.0,
    figsize=(3, 3),
    sim_dir=sim_dir,
    sacred_dir=sacred_dir,
    save_dir=save_dir,
    save=False,
    dpi=300,
    fmt="png",
    bg_color="w",
    atol=1e-6,
):

    with sim_dir.joinpath("phase_threshold.json").open("r") as f:
        j = json.load(f)
        v_init_thresh = j["v_init_thresh"]

    data_dirs = list(sacred_dir.glob("[0-9]*"))
    data_dirs = [d for d in data_dirs if d.joinpath("config.json").exists()]
    d0 = data_dirs[0]
        
    with open(d0.joinpath("config.json"), "r") as f:
        j = json.load(f)
        delay = j["delay"]
        nt_t_save = j["nt_t_save"]
        rho_max = j["rho_max"]
        g_space = np.asarray(j["g_space"])

    with h5py.File(d0.joinpath("results.hdf5"), "r") as h:
        t = np.asarray(h["t"])
        # print(*list(h.keys()), sep="\n") 
    
    nt = t.size
    delay_days = lsig.t_to_units(delay)
    t_days = lsig.t_to_units(t)
    step_delay = int(delay_days * nt_t_save)
    
    g_idx = np.isclose(g_space, 1.0).nonzero()[0][0]

    rho_0s = []
    actnum_ts = []
    v_inits = []
    for d in data_dirs:

        with open(d.joinpath("config.json"), "r") as f:
            j = json.load(f)
            rho_0 = j["rho_0"]

        if not (1.0 - atol <= rho_0 <= rho_max + atol):
            continue

        with h5py.File(d.joinpath("results.hdf5"), "r") as h:
            actnum_t = np.asarray(h["S_t_g_actnum"])[g_idx]
            v_init = np.asarray(h["v_init_g"])[g_idx]

        rho_0s.append(rho_0)
        actnum_ts.append(actnum_t)
        v_inits.append(v_init)

    sort_idx = np.argsort(rho_0s)
    rho_space = np.asarray(rho_0s)[sort_idx]
    actnum_t_rho_0 = np.asarray(actnum_ts)[sort_idx]
    
    init_rho_0    = (np.asarray(v_inits)[sort_idx] > v_init_thresh)
    activated_t   = (actnum_t_rho_0[:, :] > 0).astype(int)
    deactivated_t = np.zeros(activated_t.shape, dtype=int)
    for i, _ in enumerate(rho_space):
        if init_rho_0[i]:
            deactivate_idx = np.where(np.diff(activated_t[i]) < 0)[0]
            if deactivate_idx.size > 0:
                deactivated_t[i, deactivate_idx[0]:] = 1
    
    phase_t = init_rho_0[:, np.newaxis] * (deactivated_t + 2 * activated_t)

    # g_inv_days = lsig.mle_params.g_inv_days
    # t_OFF_days = lsig.get_t_OFF(g_inv_days, rho_space)
    # t_ON_days = lsig.get_t_ON(g_inv_days, rho_space)
    # turns_ON = (t_OFF_days > 0).astype(int)
    # # turns_OFF_t = (actnum_t_rho_0[:, :-step_delay] > 0).astype(int)
    # turned_OFF_t = np.greater.outer(t_OFF_days + delay_days, t_days).astype(int)
    # phase_t = turns_ON[:, np.newaxis] + turned_OFF_t 

    phase_cmap = mpl.colors.ListedColormap(lsig.cols_blue[::-1])

    dr = rho_space[1] - rho_space[0]
    dt = t_days[1] - t_days[0]
    extent_r = (rho_space[0] - dr / 2, rho_max + dr / 2)
    extent_t = (t_days[0] - dt / 2, t_days[-1] + dt / 2)

    fig1 = plt.figure(figsize=figsize)
    ax = plt.gca()

    plt.imshow(
        phase_t,
        aspect="auto",
        origin="lower",
        extent=(*extent_t, *extent_r),
        cmap=phase_cmap,
        interpolation="nearest",
    )

    plt.vlines(grad_t, grad_lo, grad_hi, colors="k", lw=2)
    plt.hlines((grad_lo, grad_hi), grad_t - dr * 2, grad_t + dr * 2, colors="k", lw=2)

    plt.hlines(
        rho_max,
        *extent_t,
        linestyles="dashed",
        colors="k",
        lw=2,
    )
    plt.text(-dr, rho_max, r"$\rho_\mathrm{max}$", ha="right", va="center", fontsize=12)

    plt.xlabel("Time", loc="right")
    plt.xticks([])
    plt.ylabel(r"$\rho_0$")
    
    plt.tight_layout()

    if save:
        fname = save_dir.joinpath(f"phase_vs_t_spatialgradient").with_suffix(f".{fmt}")
        print(f"Writing to: {fname.resolve().absolute()}")
        plt.savefig(fname, dpi=dpi, facecolor=bg_color)

    print()
    
    #####
    ## Run the same expreiment with low density also, scanning through rho_0
    ##  on a log-scale.
    ## Make a comparable plot with "error bars" at multiple time-points


if __name__ == "__main__":
    main(
        save=True,
    )
    