import numpy as np
import pandas as pd
from scipy import stats
import lateral_signaling as lsig

lsig.viz.default_rcParams()

import matplotlib.pyplot as plt
import seaborn as sns


def main(
    metadata_csv=lsig.data_dir.joinpath("FACS", "perturbations", "metadata.csv"),
    FACS_dir=lsig.data_dir.joinpath("FACS", "senders"),
    figsize=(5, 3),
    save_dir=lsig.plot_dir,
    save=False,
    dpi=300,
    fmt="png",
):

    # data = pd.read_csv(bs_means_csv)
    # mean_data = data.groupby("Density").agg(np.mean)
    # mean_x_col = next(c for c in mean_data.columns if "FITC" in c)
    # mean_y_col = next(c for c in mean_data.columns if "FSC" in c)

    # config = dict()
    # fig = plt.figure(figsize=figsize)
    # kde = sns.kdeplot(data=data, x="FITC", y="FSC", hue="Density", levels=[level])
    # sct = sns.scatterplot(data=mean_data, x=mean_x_col, y=mean_y_col)

    # plt.xlabel("GFP")
    # # plt.xlim([0, 4000])

    metadata = pd.read_csv(metadata_csv)
    file_to_density = {f: d for f, d in zip(metadata.filename, metadata.Density)}

    dfs = []
    summary_dfs = []
    for f in sorted(FACS_dir.glob("*.csv")):
        _density = file_to_density[f.name]
        _fsc, _fitc = pd.read_csv(f)[["FSC-A", "FITC-A"]].values.T
        size = _fsc.shape[0]

        _df = pd.DataFrame(
            {
                "Density": _density,
                "FSC-A": _fsc,
                "FITC-A": _fitc,
            }
        )
        dfs.append(_df)

        _summary_df = pd.DataFrame(
            {
                "Density": _density,
                "FSC": _fsc.mean(),
                "FSC_sem": _fsc.std() / np.sqrt(size),
                "FITC": _fitc.mean(),
                "FITC_sem": _fitc.std() / np.sqrt(size),
            },
            index=[f.name],
        )
        summary_dfs.append(_summary_df)

    data = pd.concat(dfs).sort_values("Density")
    summary_data = pd.concat(summary_dfs).sort_values("Density")

    fig = plt.figure(figsize=figsize)
    err = plt.errorbar(
        x=summary_data.FITC,
        xerr=summary_data.FITC_sem,
        y=summary_data.FSC,
        yerr=summary_data.FSC_sem,
        capsize=2,
        fmt="none",
        zorder=0,
    )
    pts = sns.scatterplot(
        data=summary_data,
        x="FITC",
        y="FSC",
        hue="Density",
    )

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        data["FITC-A"], data["FSC-A"]
    )
    r2 = 100 * r_value ** 2

    linreg = sns.regplot(
        data=data,
        x="FITC-A",
        y="FSC-A",
        ci=None,
        scatter=False,
        truncate=True,
        line_kws=dict(
            ls="dashed",
            color="k",
            lw=1,
        ),
    )

    s_intercept = f"{intercept:.1e}"
    where_modify = s_intercept.find("e")
    s_intercept = (
        f"{s_intercept[:where_modify]}\mathrm{{e}}{int(s_intercept[where_modify+1:])}"
    )
    eqn = fr"$y={{{slope:.1f}}}x + {{{s_intercept}}}$" "\n" fr"$r={{{r_value:.3f}}}$"
    plt.text(3000, 128000, eqn, ha="center", va="top")

    sns.despine()
    plt.title("Correlation of mean GFP and mean cell size")
    plt.legend(title="Density", loc="center left", bbox_to_anchor=(1.02, 0.5))
    # sns.move_legend(plt.gca(), "right")

    plt.xlabel("GFP")
    plt.xlim(1000, 4000)
    plt.ylabel("FSC")
    plt.ylim(115000, 155000)

    plt.tight_layout()

    if save:
        fname = save_dir.joinpath(f"FSC_GFP_scatterplot_stderr.{fmt}")
        print("Writing to:", fname.resolve().absolute())
        plt.savefig(fname, dpi=dpi, facecolor="w")


if __name__ == "__main__":

    main(
        save=True,
    )

    # mp_main = partial(
    #     main,
    #     save_dir=lsig.temp_plot_dir,
    #     save=True,
    # )

    # levels = (0.4, 0.5, 0.6, 0.7)
    # with mp.Pool(cpu_count(logical=True)) as pool:
    #     results = pool.map(mp_main, levels)
