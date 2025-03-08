#!/usr/bin/env python3
from datetime import datetime

import matplotlib as mpl
import numpy as np
import pandas as pd
from astropy.timeseries import TimeSeries
from jarvis import fpath, get_obs_interval, hst_fpath_list
from jarvis.utils import group_to_visit
from matplotlib import pyplot as plt
from tqdm import tqdm

infiles = [fpath("2025-03-06_17-30-59.txt"), fpath("2025-03-06_21-14-46.txt")]


def plot_visits(
    df, quantity="PFlux", corrected=None, ret="showsavefig", unit=None,
):  # corrected = False, True, None (remove negative values)
    df["EPOCH"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.sort_values(by="EPOCH")
    if "color" not in df.columns:
        df["color"] = df[quantity].apply(lambda x: "r" if x < 0 else "b")
    corr_df = df.copy()
    if corrected is True:
        corr_df[quantity] = df[quantity].where(df[quantity] > 0, 0)
    elif corrected is None:
        corr_df[quantity] = df[quantity].where(df[quantity] > 0, np.nan)
    visits = df["visit"].unique()
    fitsdirs = hst_fpath_list()[:-1]
    fig = plt.figure(figsize=(19.2, 10.8), dpi=70)

    # this generates the grid dimensions based on the number of visits and a maximum of 8 columns
    num_visits = len(visits)
    jrows = 1
    while True:
        icols = num_visits // jrows
        if num_visits % jrows == 0 and icols <= 8:
            break
        jrows += 1

    tqdm.write(f"J={jrows}, I={icols}")
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 5], hspace=0.5, wspace=0)  # hspace=0,
    mgs = gs[1].subgridspec(jrows, icols, height_ratios=[1 for _ in range(jrows)], wspace=0)  # hspace=0,wspace=0,
    main_ax = fig.add_subplot(gs[0])
    main_ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval=2))
    main_ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d/%m/%y"))
    main_ax.xaxis.set_minor_locator(mpl.dates.DayLocator(interval=1))
    axs = [[fig.add_subplot(mgs[j, i], label=f"v{visits[i+icols*j]}") for i in range(icols)] for j in range(jrows)]
    for f in fitsdirs:
        mind, maxd = get_obs_interval(f)
        mind = pd.to_datetime(mind)
        maxd = pd.to_datetime(maxd)
        # span over the interval
        main_ax.axvspan(mind, maxd, alpha=0.5, color="#aaa5")
    ylim = [corr_df["PFlux"].min(), corr_df["PFlux"].max()]
    diff = ylim[1] - ylim[0]
    yl = [ylim[0] + diff / 10, ylim[1] - diff / 10]
    for visit in visits:
        dfa = corr_df.loc[df["visit"] == visit]
        mean_d = dfa["EPOCH"].mean()
        if dfa[quantity].mean() > corr_df[quantity].mean():
            main_ax.text(mean_d, yl[0], visit, fontsize=10, color="black", va="top", ha="right", rotation=90)
        else:
            main_ax.text(mean_d, yl[1], visit, fontsize=10, color="black", va="bottom", ha="right", rotation=90)
    main_ax.scatter(corr_df["EPOCH"], corr_df[quantity], marker="x", s=5, c=corr_df["color"])
    main_ax.xaxis.tick_top()
    for j in range(jrows):
        for i in range(icols):
            dfa = corr_df.loc[df["visit"] == visits[i + icols * j]]
            axs[j][i].plot(dfa["EPOCH"], dfa[quantity], color="#77f", linewidth=0.5)
            axs[j][i].scatter(dfa["EPOCH"], dfa[quantity], marker=".", s=5, c=dfa["color"])
            axs[j][i].yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useOffset=False))
            idfint = df.where(df["visit"] == visits[i + icols * j])
            dmax, dmin = [idfint["EPOCH"].max(), idfint["EPOCH"].min()]
            delta = dmax - dmin
            axs[j][i].set_xlim([dmin - 0.01 * delta, dmax + 0.01 * delta])
            axs[j][i].xaxis.set_major_locator(
                mpl.dates.MinuteLocator(interval=int(np.floor((delta.total_seconds() / 60) / 3))),
            )
            axs[j][i].xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
            axs[j][i].annotate(
                f"v{visits[i+icols*j]}",
                xy=(0, 1),
                xycoords="axes fraction",
                xytext=(+0.5, -0.5),
                textcoords="offset fontsize",
                fontsize="medium",
                verticalalignment="top",
                weight="bold",
                bbox={"facecolor": "#0000", "edgecolor": "none", "pad": 3.0},
            )
            axs[j][i].annotate(
                f'{dmin.strftime("%d/%m/%y")}',
                xy=(1, 1),
                xycoords="axes fraction",
                xytext=(-0.05, +0.1),
                textcoords="offset fontsize",
                fontsize="medium",
                verticalalignment="bottom",
                horizontalalignment="right",
                annotation_clip=False,
                bbox={"facecolor": "#0000", "edgecolor": "none", "pad": 3.0},
            )
            axs[j][i].tick_params(axis="both", pad=0)
    # for i in range(I):
    #     axs[0][i].xaxis.tick_top()
    #     axs[0][i].xaxis.set_label_position('top')
    for j in range(jrows):
        axs[j][0].set_ylabel(f"{quantity}" + (f" [{unit}]" if unit else ""))
    main_ax.set_ylabel(f"{quantity}" + (f" [{unit}]" if unit else ""))
    fig.suptitle(f'{quantity} over visits {df['visit'].min()} to {df['visit'].max()}', fontsize=20)
    if "save" in ret:
        plt.savefig(fpath(f'figures/imgs/{quantity}_{datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}.png'))
    if "show" in ret:
        plt.show()
    if "fig" in ret:
        return fig
    return None


def plot_visits_multi(
    dfs, quantity="PFlux", ret="showsavefig", unit=None, figure=None,
):  # corrected = False, True, None (remove negative values)
    corr_, visits_ = [], []
    for df in dfs:
        df["EPOCH"] = pd.to_datetime(df["Date"] + " " + df["Time"])
        df = df.sort_values(by="EPOCH")
        if "color" not in df.columns:
            df["color"] = df[quantity].apply(lambda x: "r" if x < 0 else "b")
        visits = df["visit"].unique()
        corr_.append(df)
        visits_.append(visits)
    visits = set()
    for visit in visits_:
        for v in visit:
            if not any(v == u for u in visits):
                visits.update({v})
    df = dfs[0].copy()
    for i, df_ in enumerate(dfs[1:]):
        df = pd.merge(
            df,
            df_,
            how="outer",
            on=["visit", "Date", "Time", "Power", "PFlux", "Area", "EPOCH", "color"],
            suffixes=("", f"_{i+1}"),
        )

    fitsdirs = hst_fpath_list()[:-1]
    # this generates the grid dimensions based on the number of visits and a maximum of 8 columns
    num_visits = len(visits)
    # J = 1
    for colguess in [5, 6, 7, 4, 8]:
        if num_visits % colguess == 0:
            icols, jrows = colguess, num_visits // colguess
            break
    else:
        for colguess in [5, 6, 7, 4, 8]:
            if num_visits % colguess < colguess // 2:
                icols, jrows = colguess + 1, num_visits // colguess
                break
        else:
            icols, jrows = num_visits // 2 + 1, 2
    tqdm.write(f"{jrows=}, {icols=}, {visits=}")
    fig = plt.figure(figsize=(19.2, 10.8), dpi=70) if figure is None else figure

    gs = fig.add_gridspec(2, 1, height_ratios=[4, 5], hspace=0.1)  # hspace=0,wspace=0,

    main_ax = fig.add_subplot(gs[0])
    main_ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval=2))
    main_ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d/%m/%y"))
    main_ax.xaxis.set_minor_locator(mpl.dates.DayLocator(interval=1))
    uniquevisits = []
    for f in fitsdirs:
        mind, maxd = get_obs_interval(f)
        mind = pd.to_datetime(mind)
        maxd = pd.to_datetime(maxd)
        # span over the interval
        fvisit = group_to_visit(int(f.split("_")[-1][:2]))
        if fvisit in visits:
            col = "#aaa"
            main_ax.annotate(
                f"v{fvisit}",
                xy=(mind, 0),
                xytext=(mind, 0),
                fontsize=10,
                color="black",
                va="top",
                ha="center",
                rotation=90,
            )
            uniquevisits.append(fvisit)
        else:
            col = "#aaa5"
        main_ax.axvspan(mind, maxd, alpha=0.5, color=col)

    mgs = gs[1].subgridspec(jrows, icols, height_ratios=[1 for _ in range(jrows)])  # hspace=0,wspace=0,
    axs = [
        [
            fig.add_subplot(mgs[j, i], label=f"v{uniquevisits[i+icols*j]}") if i + icols * j < num_visits else None
            for i in range(icols)
        ]
        for j in range(jrows)
    ]

    for cdf in corr_:
        main_ax.scatter(cdf["EPOCH"], cdf[quantity], marker="x", s=5, c=cdf["color"], zorder=5)
        main_ax.plot(cdf["EPOCH"], cdf[quantity], color="#aaf", linewidth=0.5, linestyle="--")
    visits = []
    ranges = {}
    main_ax.xaxis.tick_top()
    for i in range(icols):
        for j in range(jrows):
            if i + icols * j < num_visits:
                for cdf in corr_:
                    dfa = cdf.loc[cdf["visit"] == uniquevisits[i + icols * j]]
                    if not dfa.empty:
                        axs[j][i].plot(dfa["EPOCH"], dfa[quantity], color="#aaf", linewidth=0.5)
                        axs[j][i].scatter(
                            dfa["EPOCH"], dfa[quantity], marker=".", s=5, color=dfa["color"].tolist()[0], zorder=5,
                        )
                axs[j][i].yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
                idfint = df.where(df["visit"] == uniquevisits[i + icols * j])
                dmax, dmin = [idfint["EPOCH"].max(), idfint["EPOCH"].min()]
                delta = dmax - dmin
                axs[j][i].set_xlim([dmin - 0.01 * delta, dmax + 0.01 * delta])
                axs[j][i].xaxis.set_major_locator(
                    mpl.dates.MinuteLocator(interval=int(np.floor((delta.total_seconds() / 60) / 3))),
                )
                axs[j][i].xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
                axs[j][i].annotate(
                    f"v{uniquevisits[i+icols*j]}",
                    xy=(0, 1),
                    xycoords="axes fraction",
                    xytext=(+0.5, -0.5),
                    textcoords="offset fontsize",
                    fontsize="medium",
                    verticalalignment="top",
                    weight="bold",
                    bbox={"facecolor": "#0000", "edgecolor": "none", "pad": 3.0},
                )
                axs[j][i].annotate(
                    f'{dmin.strftime("%d/%m/%y")}',
                    xy=(1, 1),
                    xycoords="axes fraction",
                    xytext=(-0.05, +0.1),
                    textcoords="offset fontsize",
                    fontsize="medium",
                    verticalalignment="bottom",
                    horizontalalignment="right",
                    annotation_clip=False,
                    bbox={"facecolor": "#0000", "edgecolor": "none", "pad": 3.0},
                )
                axs[j][i].tick_params(axis="both", pad=0)
                qmin, qmax = [idfint[quantity].min(), idfint[quantity].max()]
                axs[j][i].set_ylim(
                    [max(0, qmin - 0.1 * (qmax - qmin)), min(qmax + 0.1 * (qmax - qmin), df[quantity].max())],
                )
                ranges[i + icols * j] = dmin, dmax, qmin, qmax

    arr = []
    for i, [dmn, dmx, mn, mx] in ranges.items():
        arr.extend([[dmn, mn, mx], [dmx, mn, mx]])
    mmdf = pd.DataFrame(arr, columns=["EPOCH", "min", "max"])
    mmdf = mmdf.dropna()

    mmdf = mmdf.sort_values(by=["EPOCH"])
    main_ax.fill_between(mmdf["EPOCH"], mmdf["min"], mmdf["max"], color="#aaa5", alpha=0.5, interpolate=True)
    for j in range(jrows):
        axs[j][0].set_ylabel(f"{quantity}" + (f" [{unit}]" if unit else ""))
    main_ax.set_ylabel(f"{quantity}" + (f" [{unit}]" if unit else ""))
    main_ax.set_ylim([0, df[quantity].max()])
    for i in range(icols):
        for j in range(jrows):
            if i + icols * j < num_visits:
                axs[j][i].set_ylim([0, df[quantity].max()])

    main_ax.set_title(f'{quantity} over visits {df['visit'].min()} to {df['visit'].max()}', fontsize=20)
    if "save" in ret and figure is None:
        plt.savefig(fpath(f'figures/imgs/{quantity}_{datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}.png'))
    if "show" in ret:
        plt.show()
    if "fig" in ret:
        return fig, main_ax, axs, (df["EPOCH"].min(), df["EPOCH"].max())
    return None


dfs = [
    pd.read_csv(f, sep=" ", index_col=False, names=["visit", "Date", "Time", "Power", "PFlux", "Area"]) for f in infiles
]
clrs = ["r", "g", "b"]  # cmr.take_cmap_colors('rainbow', len(dfs), return_fmt='hex')
for i, d in enumerate(dfs):
    dfs[i]["color"] = [clrs[i] for _ in range(len(dfs[i]))]
fig = plt.figure(figsize=(8, 6), constrained_layout=True)
# subfig = fig.subfigures(1,3, wspace=0, )

table = TimeSeries.read(fpath("datasets/Hisaki_SW-combined.csv"))
# remove nan values
table = table.to_pandas()
table_sw = table.loc[np.isfinite(table["jup_sw_pdyn"])]
table_sw = TimeSeries().from_pandas(table_sw)
table_torus = table.loc[np.isfinite(table["TPOW0710ADAWN"])]
table_torus = TimeSeries().from_pandas(table_torus)
table = TimeSeries().from_pandas(table)
# plot_visits_multi(dfs, 'PFlux', unit='GW/km²',figure=subfig[0], ret='fig')
# plot_visits_multi(dfs, 'Power', unit='GW', figure=subfig[1], ret='fig')
# plot_visits_multi(dfs, 'Area', unit='km²', figure=subfig[2], ret='fig')
fig, ax, axs, lims = plot_visits_multi(dfs, "PFlux", unit="GW/km²", figure=fig, ret="fig")
for m in axs + [[ax]]:
    for a in m:
        ax0t = a.twinx()
        ax0t.plot(table_sw.time.datetime64, table_sw["jup_sw_pdyn"], color="#f88", linewidth=0.7, linestyle="--")
        ax0t.plot(table.time.datetime64, table["jup_sw_pdyn"], color="red", linewidth=0.7)
        ax0t.scatter(table.time.datetime64, table["jup_sw_pdyn"], color="red", marker=".", s=5)
        ax0tt = a.twinx()
        ax0tt.plot(
            table_torus.time.datetime64, table_torus["TPOW0710ADAWN"], color="#88f", linewidth=0.7, linestyle="--",
        )
        ax0tt.plot(table.time.datetime64, table["TPOW0710ADAWN"], color="blue", linewidth=0.7)
        ax0tt.scatter(table.time.datetime64, table["TPOW0710ADAWN"], color="blue", marker=".", s=5)
ax.set_xlim(lims[0] - pd.Timedelta(hours=1), lims[1] + pd.Timedelta(hours=1))

# print(lims, t, table.columns)


plt.show()

# testfits = fits.open(fpath('datasets/HST/group_13/jup_16-148-17-19-53_0100_v16_stis_f25srf2_proj.fits'))
#
# df = table.to_pandas()
# df.plot(x='EPOCH',y='TPOW0710ADAWN')
# plt.show()

# get the first color value:
