#!/usr/bin/env python3
from datetime import datetime
from glob import glob
from random import choice
import re

import matplotlib as mpl
import numpy as np
import pandas as pd
from astropy.timeseries import TimeSeries
from jarvis import fpath, get_obs_interval, hst_fpath_list
from jarvis.utils import group_to_visit, rpath
from matplotlib import pyplot as plt
from tqdm import tqdm

from jarvis.const import Dirs


mpl.rcParams["hatch.linewidth"] = 2
#infiles =[#fpath(x) for x in [
    #"2025-03-06_17-30-59.txt",
    #"2025-03-06_21-14-46.txt",
    #"2025-03-11T14-04-20_powers_coadds.txt",
    #"2025-03-11T18-34-00_DPR.txt",
    #"2025-03-11T18-34-05_DPR.txt",
    #"2025-03-11T18-35-00_DPR.txt",
    #"2025-03-11T18-36-00_DPR.txt",
    #"powers.txt",
    #]]
# print(str(Dirs.GEN)+"/*.txt")
infiles = glob(str(Dirs.GEN)+"/*.txt")
# print(infiles)
# sort by date on filename, but not specific to where in filename. could be anywhere in it, but in the form YYYY-MM-DDTHH-MM-SS
 # examples include "2025-03-11T18-36-00_DPR.txt", "BOUNDARY_2025-03-12T17-24-01.txt", "BOUNDARY_2025-03-12T23-49-39_window[5].txt"
 # and should deal with not identifying one by placing it at the end of the list. 
 # the globs will also still have the full path.
orde = []
for i in infiles:
    spos = i.find("202")
    if spos == -1:
        orde.append("9")
    else:
        orde.append(i[spos:spos+19])
infiles = [x for _,x in sorted(zip(orde,infiles), reverse=True)]
# print(infiles)

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
            on=["visit", "Date", "Time", "Power", "PFlux", "Area", "EPOCH", "color", "marker", "zorder"],
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
        #main_ax.plot(cdf["EPOCH"], cdf[quantity], color="#aaf", linewidth=0.5, linestyle="--")
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


def plot_visits_v2(
    dfs, axs, quantities):  # corrected = False, True, None (remove negative values)
    corr_, visits_ = [], []
    for df in dfs:
        df["EPOCH"] = pd.to_datetime(df["Date"] + " " + df["Time"])
        df = df.sort_values(by="EPOCH")
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
            on=["visit", "Date", "Time", "Power", "PFlux", "Area", "EPOCH", "color","marker", "zorder"],
            suffixes=("", f"_{i+1}"),
        )
    fitsdirs = hst_fpath_list()[:-1]
    uniquevisits = []
    alt_xs = []
    for f in fitsdirs:
        mind, maxd = get_obs_interval(f)
        mind = pd.to_datetime(mind)
        maxd = pd.to_datetime(maxd)
        # span over the interval
        fvisit = group_to_visit(int(f.split("_")[-1][:2]))
        if fvisit in visits:
            col = "#aaa"
            alt_xs.append([mind+(maxd-mind)/2,fvisit])
            uniquevisits.append(fvisit)
        else:
            col = "#aaa5"
        for ax in axs:
            ax.axvspan(mind, maxd, alpha=0.5, color=col)

    #top plot
    axs[-1].set_xticks([a[0] for a in alt_xs], labels=[f"v{a[1]}" for a in alt_xs])
    hch = ["\\\\","//","--","||","oo","xx","**"]+["\\\\","//","--","||","oo","xx","**"]+["\\\\","//","--","||","oo","xx","**"]
    # main scatter&plot loop
    for x,cdf in enumerate(corr_):
        for i in range(len(quantities)):
            if quantities[i] is not None:
                axs[i].scatter(cdf["EPOCH"], cdf[quantities[i]], marker=cdf["marker"][0], s=5, c=cdf["color"], zorder=cdf["zorder"][0])
                #axs[i].plot(cdf["EPOCH"], cdf[quantities[i]], color="#aaf", linewidth=0.5, linestyle="--", zorder=cdf["zorder"][0])
                # define minimums and maximums per visit, get minquant, maxquant, tmin, tmax.
                ranges = []
                for j,u in enumerate(uniquevisits):
                    if u in cdf["visit"].unique():
                        idfint = df.where(df["visit"] == u)
                        dmax, dmin = [idfint["EPOCH"].max(), idfint["EPOCH"].min()]
                        qmin, qmax = [idfint[quantities[i]].min(), idfint[quantities[i]].max()]
                        ranges.append([dmin, qmin, qmax])
                        ranges.append([dmax, qmin, qmax])
                        # print(dmin, dmax, qmin, qmax)

                ranges=pd.DataFrame(ranges, columns=["time","min","max"])
                ranges = ranges.dropna()
                ranges = ranges.sort_values(by=["time"])
                
                
                fl = axs[i].fill_between(ranges['time'],  ranges["max"], ranges["min"],facecolor=(cdf["color"][0],0.05), edgecolor=(cdf["color"][0],0.2),hatch=hch[x],linewidth=0.5,  interpolate=True, zorder=cdf["zorder"][0])
                #axs[i].fill_between(ranges['time'],  ranges["max"], ranges["min"],facecolor=(0,0,0,0), edgecolor=(1,1,1), interpolate=True, zorder=cdf["zorder"][0]+1)
                
               
                
                
    # define lims per visit, per quantity
    ranges = {q:{} for q in quantities if q is not None}
    for quantity in quantities:
        if quantity is not None:
            for i,_ in enumerate(uniquevisits):
                idfint = df.where(df["visit"] == uniquevisits[i])
                dmax, dmin = [idfint["EPOCH"].max(), idfint["EPOCH"].min()]
                qmin, qmax = [idfint[quantity].min(), idfint[quantity].max()]
                ranges[quantity][i] = dmin, dmax, qmin, qmax
    visits = []
    lim = {q : [] for q in quantities if q is not None}
    for q in quantities:
        for i, [dmn, dmx, mn, mx] in ranges[q].items():
            lim[q].extend([[dmn, mn, mx], [dmx, mn, mx]])

        mmdf = pd.DataFrame(lim[q], columns=["EPOCH", "min", "max"])
        mmdf = mmdf.dropna()
        mmdf = mmdf.sort_values(by=["EPOCH"])
        lim[q] = mmdf
    # finally plot the fill_between
    for i,q in enumerate(quantities):
        if quantities[i] is not None:
            #axs[i].fill_between(lim[q]["EPOCH"], lim[q]["min"], lim[q]["max"], color="#aaa5", alpha=0.5, interpolate=True)
            axs[i].set_ylim([0, df[quantities[i]].max()])

       
    return axes, (df["EPOCH"].min(), df["EPOCH"].max()), lim

def plot_discontinuous_angle(ax,ts, colname, threshold=180,  **kwargs):
    time = ts.time
    angles = ts[colname]    # Compute differences to identify discontinuities
    diffs = np.abs(np.diff(angles))
    discontinuities = np.where(diffs > threshold)[0]    # Split indices at discontinuities
    segments = np.split(np.arange(len(angles)), discontinuities + 1)
    for segment in segments:
        t_seg = time[segment]
        ang_seg = angles[segment]# Extend segment at both ends if possible
        if len(segment) > 1:
            t_extended = time[[segment[0], *segment, segment[-1]]]  # Keep as Time object
            ang_extended = np.array([ang_seg[0], *ang_seg, ang_seg[-1]])
        else:
            t_extended, ang_extended = t_seg, ang_seg
        ax.plot(t_extended.datetime64, ang_extended, **kwargs)
    ax.set_ylim([0, 360])
    ax.yaxis.set_major_locator(plt.MultipleLocator(90))

dfs = [
    pd.read_csv(f, sep=" ", index_col=False, names=["visit", "Date", "Time", "Power", "PFlux", "Area","EXTNAME","lmin","lmax","NUMPTS"]) for f in infiles
]
dfs = [df for df in dfs if len(df) > 0]

clrs = ["#0ff","#0aa","#0af","#00f","#a0f","#f0f","#800","#d00","#ff5500","#fa0","#ff0","#0f0","#070","#040"]  # cmr.take_cmap_colors('rainbow', len(dfs), return_fmt='hex')
if len(clrs)<len(dfs):
    diff = len(dfs)-len(clrs)
    for i in range(diff):
        clrs.insert(0,"#555")
    
mkrs = ["1","2","3","4","x","+","1","2","3","4","x","+",]                    # cmr.take_cmap_colors('rainbow', len(dfs), return_fmt='hex')
for i, d in enumerate(dfs):
    dfs[i]["color"] = [clrs.pop(-1),] * len(dfs[i])
    #clrs.remove(dfs[i]["color"][0])
    dfs[i]["marker"] = [choice(mkrs),] * len(dfs[i])
    dfs[i]["zorder"] = [len(dfs)-i+2,] * len(dfs[i])
    #mkrs.remove(dfs[i]["marker"][0])
    # for each df, make any values < 0  equal 0
    # dfs[i]["PFlux"] = dfs[i]["PFlux"].where(dfs[i]["PFlux"] > 0, 0)
    # dfs[i]["Power"] = dfs[i]["Power"].where(dfs[i]["Power"] > 0, 0)
    # dfs[i]["Area"] = dfs[i]["Area"].where(dfs[i]["Area"] > 0, 0)
# fig = plt.figure(figsize=(8, 6), constrained_layout=True)
# subfig = fig.subfigures(1,3, wspace=0, )  # noqa: ERA001

table = TimeSeries.read(fpath("datasets/Hisaki_SW-combined.csv"))
def hisaki_sw_get_safe(col):
    return TimeSeries().from_pandas(table.to_pandas().loc[np.isfinite(table[col])])
# remove nan values
table = table.to_pandas()
table_sw = table.loc[np.isfinite(table["jup_sw_pdyn"])]
table_sw = TimeSeries().from_pandas(table_sw)
table_torus = table.loc[np.isfinite(table["TPOW0710ADAWN"])]
table_torus = TimeSeries().from_pandas(table_torus)
table = TimeSeries().from_pandas(table)

plot_visits_multi(dfs[0:4], 'PFlux', unit='GW/km²', ret='saveshow')
plot_visits_multi(dfs[0:4], 'Power', unit='GW' , ret='saveshow')
plot_visits_multi(dfs[0:4], 'Area', unit='km²', ret='saveshow')

fig,axes = plt.subplots(6, 1, figsize=(8, 5), dpi=100, gridspec_kw={"hspace": 0, "wspace": 0})
axes, dlims,lims = plot_visits_v2(dfs,  axs=axes,quantities=["PFlux","Power","Area"])
axes[0].xaxis.set_major_locator(mpl.dates.DayLocator(interval=2))
axes[0].xaxis.set_major_formatter(mpl.dates.DateFormatter("%d/%m/%y"))
axes[0].xaxis.set_minor_locator(mpl.dates.DayLocator(interval=1))
axes[0].xaxis.tick_top()
axes[3].plot(table_sw.time.datetime64, table_sw["jup_sw_pdyn"], color="#f88", linewidth=0.7, linestyle="--")
axes[3].plot(table.time.datetime64, table["jup_sw_pdyn"], color="red", linewidth=0.7)
axes[3].scatter(table.time.datetime64, table["jup_sw_pdyn"], color="red", marker=".", s=5)
      
axes[4].plot(
            table_torus.time.datetime64, table_torus["TPOW0710ADUSK"], color="#88f", linewidth=0.7, linestyle="--",
        )
axes[4].plot(table.time.datetime64, table["TPOW0710ADUSK"], color="blue", linewidth=0.7)
axes[4].scatter(table.time.datetime64, table["TPOW0710ADUSK"], color="blue", marker=".", s=5)
axes[4].plot(
            table_torus.time.datetime64, table_torus["TPOW0710ADAWN"], color="#8F8", linewidth=0.7, linestyle="--",
        )
axes[4].plot(table.time.datetime64, table["TPOW0710ADAWN"], color="green", linewidth=0.7)
axes[4].scatter(table.time.datetime64, table["TPOW0710ADAWN"], color="green", marker=".", s=5)
# make legend from filename and line2d object based on assigned props for each df 
handles = [mpl.lines.Line2D([0], [0], color=dfs[i]["color"][0], marker=dfs[i]["marker"][0], linestyle="--", label=f"{rpath(infiles[i])}") for i in range(len(dfs))]
fig.legend(handles=handles, title="Files", fontsize="small",bbox_to_anchor=(0.5, -0.2), loc='lower center', ncol=3,) 

for ax in axes:
    ax.set_xlim(dlims[0] - pd.Timedelta(hours=24), dlims[1] + pd.Timedelta(hours=24))
axes[5].set_ylabel("CML [°]")
plot_discontinuous_angle(axes[5],hisaki_sw_get_safe("CML"),"CML", threshold=45, color="black", linewidth=0.7)
# print(lims, t, table.columns)

conv = {"system":["initial","si","cgs"],
        "Flux":[1*1e8,1e4,1e6],
        "FluxU":["$\\times 10^{-8}~ GW~ km^{-2}$","$W~ m^{-2}$","$erg~ cm^{2}~ s^{-1}$"],
        "Power":[1,1e9,1e16*1e-18],
        "PowerU":["$GW$","$W$","$\\times 10^{18}~ erg~ s^{-1}$"],
        "Area":[1*1e-9,1e6*1e-15,1e10*1e-18],
        "AreaU":["$\\times 10^{9}~ km^{2}$","$\\times 10^{15}~ m^{2}$","$\\times 10^{18}~ cm^{2}$"],}
USYS = conv["system"].index("initial")

axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*conv["Flux"][USYS]:g}")) # 1[GW/km^2] -->  10^4[W/m^2] | 10^6 [erg cm^2 s^-1]
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*conv["Power"][USYS]:g}")) # 1[GW] --> 10^9 [W] | 10^16 [erg s^-1]
axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*conv["Area"][USYS]:g}")) # 1[km^2] --> 10^6 [m^2] | 10^10 [cm^2]

#TODO @samoo2000000 @will-roscoe @Yeefhan @RonanSzeto check units
axes[0].set_ylabel("Flux [{}]".format(conv["FluxU"][USYS]))
axes[1].set_ylabel("Power [{}]".format(conv["PowerU"][USYS]))
axes[2].set_ylabel("Area [{}]".format(conv["AreaU"][USYS]))

axes[3].set_ylabel(r"SW $[nPa]$")
axes[4].set_ylabel(r"Torus Power $[W/m^{2}]$")
a4= axes[2].twinx()
a4.plot(dfs[0]["EPOCH"], dfs[0]["lmax"], marker="x", lw=3, )
a4.plot(dfs[0]["EPOCH"], dfs[0]["lmin"], marker="x", lw=3, )
plt.show()

# testfits = fits.open(fpath('datasets/HST/group_13/jup_16-148-17-19-53_0100_v16_stis_f25srf2_proj.fits'))
#
# df = table.to_pandas()
# df.plot(x='EPOCH',y='TPOW0710ADAWN')
# plt.show()

# # get the first color value:
# 
#  axs[0].set_ylabel(f"{quantity}" + (f" [{unit}]" if unit else ""))
#     axs[0].set_ylim([0, df[quantity].max()])
#   
