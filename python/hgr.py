#!/usr/bin/env python3

from datetime import datetime
from random import choice

import matplotlib as mpl
import numpy as np
import pandas as pd
from astropy.timeseries import TimeSeries
from jarvis import fpath, get_obs_interval, hst_fpath_list
from jarvis.utils import approx_grid_dims, get_power_datapaths, group_to_visit, rpath
from matplotlib import pyplot as plt
from tqdm import tqdm

from jarvis.plotting import plot_discontinuous_angle


    
def plot_visit_intervals(axs,visits,fitsdirs=hst_fpath_list()[:-1],color="#aaa", annotate_ax=False, **kwargs ):
     for f in fitsdirs:
        mind, maxd = get_obs_interval(f)
        mind = pd.to_datetime(mind)
        maxd = pd.to_datetime(maxd)
        # span over the interval
        fvisit = group_to_visit(int(f.split("_")[-1][:2]))
        for ax in axs if isinstance(axs,list) else [axs]:
            ax.axvspan(mind, maxd, alpha=0.5, color="#aaa5" if fvisit not in visits else color, **kwargs)
            if annotate_ax:
                ax.annotate(f"v{fvisit}",xy=(mind, 0),xytext=(mind, 0),fontsize=10,color="black",va="top",ha="center",rotation=90)

def visit_ticks(ax, visits, xs):
        ax.set_xticks(xs, labels=[f"v{a}" for a in visits])
    
def fill_between_qty(ax,xdf,df,quantity,visits, edgecorrect=False, fc=(0.3,0.3,0.3,0.3), ec=(0.3,0.3,0.3,0.3), hatch="\\\\",zord=5):             
    # define minimums and maximums per visit, get minquant, maxquant, tmin, tmax.
    ranges = []
    for j,u in enumerate(visits):
        if u in xdf["visit"].unique():
            idfint = df.where(df["visit"] == u)
            dmax, dmin = [idfint["EPOCH"].max(), idfint["EPOCH"].min()]
            qmin, qmax = [idfint[quantity].min(), idfint[quantity].max()]
            ranges.append([dmin, qmin, qmax])
            ranges.append([dmax, qmin, qmax])
    ranges=pd.DataFrame(ranges, columns=["time","min","max"])
    ranges = ranges.dropna()
    ranges = ranges.sort_values(by=["time"])
    ax.fill_between(ranges['time'],  ranges["max"], ranges["min"],facecolor=fc, edgecolor=ec,hatch=hatch,linewidth=0.5,  interpolate=True, zorder=zord)
    if edgecorrect:
        ax.fill_between(ranges['time'],  ranges["max"], ranges["min"],facecolor=(0,0,0,0), edgecolor=(1,1,1), interpolate=True, zorder=zord+1)

def mgs_grid(fig, visits):
    icols, jrows = approx_grid_dims(visits)
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 5], hspace=0.5, wspace=0)  # hspace=0,
    mgs = gs[1].subgridspec(jrows, icols, height_ratios=[1 for _ in range(jrows)], wspace=0)  # hspace=0,wspace=0,
    main_ax = fig.add_subplot(gs[0])
    main_ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval=2))
    main_ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d/%m/%y"))
    main_ax.xaxis.set_minor_locator(mpl.dates.DayLocator(interval=1))
    axs = [[fig.add_subplot(mgs[j, i], label=f"v{visits[i+icols*j]}") for i in range(icols)] for j in range(jrows)]
    return main_ax, axs





def plot_visits(
    df, quantity="PFlux",  ret="showsavefig", unit=None,
):  # corrected = False, True, None (remove negative values)
    ps = prep_sort_dfs([df])
    corr_df, visits = ps[0][0], ps[1]
    fitsdirs = hst_fpath_list()[:-1]
    fig = plt.figure(figsize=(19.2, 10.8), dpi=70)
    # this generates the grid dimensions based on the number of visits and a maximum of 8 columns
    icols, jrows = approx_grid_dims(visits)
    main_ax, axs = mgs_grid(fig, visits)
    plot_visit_intervals(main_ax,visits)
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
            axs[j][i].xaxis.set_major_locator(mpl.dates.MinuteLocator(interval=int(np.floor((delta.total_seconds() / 60) / 3))))
            axs[j][i].xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
            axs[j][i].annotate(f"v{visits[i+icols*j]}",)
            axs[j][i].annotate(f'{dmin.strftime("%d/%m/%y")}',)
            axs[j][i].tick_params(axis="both", pad=0)
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
    corr_, visits = prep_sort_dfs(dfs)
    df = merge_dfs(dfs)
    fitsdirs = hst_fpath_list()[:-1]
    icols, jrows = approx_grid_dims(visits)
    fig = plt.figure(figsize=(19.2, 10.8), dpi=70) if figure is None else figure
    uniquevisits = unique_visits(visits)[0]
    main_ax, axs = mgs_grid(fig, uniquevisits)
    main_ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval=2))
    main_ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d/%m/%y"))
    main_ax.xaxis.set_minor_locator(mpl.dates.DayLocator(interval=1))
    
    plot_visit_intervals(main_ax,uniquevisits, annotate_ax=True)
    for cdf in corr_:
        main_ax.scatter(cdf["EPOCH"], cdf[quantity], marker="x", s=5, c=cdf["color"], zorder=5)
        #main_ax.plot(cdf["EPOCH"], cdf[quantity], color="#aaf", linewidth=0.5, linestyle="--")
    visits = []
    ranges = {}
    main_ax.xaxis.tick_top()
    for i in range(icols):
        for j in range(jrows):
            if i + icols * j < len(visits):
                for cdf in corr_:
                    dfa = cdf.loc[cdf["visit"] == uniquevisits[i + icols * j]]
                    if not dfa.empty:
                        axs[j][i].plot(dfa["EPOCH"], dfa[quantity], color="#aaf", linewidth=0.5)
                        axs[j][i].scatter(dfa["EPOCH"], dfa[quantity], marker=".", s=5, color=dfa["color"].tolist()[0], zorder=5,)
                axs[j][i].yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
                idfint = df.where(df["visit"] == uniquevisits[i + icols * j])
                dmax, dmin = [idfint["EPOCH"].max(), idfint["EPOCH"].min()]
                delta = dmax - dmin
                axs[j][i].set_xlim([dmin - 0.01 * delta, dmax + 0.01 * delta])
                axs[j][i].xaxis.set_major_locator(mpl.dates.MinuteLocator(interval=int(np.floor((delta.total_seconds() / 60) / 3))),)
                axs[j][i].xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
                axs[j][i].annotate(f"v{uniquevisits[i+icols*j]}",xy=(0, 1),xycoords="axes fraction",xytext=(+0.5, -0.5),textcoords="offset fontsize",fontsize="medium",verticalalignment="top",weight="bold",bbox={"facecolor": "#0000", "edgecolor": "none", "pad": 3.0},)
                axs[j][i].annotate(f'{dmin.strftime("%d/%m/%y")}',xy=(1, 1),xycoords="axes fraction",xytext=(-0.05, +0.1),textcoords="offset fontsize",fontsize="medium",verticalalignment="bottom",horizontalalignment="right",annotation_clip=False,bbox={"facecolor": "#0000", "edgecolor": "none", "pad": 3.0},)
                axs[j][i].tick_params(axis="both", pad=0)
                qmin, qmax = [idfint[quantity].min(), idfint[quantity].max()]
                axs[j][i].set_ylim([max(0, qmin - 0.1 * (qmax - qmin)), min(qmax + 0.1 * (qmax - qmin), df[quantity].max())],)
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
            if i + icols * j < len(visits):
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
    dt_dfs, visits = prep_sort_dfs(dfs)
    df = merge_dfs(dfs)
    uniquevisits, alt_xs = unique_visits(visits)
    plot_visit_intervals(axs,uniquevisits)
    visit_ticks(axs[-1], alt_xs)
    #top plot
    hch = ["\\\\","//","--","||","oo","xx","**"] + ["+" for _ in range(len(dt_dfs)-7)]
        # main scatter&plot loop
    for x,xdf in enumerate(dt_dfs):
        for i in range(len(quantities)):
            if quantities[i] is not None:
                    ax[i].scatter(xdf["EPOCH"], xdf[quantities[i]], marker=xdf["marker"][0], s=5, c=xdf["color"], zorder=xdf["zorder"][0])
                    ax[i].plot(xdf["EPOCH"], xdf[quantities[i]], color="#aaf", linewidth=0.5, linestyle="--", zorder=xdf["zorder"][0])
                    fill_between_qty(ax[i],xdf,df,quantities[i],uniquevisits, fc=(xdf["color"][0],0.05),ec= (xdf["color"][0],0.2), hatch=hch[x], zord=xdf["zorder"][0])
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







table_sw = hisaki_sw_get_safe("jup_sw_pdyn")
table_torus = hisaki_sw_get_safe("TPOW0710ADUSK")
table = hisaki_sw_get_safe()

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
fig.savefig(fpath("figures/imgs/combined.png"))
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
