#!/usr/bin/env python3
import datetime
import numpy as np
from astropy.io import fits
from pytest import mark
from jarvis import fpath, powercalc
from jarvis.plotting import moind, set_yaxis_exfmt
from matplotlib import pyplot as plt
from jarvis import fits_from_glob, hst_fpath_list
from jarvis.extensions import extract_conf, pathfinder
from jarvis.cvis import generate_coadded_fits
from glob import glob

from jarvis.utils import get_datapaths, prepdfs, get_time_interval_from_multi, hisaki_sw_get_safe
from jarvis.plotting import figsize, apply_plot_defaults
from jarvis.const import HISAKI, HST

# fpaths = fits_from_glob(hst_fpath_list()[6])

# copath = fpath('temp/coadds/v09sB.fits')
# avgpaths = glob(fpath('temp/rollavgs/v09sB/*.fits'))
# fit = generate_coadded_fits(fpaths, saveto=copath, kernel_params=(3,1), overwrite=True, indiv=False, coadded=True)
# fit.close()
# pt = pathfinder(copath, )
# conf=extract_conf(fits.getheader(copath, "BOUNDARY"))
# pt2 = pathfinder(avgpaths[0], **conf, conf=conf)
# print(pt)
# print(pt2)
# pc = powercalc(fits.open(avgpaths[0]),writeto=fpath("savetest.txt"))
# fp = fits.open(fpath("datasets/HST/custom/rollavgs/g01sA/jup_16-137-23-43-30_0100_v01_stis_f25srf2_proj.fits"))
# image_data = fp[1].data
# fp.info()
# print(f"{np.mean(image_data,axis=0)=}, {np.min(image_data)=}, {np.max(image_data)=}")
# f = moind(fp)
# plt.show()
def figure_gen(include="last5",pervisit=False,overall=True,overlaid=False,megafigure=True,plot_hst=["Total_Power","Avg_Flux","Area"],plot_hisaki_sw=["Torus_Power_Dawn","Torus_Power_Dusk","SW","Aurora_Power","Aurora_Flux"]):
        hisaki_colnames = list(HISAKI.colnames.values())
        hst_colnames = list(HST.colnames.values())
        hst_cols = [h for h in plot_hst if h in hst_colnames]
        hisaki_cols = [h for h in plot_hisaki_sw if h in hisaki_colnames]
        if "last" in include:
            num = int("".join([char if char.isnumeric() else "" for char in include]))
            hst_datasets = get_datapaths()[0:num+1]
        else:
            hst_datasets = get_datapaths()
        hst_datasets = prepdfs(hst_datasets)
        hisaki_dataset = hisaki_sw_get_safe(method="all")
        time = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        if overall:
            # plot a stacked x-axis plot of all columns in the datasets, over the timeperiod of hst_datasets
            xlim = get_time_interval_from_multi(hst_datasets)
            extend = np.timedelta64(1, "D")
            xlim = (xlim[0]-extend, xlim[1]+extend)
            figure,axs = plt.subplots(len(hst_cols)+len(hisaki_cols),1,figsize=figsize(max=True),sharex=True, gridspec_kw={"hspace":0.05}, subplot_kw={"xmargin":0.02, "ymargin":0.02})
            
            for i, col in enumerate(hst_cols):
                for hst in hst_datasets:
                    axs[i].scatter(hst["time"],hst[col],label=col, color=hst["color"][0], marker=hst["marker"][0], zorder=hst["zorder"][0], s=2)
                    mv = HST.mapval(col, ["label","unit"])
                    set_yaxis_exfmt(axs[i], label=mv[0], unit=mv[1])
            for i, col in enumerate(hisaki_cols):
                axs[i+len(hst_cols)].plot(hisaki_dataset.time.datetime64,hisaki_dataset[col],label=col)
                mv = HISAKI.mapval(col, ["label","unit"])
                set_yaxis_exfmt(axs[i+len(hst_cols)], label=mv[0], unit=mv[1])
            apply_plot_defaults(fig=figure, time_interval=xlim, day_interval=1)

            
                 
            plt.savefig(fpath(f"figures/imgs/overall_{time}.png"))
            plt.close()

        if overlaid: # plot all datasets on the same plot
            figure,ax = plt.subplots(figsize=figsize(max=True))
            for hst in hst_datasets:
                for col in hst_cols:
                    ax.scatter(hst["time"],hst[col],label=col, color=hst["color"][0], marker=hst["marker"][0], zorder=hst["zorder"][0], s=2)
            for col in hisaki_cols:
                ax.plot(hisaki_dataset.time.datetime64,hisaki_dataset[col],label=col)
            apply_plot_defaults(fig=figure)
            plt.savefig(fpath(f"figures/imgs/overlaid_{time}.png"))
            plt.close()
        if megafigure:
            # plot all datasets over full interval like overlaid, but then plot each visit separately below it in a grid.
            # example if we find 8 unique visits in the hst dataset, we will have a grid of 8 plots below the overlaid plot.
figure_gen()