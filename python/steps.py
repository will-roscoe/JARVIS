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
import scipy as sci
from jarvis.utils import approx_grid_dims, get_datapaths, merge_dfs, prepdfs, get_time_interval_from_multi, hisaki_sw_get_safe
from jarvis.plotting import figsize, apply_plot_defaults
from jarvis.const import CONST, HISAKI, HST
import matplotlib as mpl
from itertools import zip_longest
import pandas as pd
from jarvis.utils import get_obs_interval, group_to_visit






# dpr_histogram_plot(glob(fpath("temp/bindata/09_roi*")))
# plt.show()




figure_gen()
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
