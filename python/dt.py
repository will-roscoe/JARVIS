#!/usr/bin/env python3

from datetime import datetime

import numpy as np
from astropy.table import Table, vstack
from astropy.timeseries import TimeSeries
from jarvis.const import Dirs

# sw_fp = fpath("datasets/Solar_Wind.txt")
# df_sw = Table.read(sw_fp, format='ascii')
# df_sw['Time_UT'] = [x[:10]+"T"+x[11:] for x in df_sw['Time_UT']]
# df_sw.rename_column('Time_UT','time')
# df_sw = TimeSeries(data=df_sw)
# fits_torus, fits_aurora = fits_from_glob(str(TORUS_DIR)), fits_from_glob(str(AURORA_DIR))
# df_torus = get_data_over_interval(fits_torus, (datetime(2000,1,1), datetime(2099,1,1))) # this range is arbitrary, i just want the combined datasets for now
# df_torus["time"] = [yrdoysod_to_datetime(int(yr), int(doy), int(sod)) for yr, doy, sod in zip(df_torus["YEAR"], df_torus["DAYOFYEAR"], df_torus["SECOFDAY"])]
# df_torus = TimeSeries(df_torus)
# df_aurora = get_data_over_interval(fits_aurora, (datetime(2000,1,1), datetime(2099,1,1)))
# df_aurora["time"] = [yrdoysod_to_datetime(int(yr), int(doy), int(sod)) for yr, doy, sod in zip(df_aurora["YEAR"], df_aurora["DAYOFYEAR"], df_aurora["SECOFDAY"])]
# df_aurora = TimeSeries(df_aurora)
# for fit in [*fits_torus, *fits_aurora]:
#     fit.close()
# # ensure each dataset has the same columns by adding the missing columns with NaN values
# cols = set(df_sw.columns) | set(df_torus.columns) | set(df_aurora.columns)
# for col in cols:
#     if col not in df_sw.columns:
#         df_sw[col] = [np.nan]*len(df_sw)
#     if col not in df_torus.columns:
#         df_torus[col] = [np.nan]*len(df_torus)
#     if col not in df_aurora.columns:
#         df_aurora[col] = [np.nan]*len(df_aurora)
# # combine the datasets
# ts_main = vstack([df_sw, df_torus, df_aurora])
# # now we want to identify any non unique timestamps, and where possible, combine those rows if eg row A has values in 1,2,3,4 and B has values in 5,6,7,8 (excluding the timestamp). if there are values present in both A and B, combine if they are equal.
# ts_main = ts_main.group_by('time')
# # we can now combine the rows with the same timestamp
# ts_main = ts_main.groups.aggregate(np.nanmean)
# ts_main.write(fpath("datasets/Hisaki_SW-combined.csv"), format='fits', overwrite=True)
### del ts_main
#ts_main = TimeSeries.read(fpath("datasets/Hisaki_SW-combined.csv"))
# remove columns with any NaN values
# fig = plt.figure(figsize=(12,8))
# axs = fig.subplots(3,1, sharex='col', gridspec_kw={'hspace':0})
# plotlist = ((0, 'jup_sw_pdyn'), (1, 'TPOW1190A'), (2, 'TPOW0710ADAWN'), (2, 'TPOW0710ADUSK'))
# for i, col in plotlist:
#     axs[i].plot(ts_main.time.datetime64, ts_main[col])
# plt.show()
# from jarvis.transforms import azimuthal_equidistant
# from jarvis.utils import fpath, hst_fpath_list, fits_from_glob
# from astropy.io import fits
# from matplotlib import pyplot as plt
# import cmasher as cmr
# from glob import glob
# bydur = [glob(fp+"/*.fits") for fp in hst_fpath_list()]
# figs = [[plt.figure(figsize=(5,5)) for fit in fp] for i,fp in enumerate(bydur) if i<2]
# for i,d in enumerate(bydur):
#     if i<2:
#         for j,fit in enumerate(d):
#             azim=azimuthal_equidistant(fit)
#             ax = figs[i][j].subplots(2,1)
#             ax[0].imshow(azim, cmap=cmr.neutral)
#             ax[1].imshow(fits.getdata(fit,1), cmap=cmr.neutral)
#             figs[i][j].set(facecolor="black")
#             figs[i][j].savefig(f"temp{i},{j}.png")
#             #fit.close()
from jarvis.extensions import pathfinder
from jarvis.utils import fits_from_glob, fpath, get_data_over_interval, yrdoysod_to_datetime
from matplotlib import pyplot as plt

pathfinder('temp/coadds/v01sB.fits')
