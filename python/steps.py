#!/usr/bin/env python3
import numpy as np
from astropy.io import fits
from jarvis import fpath, powercalc
from jarvis.polar import moind
from matplotlib import pyplot as plt
from jarvis import fits_from_glob, hst_fpath_list
from jarvis.extensions import extract_conf, pathfinder
from jarvis.cvis import generate_coadded_fits
from glob import glob
fpaths = fits_from_glob(hst_fpath_list()[6])

copath = fpath('temp/coadds/v09sB.fits')
avgpaths = glob(fpath('temp/rollavgs/v09sB/*.fits'))
# fit = generate_coadded_fits(fpaths, saveto=copath, kernel_params=(3,1), overwrite=True, indiv=False, coadded=True)
# fit.close()
pt = pathfinder(copath, )
conf=extract_conf(fits.getheader(copath, "BOUNDARY"))
pt2 = pathfinder(avgpaths[0], **conf, conf=conf)
print(pt)
print(pt2)
pc = powercalc(fits.open(avgpaths[0]),writeto=fpath("savetest.txt"))
# fp = fits.open(fpath("datasets/HST/custom/rollavgs/g01sA/jup_16-137-23-43-30_0100_v01_stis_f25srf2_proj.fits"))
# image_data = fp[1].data
# fp.info()
# print(f"{np.mean(image_data,axis=0)=}, {np.min(image_data)=}, {np.max(image_data)=}")
# f = moind(fp)
# plt.show()
