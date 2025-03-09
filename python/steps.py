#!/usr/bin/env python3
import cmasher as cmr
from jarvis import fits_from_glob, fpath, hst_fpath_list
from jarvis.polar import moind
from astropy.io import fits
from matplotlib import pyplot as plt
import numpy as np
#fpaths = fits_from_glob(hst_fpath_list()[6])

# copath = fpath('temp/gaussian-coadded.fits')
# fit = generate_coadded_fits(fpaths, saveto=copath, gaussian=(3,1), overwrite=True, indiv=False, coadded=True)
# fit.close()
# pt = pathfinder(copath, steps=True, fixlrange=(0.21,0.35))
fp = fits.open(fpath("datasets/HST/group_20/jup_16-159-16-11-30_0100_v29_stis_f25srf2_proj.fits"))
image_data = np.asarray(fp[2].data) 
print(f"{np.mean(image_data)=}, {np.min(image_data)=}, {np.max(image_data)=}")
#f = moind(fp)
plt.show()
