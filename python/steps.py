#!/usr/bin/env python3
import numpy as np
from astropy.io import fits
from jarvis import fpath
from jarvis.polar import moind
from matplotlib import pyplot as plt

# fpaths = fits_from_glob(hst_fpath_list()[6])

# copath = fpath('temp/gaussian-coadded.fits')
# fit = generate_coadded_fits(fpaths, saveto=copath, gaussian=(3,1), overwrite=True, indiv=False, coadded=True)
# fit.close()
# pt = pathfinder(copath, steps=True, fixlrange=(0.21,0.35))
fp = fits.open(fpath("datasets/HST/custom/rollavgs/g01sA/jup_16-137-23-43-30_0100_v01_stis_f25srf2_proj.fits"))
image_data = fp[1].data
fp.info()
print(f"{np.mean(image_data,axis=0)=}, {np.min(image_data)=}, {np.max(image_data)=}")
f = moind(fp)
plt.show()
