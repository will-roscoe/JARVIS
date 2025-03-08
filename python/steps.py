#!/usr/bin/env python3
import cmasher as cmr
from jarvis import fits_from_glob, fpath, hst_fpath_list
from jarvis.polar import moind

fpaths = fits_from_glob(hst_fpath_list()[6])

# copath = fpath('temp/gaussian-coadded.fits')
# fit = generate_coadded_fits(fpaths, saveto=copath, gaussian=(3,1), overwrite=True, indiv=False, coadded=True)
# fit.close()
# pt = pathfinder(copath, steps=True, fixlrange=(0.21,0.35))
fp = fpaths[0]
fig = moind(fp, cmap=cmr.neutral)[0]
fig.savefig(fpath("temp/moind.png"))
