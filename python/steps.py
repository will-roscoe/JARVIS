from jarvis import fpath, hst_fpath_list, fits_from_glob
from jarvis.cvis import generate_coadded_fits
from jarvis.extensions import pathfinder
from jarvis.polar import moind
import cmasher as cmr
fpaths = fits_from_glob(hst_fpath_list()[6])

# copath = fpath('temp/gaussian-coadded.fits')
# fit = generate_coadded_fits(fpaths, saveto=copath, gaussian=(3,1), overwrite=True, indiv=False, coadded=True)
# fit.close()
# pt = pathfinder(copath, steps=True, fixlrange=(0.21,0.35))
fp = fpaths[0]
fig = moind(fp, cmap=cmr.neutral)[0]
fig.savefig(fpath('temp/moind.png'))