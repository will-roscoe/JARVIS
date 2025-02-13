from jarvis.transforms import align_cmls, coadd, gradmap, dropthreshold
from jarvis.polar import process_fits_file, plot_polar, prepare_fits, fits_from_parent
from jarvis.utils import fpath
from astropy.io import fits
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cmasher as cmr
targetpaths = glob(fpath(r'datasets\HST\v06\*.fits'))
fitsfiles = [fits.open(x) for x in targetpaths]
aligned_data = [x[1].data for x in fitsfiles]
coadded_data = coadd(aligned_data)
graddat = np.where(gradmap(coadded_data)<200,np.nan,gradmap(coadded_data))
procorig = prepare_fits(fitsfiles[0], fixed='LT', full=False)
procgrad = process_fits_file(fits_from_parent(procorig, new_data=graddat))
proc = process_fits_file(fits_from_parent(procorig, new_data=coadded_data))

fitsd =[proc, procgrad, prepare_fits(fitsfiles[0], fixed='LT', full=False)]
    
fig= plt.figure()
ax = fig.subplots(1,1,subplot_kw={'projection': 'polar'})

plot_polar(fitsd[2], ax, cmap=cmr.cosmic)
plot_polar(fitsd[1], ax, cmap=cmr.ember, norm=mcolors.PowerNorm(gamma=0.9, vmin=0, vmax=2000))
plt.show()


