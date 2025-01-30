from astropy.io import fits
from jarvis import *

n = fpath(r'datasets\HST\v02\jup_16-138-18-48-16_0100_v02_stis_f25srf2_proj.fits')
hdulist = fits.open(n)
header = hdulist[1].header
image = hdulist[1].data
hemis = header['HEMISPH']
moind(image, header, 'datasets/HST/v04/jup_16-140-20-05-39_0100_v04_stis_f25srf2_proj.fits','',  dpi = 300, crop=1, rlim = 90, fixed = 'lon',
             hemis = hemis, full=True, photo=n)
hdulist.close()
print(f'Image {n} created.')

#multigif(['v01'],'2016','', '', '100', '40', True, 'lon', )

