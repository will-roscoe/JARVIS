from astropy.io import fits
from jarvis import make_gif, moind, fpath

# makes an image
n = fpath(r'datasets\HST\v02\jup_16-138-18-48-16_0100_v02_stis_f25srf2_proj.fits')
moind(n, 'temp')
# makes a gif
#make_gif('datasets\\HST\\v01', dpi=300)
hdu = fits.open(n)
d = hdu[1].data
print(d)
