from astropy.io import fits
from jarvis import make_gif, moind, fpath

# makes an image
n = fpath(r'datasets\HST\v20\jup_16-159-15-36-30_0100_v29_stis_f25srf2_proj.fits')
#moind(n, 'temp')
# makes a gif
make_gif('datasets\\HST\\v01', dpi=300)
hdu = fits.open(n)
d = hdu[1].data
print(d)
