from astropy.io import fits
from jarvis import make_gif, moind, fpath
import glob

            
# makes an image
n = fpath(r'datasets\HST\v04\jup_16-140-20-48-59_0103_v04_stis_f25srf2_proj.fits')
moind(n, 'temp')
# makes a gif
#make_gif('datasets/HST/v06', dpi=300)

#hdu = fits.open(n)
#d = hdu[1].data
#print(d)
