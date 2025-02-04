from astropy.io import fits
from jarvis import make_gif, moind, fpath, gradmap
import glob
         
# makes an image
n = fpath(r'datasets\HST\v11\jup_16-146-16-00-57_2703_v11_stis_f25srf2_proj.fits.longexposure')
moind(n, 'temp', preproj_func=gradmap)
# makes a gif
#make_gif('datasets/HST/v06', dpi=300)

#hdu = fits.open(n)
#d = hdu[1].data
#print(d)


