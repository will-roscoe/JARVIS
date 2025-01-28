from astropy.io import fits
from proj_images_og import moind
from const import fpath
i,n=0,0
n = fpath(r'datasets\HST\jup_16-138-00-08-30_0100_v01_stis_f25srf2_proj.fits')
hdulist = fits.open(n)
header = hdulist[1].header
image = hdulist[1].data
#print(header)
try:
    hemis = header['HEMISPH']
except NameError:
    hemis = str(input('Input hemisphere manually:  ("north" or "south")  '))
#filename = str(i)[-51:-5]
moind(image, header, filename=n, prefix='ocx8', dpi = 300, crop=1, rlim = 90, fixed = 'lon',
            hemis = hemis, full=True, moonfp=False, photo=n)
hdulist.close()
print(f'Image {n} of {str(i)[-51:-5]} created.')