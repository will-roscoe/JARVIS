from astropy.io import fits
from jarvis import make_gif

# n = fpath(r'datasets\HST\v02\jup_16-138-18-48-16_0100_v02_stis_f25srf2_proj.fits')
# hdulist = fits.open(n)
# header = hdulist[1].header
# image = hdulist[1].data
# hemis = header['HEMISPH']
# moind(image, header, filename=n, prefix='ocx8', dpi = 300, crop=1, rlim = 90, fixed = 'lon',
#             hemis = hemis, full=True, photo=n)
# hdulist.close()
# print(f'Image {n} created.')

make_gif('datasets\\HST\\v01-may16\\')

chr(0x1F600)
