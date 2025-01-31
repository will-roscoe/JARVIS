import numpy as np

# Set up matplotlib
import matplotlib.pyplot as plt

from astropy.io import fits
from jarvis import fpath
jtest = fpath('datasets\HST\v01\jup_16-137-23-43-30_0100_v01_stis_f25srf2_proj.fits')

#hdul = fits.open(jtest)
#hdul.info()
hdu_list = fits.open(jtest)
hdu_list.info()
 
#with fits.open('datasets\HST\jup_16-137-23-43-30_0100_v01_stis_f25srf2_proj.fits') as hdu:
# 
#    science_data1 = hdu[1].data
#    science_header1 = hdu[1].header

#print(science_header1)

image_data = hdu_list[1].data
print(type(image_data))
print(image_data.shape)

print(image_data)

plt.imshow(image_data, cmap='hot')
plt.show()