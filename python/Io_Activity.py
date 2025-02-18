# import relevant packages
from astropy.io import fits					#handling of FITS files
from matplotlib import pyplot as plt		#for making plots


#############################################
#### Retrieving data from the FITS file #####
#############################################

# file of interest (assumes file is in same directory as this script - if it isn't, you need to add the path to the file)
filename = 'TestData/exeuv_torus_20150131_lv03_LT00-24_dt00010_vr01_00.fits'

# load the file
hdul = fits.open(filename)

# 'hdul' is a HDUList (header-data unit list) containing 3 elements; using Python indexing you can retrieve the individual elements:
# hdul[0] contains 'metadata' - you'll rarely have to use this one
# hdul[1] contains image data, stored as a 2D array where each element is a value corresponding to a specific pixel in the image
# hdul[2] is a table containing the time series data, and we can retrieve the data using hdul[2].data
time_series_data = hdul[2].data

#to get data from a specific column in the table, use the 'field' function; we want 'SECOFDAY', 'TPOW0710ADAWN' and 'TPOW0710ADUSK'
time = time_series_data.field('SECOFDAY')
intensity_dawn = time_series_data.field('TPOW0710ADAWN')
intensity_dusk = time_series_data.field('TPOW0710ADUSK')



############################
#### Plotting the data #####
############################

#set up a figure
f, ax = plt.subplots(1, 1)

#label the axes
ax.set_xlabel('Time (s)')
ax.set_ylabel('Intensity (GW)')

#plot the data
ax.plot(time, intensity_dawn, label='Dawn')
ax.plot(time, intensity_dusk, label='Dusk')

#add a legend to the plot (it will automatically contain the labels defined above)
titlesp = str(time_series_data[0].field('YEAR')) + '-' + str(time_series_data[0].field('DAYOFYEAR'))
ax.legend()
plt.title(titlesp)

#removes unnecessary whitespace from plot (comment this out to see the difference)
plt.tight_layout()

#save the figure (this will save it in the current directory; to save it somewhere else edit the filename to include the desired directory)
figname = 'EUV_intensity_vs_time.pdf'
plt.savefig(figname, bbox_inches='tight')

