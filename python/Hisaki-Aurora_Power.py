# import relevant packages
import numpy as np
from astropy.io import fits					#handling of FITS files
import matplotlib.pyplot as plt		#for making plots
from pathlib import Path 
import datetime
import os
GHROOT = Path(__file__).parents[1]
def fpath(x):
    return os.path.join(GHROOT, x)

# function to plot x-axis in H:M:S format
def HMS(sec_of_day, pos):
    sec_of_day = int(sec_of_day)
    hours = sec_of_day // 3600
    rsec = sec_of_day % 3600
    minutes = rsec // 60
    seconds = rsec % 60
    return f"{hours:0>2}:{minutes:0>2}:{seconds:0>2}"

# function to plot x-axis in hours only
def hours_conversion(sec_of_day, pos):
    hours = sec_of_day / 3600
    return hours
#############################################
#### Retrieving data from the FITS file #####
#############################################

# file of interest (assumes file is in same directory as this script - if it isn't, you need to add the path to the file)
filename = 'C:datasets\Hisaki\Aurora Power\exeuv_aurora_20160504_lv03_LT00-24_dt00010_vr01_00.fits'

# load the file
hdul = fits.open(fpath(filename))

# print initial and final times of observation
init_time = hdul[1].header["DATE-OBS"]
fin_time = hdul[1].header["DATE-END"]
print("From "+ init_time + " to " + fin_time)

# 'hdul' is a HDUList (header-data unit list) containing 3 elements; using Python indexing you can retrieve the individual elements:
# hdul[0] contains 'metadata' - you'll rarely have to use this one
# hdul[1] contains image data, stored as a 2D array where each element is a value corresponding to a specific pixel in the image
# hdul[2] is a table containing the time series data, and we can retrieve the data using hdul[2].data

# extract time series data for use on the x-axis
time_series_data = hdul[2].data

#to get data from a specific column in the table, use the 'field' function; we want 'SECOFDAY', 'TPOW0710ADAWN' and 'TPOW0710ADUSK'
time = time_series_data.field('SECOFDAY')
aurora_power = time_series_data.field('TPOW1190A')

############################
#### Plotting the data #####
############################

#set up a figure
f, ax = plt.subplots(1, 1)

#label the axes
ax.set_xlabel('Time (h)')
ax.set_ylabel('Power (GW)')

#plot the data
ax.plot(time, aurora_power, label='Power')

#add a legend to the plot (it will automatically contain the labels defined above)
titlesp = str(time_series_data[0].field('YEAR')) + '-' + str(time_series_data[0].field('DAYOFYEAR'))
ax.legend()
plt.title(titlesp)

#removes unnecessary whitespace from plot (comment this out to see the difference)
plt.tight_layout()

figname = 'Tot._auroral_power_vs_time.pdf'

# spaces x-axis ticks to be every 2 hours
plt.xticks(np.arange(0, 86401, step = 7200))

# codes for formatting the time axis (x-axis) in H:M:S format or integer hours only, only use one at a time
#ax.xaxis.set_major_formatter(plt.FuncFormatter(HMS))
ax.xaxis.set_major_formatter(plt.FuncFormatter(hours_conversion))

plt.show()

#save the figure (this will save it in the current directory; to save it somewhere else edit the filename to include the desired directory)
#plt.savefig(figname, bbox_inches='tight')