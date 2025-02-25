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

def HMS(seconds, pos):
    seconds = int(seconds)
    hours = seconds / 3600
    seconds -= 3600 * hours
    minutes = seconds / 60
    seconds -= 60 * minutes
    if hours == 0:
        if minutes == 0:
            return "%ds" % (seconds)
        return "%dm%02ds" % (minutes, seconds)
    return "%dh%02dm" % (hours, minutes)

#############################################
#### Retrieving data from the FITS file #####
#############################################

# file of interest (assumes file is in same directory as this script - if it isn't, you need to add the path to the file)
filename = 'C:datasets\Hisaki\Torus Power\exeuv_torus_20160504_lv03_LT00-24_dt00010_vr01_00.fits'

# load the file
hdul = fits.open(fpath(filename))
init_time = hdul[1].header["BLK_STA"]
fin_time = hdul[1].header["BLK_END"]
print("From "+ init_time + " to " + fin_time)

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

figname = 'EUV_intensity_vs_time.pdf'


# fig & axes code here
#ax.xaxis.set_major_formatter(plt.FuncFormatter(HMS))

plt.show()

#save the figure (this will save it in the current directory; to save it somewhere else edit the filename to include the desired directory)
#plt.savefig(figname, bbox_inches='tight')