#!/usr/bin/env python3



###### Aurora Power Plotting ######
# import relevant packages
import os
from pathlib import Path

import matplotlib.pyplot as plt  # for making plots
import numpy as np
from astropy.io import fits  # handling of FITS files
from .utils import fpath
import glob
import pandas as pd
import datetime
# function to plot x-axis in H:M:S format
def hms_convert(sec_of_day, pos):
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
def aurora_plot(aurora_filename):
    # load the file
    hdul = fits.open(fpath(aurora_filename))
    # print initial and final times of observation
    init_time = hdul[1].header["DATE-OBS"]
    fin_time = hdul[1].header["DATE-END"]
    print("From " + init_time + " to " + fin_time)
    # 'hdul' is a HDUList (header-data unit list) containing 3 elements; using Python indexing you can retrieve the individual elements:
    # hdul[0] contains 'metadata' - you'll rarely have to use this one
    # hdul[1] contains image data, stored as a 2D array where each element is a value corresponding to a specific pixel in the image
    # hdul[2] is a table containing the time series data, and we can retrieve the data using hdul[2].data
    # extract time series data for use on the both axis
    time_series_data = hdul[2].data
    # to get data from a specific column in the table, use the 'field' function; we want 'SECOFDAY', 'TPOW0710ADAWN' and 'TPOW0710ADUSK'
    time = time_series_data.field("SECOFDAY")
    aurora_power = time_series_data.field("TPOW1190A")
    # print(aurora_power)
    print(time)
    ############################
    #### Plotting the data #####
    ############################
    # set up a figure
    f, ax = plt.subplots(1, 1)
    # label the axes
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Power (GW)")
    # plot the data
    ax.plot(time, aurora_power, label="Power")
    # add a legend to the plot (it will automatically contain the labels defined above)
    titlesp = str(time_series_data[0].field("YEAR")) + "-" + str(time_series_data[0].field("DAYOFYEAR"))
    ax.legend()
    plt.title(titlesp)
    # removes unnecessary whitespace from plot (comment this out to see the difference)
    plt.tight_layout()
    # figname = 'Tot._auroral_power_vs_time.pdf'
    # spaces x-axis ticks to be every 2 hours
    plt.xticks(np.arange(0, 86401, step=14400))
    # codes for formatting the time axis (x-axis) in H:M:S format or integer hours only, only use one at a time
    ax.xaxis.set_major_formatter(plt.FuncFormatter(hms_convert))
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(hours_conversion))
    plt.show()
    # save the figure (this will save it in the current directory; to save it somewhere else edit the filename to include the desired directory)
    # plt.savefig(figname, bbox_inches='tight')

###### Torus Power Plotting ######
def torus_plot(torus_filename):
    # load the file
    hdul = fits.open(fpath(torus_filename))

    # print initial and final times of observation
    init_time = hdul[1].header["DATE-OBS"]
    fin_time = hdul[1].header["DATE-END"]
    print("From " + init_time + " to " + fin_time)

    # 'hdul' is a HDUList (header-data unit list) containing 3 elements; using Python indexing you can retrieve the individual elements:
    # hdul[0] contains 'metadata' - you'll rarely have to use this one
    # hdul[1] contains image data, stored as a 2D array where each element is a value corresponding to a specific pixel in the image
    # hdul[2] is a table containing the time series data, and we can retrieve the data using hdul[2].data

    # extract time series data for use on both axis
    time_series_data = hdul[2].data

    # to get data from a specific column in the table, use the 'field' function; we want 'SECOFDAY', 'TPOW0710ADAWN' and 'TPOW0710ADUSK'
    time = time_series_data.field("SECOFDAY")
    power_dawn = time_series_data.field("TPOW0710ADAWN")
    power_dusk = time_series_data.field("TPOW0710ADUSK")
    print(power_dawn)
    print(time)

    ############################
    #### Plotting the data #####
    ############################

    # set up a figure
    f, ax = plt.subplots(1, 1)

    # label the axes
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Intensity (GW)")

    # plot the data
    ax.plot(time, power_dawn, label="Dawn")
    ax.plot(time, power_dusk, label="Dusk")

    # add a legend to the plot (it will automatically contain the labels defined above)
    titlesp = str(time_series_data[0].field("YEAR")) + "-" + str(time_series_data[0].field("DAYOFYEAR"))
    ax.legend()
    plt.title(titlesp)

    # removes unnecessary whitespace from plot (comment this out to see the difference)
    plt.tight_layout()

    # figname = 'EUV_intensity_vs_time.pdf'

    # spaces x-axis ticks to be every 2 hours
    plt.xticks(np.arange(0, 86401, step=14400))

    # codes for formatting the time axis (x-axis) in H:M:S format or integer hours only, only use one at a time
    ax.xaxis.set_major_formatter(plt.FuncFormatter(hms_convert))
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(hours_conversion))

    plt.show()

    # save the figure (this will save it in the current directory; to save it somewhere else edit the filename to include the desired directory)
    # plt.savefig(figname, bbox_inches='tight')
def hisaki_sw():
    def date_aligner(sec_of_day, pos):
        day = sec_of_day / 86400
        current_date = datetime.date(2016, 5, 6)
        current_date = current_date + datetime.timedelta(days = day)
        return current_date

    aurora_counter = 0
    torus_counter = 0

    aurora_fitspath = glob.glob(r"C:datasets/Hisaki/Aurora Power/*.fits")
    torus_fitspath = glob.glob(r"C:datasets/Hisaki/Torus Power/*.fits")
    aurora_groups_list = aurora_fitspath[2:43]
    torus_groups_list = torus_fitspath[2:43]

    aurora_power_all = np.array([])
    aurora_time_all = np.array([])
    power_dawn_all = np.array([])
    power_dusk_all = np.array([])
    torus_time_all = np.array([])

    testlist = np.array([r"C:datasets\Hisaki\Aurora Power\exeuv_aurora_20160506_lv03_LT00-24_dt00010_vr01_00.fits"])

    for i in aurora_groups_list:
        hdul = fits.open(fpath(i))
        time_series_data = hdul[2].data
        aurora_time = time_series_data.field('SECOFDAY') +86400 * aurora_counter
        if i == r'C:datasets/Hisaki/Aurora Power\exeuv_aurora_20160525_lv03_LT00-24_dt00010_vr01_00.fits':
            aurora_counter = aurora_counter + 2
        aurora_counter = aurora_counter + 1
        aurora_power = time_series_data.field('TPOW1190A')
        for j in aurora_power:
            aurora_power_all = np.append(aurora_power_all, j)   
        for k in aurora_time:
            aurora_time_all = np.append(aurora_time_all, k)

    for l in torus_groups_list:
        hdul = fits.open(fpath(l))
        time_series_data = hdul[2].data
        torus_time = time_series_data.field('SECOFDAY') + 86400 * torus_counter
        if l == r'C:datasets/Hisaki/Torus Power\exeuv_torus_20160525_lv03_LT00-24_dt00010_vr01_00.fits':
            torus_counter = torus_counter + 2
        torus_counter = torus_counter + 1
        power_dawn = time_series_data.field('TPOW0710ADAWN')
        power_dusk = time_series_data.field('TPOW0710ADUSK')
        for m in power_dawn:
            power_dawn_all = np.append(power_dawn_all, m)
        for n in power_dusk:
            power_dusk_all = np.append(power_dusk_all, n)
        for o in torus_time:
            torus_time_all = np.append(torus_time_all, o)

    pd.options.display.max_rows = 9999

    sw_filename = r"datasets\Solar_Wind.txt"
    table = pd.read_csv(fpath(sw_filename), header=0,sep=r'\s+')
    table['Time_UT'] = pd.to_datetime(table['Time_UT'], format='%Y-%m-%dT%H:%M:%S.%f') # see https://strftime.org/ for format codes

    f, ax = plt.subplots(3)

    ax[0].set_xlabel('Time (Days)')
    ax[0].set_ylabel('Aurora Power (GW)')

    ax[1].set_xlabel('Time (Days)')
    ax[1].set_ylabel('Torus Power (GW)')

    ax[2].set_xlabel('Time (Days)')
    ax[2].set_ylabel('Dyanmic Pressure (nPa)')

    ax[0].scatter(aurora_time_all, aurora_power_all, s = 0.5, label = 'Aurora Power')
    ax[1].scatter(torus_time_all, power_dawn_all, s = 0.5, label = 'Dawn')
    ax[1].scatter(torus_time_all, power_dusk_all, s = 0.5, label = 'Dusk')
    ax[2].scatter(table['Time_UT'].to_list(), table['jup_sw_pdyn'].to_list(), s = 0.5, label = 'Solar Wind Pressure')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    ax[0].set_title('Aurora Power, Io Dawn/Dusk Power and Solar Wind Power over Time')


    plt.tight_layout

    a = table['Time_UT'].to_list()

    ax[0].set_xticks(np.arange(0, 86400 * aurora_counter + 1, step = 86400 * 5))
    ax[1].set_xticks(np.arange(0, 86400 * torus_counter + 1, step = 86400 * 5))
    ax[2].set_xticks(a[0::24*5])

    ax[0].xaxis.set_major_formatter(plt.FuncFormatter(date_aligner))
    ax[1].xaxis.set_major_formatter(plt.FuncFormatter(date_aligner))

    plt.show()


if __name__ == "__main__":
    # file of interest (assumes file is in same directory as this script - if it isn't, you need to add the path to the file)
    aurora_filename = r"C:datasets\Hisaki\Aurora Power\exeuv_aurora_20160508_lv03_LT00-24_dt00010_vr01_00.fits"
    aurora_plot(aurora_filename)
    # file of interest (assumes file is in same directory as this script - if it isn't, you need to add the path to the file)
    torus_filename = fpath(r"datasets\Hisaki\Torus Power\exeuv_torus_20160507_lv03_LT00-24_dt00010_vr01_00.fits")
    torus_plot(torus_filename)
