#ignore!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 13:01:48 2023
Version of prof_final_comparer cropped to show only images without the results
of the fitting algorithm
@author: dmoral
"""
#import all the necessary modules
import os
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import matplotlib.ticker as ticker
import numpy as np

from dateutil.parser import parse
import datetime as dt
#import pandas as pd
from matplotlib.colors import LogNorm

#from scipy import stats
import glob
import imageio
from astropy.io import fits
from tqdm import tqdm 
from .const import fpath

    
#this is the funcion used for plotting the images
def moind(image_data:np.ndarray, header, filename, prefix, crop = 1, rlim = 30, fixed = 'lon', hemis='North', full=True, regions=False,**kwargs):
    """
            Generate a polar projection plot of an image with various customization options.
            Parameters:
            -----------
            image_data : numpy.ndarray
                2D array representing the image data, typically astropy.io.fits data
            header : dict
                Dictionary containing header information such as CML, DECE, EXPT, etc.
            filename : str
                The name of the file to be saved.
            prefix : str
                Prefix to be used in the saved file name.
            dpi : int, optional
                Dots per inch for the saved image (default is 300).
            crop : int, optional
                Factor to crop the image (default is 1, no cropping).
            rlim : int, optional
                Radial limit for the plot (default is 30).
            fixed : str, optional
                Fixed parameter for the plot, either 'lon' or 'lt' (default is 'lon').
            hemis : str, optional
                Hemisphere to be plotted, either 'North' or 'South' (default is 'North').
            full : bool, optional
                Whether to plot the full circle or half circle (default is True).
            regions : bool, optional
                Whether to plot specific regions (default is False).
            photo : int, optional
                Placeholder parameter (default is 0).
            Returns:
            --------
            None
    """
    
    # Scatter plot:
    cml = header['CML'] # store CML value
    dece = header['DECE']
    exp_time = header['EXPT']
    iolon = header['IOLON']
    iolon1 = header['IOLON1']
    iolon2 = header['IOLON2']
    eulon = header['EULON']
    eulon1 = header['EULON1']
    eulon2 = header['EULON2']
    galon = header['GALON']
    galon1 = header['GALON1']
    galon2 = header['GALON2']
    
    #Jupiter times
    start_time = parse(header['UDATE'])     # create datetime object
    try:
        dist_org = header['DIST_ORG']
        ltime = dist_org*1.495979e+11/299792458 #c.au/c.c
        lighttime = dt.timedelta(seconds=ltime)
    except ValueError:
        lighttime = dt.timedelta(seconds=2524.42) 
    exposure = dt.timedelta(seconds=exp_time)
    
    start_time_jup = start_time - lighttime       # correct for light travel time
    end_time_jup = start_time_jup + exposure      # end of exposure time
    mid_ex_jup = start_time_jup + (exposure/2.)   # mid-exposure
    
    #plot
    latbins = np.radians(np.linspace(-90, 90, num=image_data.shape[0]))
    lonbins = np.radians(np.linspace(0, 360, num=image_data.shape[1]))
    
    #polar variables
    rho   = np.linspace(0,180, num=int(image_data.shape[0]))# colatitude vector with image pixel resolution steps
    theta = np.linspace(0,2*np.pi,num=image_data.shape[1])
    if hemis == "South" or hemis == "south" or hemis == "S" or hemis == "s":
        rho = rho[::-1]      #if south, changes the orientation 

    #creating the mask
    mask = np.zeros((int(image_data.shape[0]),image_data.shape[1]))   # rows by columns
    cmlr = np.radians(cml)                  # convert CML to radians
    dec = np.radians(dece)        # convert declination angle to radians
    
    #filling the mask
    for i in range(0,mask.shape[0]):
        mask[i,:] = np.sin(latbins[i])*np.sin(dec) + np.cos(latbins[i])*np.cos(dec)*np.cos(lonbins-cmlr)
    
    mask = np.flip(mask,axis=1) # flip the mask horizontally, not sure why this is needed
    cliplim = np.cos(np.radians(89))       # set a minimum required vector normal surface-sun angle
    clipind = np.squeeze([mask >= cliplim])
    
    #applying the mask
    image_data[clipind == False] = np.nan

#   KR_MIN = cliplim
    #aa[aa < KR_MIN] = cliplim
##########################################################################
    #plotting the polar projection of the image
    plt.figure(figsize=(7,6))
        
    ax = plt.subplot(projection='polar')
    radials = np.linspace(0,rlim,6,dtype='int')
