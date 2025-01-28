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
from const import fpath

    
#this is the funcion used for plotting the images
def moind(aa, header, filename, prefix, dpi = 300, crop = 1, rlim = 30, fixed = 'lon',
          hemis = 'North', full = True, moonfp = True, regions = False, photo=0):
    
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
    latbins = np.radians(np.linspace(-90, 90, num=aa.shape[0]))
    lonbins = np.radians(np.linspace(0, 360, num=aa.shape[1]))
    
    #polar variables
    rho   = np.linspace(0,180, num=int(aa.shape[0]))# colatitude vector with image pixel resolution steps
    theta = np.linspace(0,2*np.pi,num=aa.shape[1])
    if hemis == "South" or hemis == "south" or hemis == "S" or hemis == "s":
        rho = rho[::-1]      #if south, changes the orientation 

    #creating the mask
    mask = np.zeros((int(aa.shape[0]),aa.shape[1]))   # rows by columns
    cmlr = np.radians(cml)                  # convert CML to radians
    dec = np.radians(dece)        # convert declination angle to radians
    
    #filling the mask
    for i in range(0,mask.shape[0]):
        mask[i,:] = np.sin(latbins[i])*np.sin(dec) + np.cos(latbins[i])*np.cos(dec)*np.cos(lonbins-cmlr)
    
    mask = np.flip(mask,axis=1) # flip the mask horizontally, not sure why this is needed
    cliplim = np.cos(np.radians(89))       # set a minimum required vector normal surface-sun angle
    clipind = np.squeeze([mask >= cliplim])
    
    #applying the mask
    aa[clipind == False] = np.nan

#   KR_MIN = cliplim
    #aa[aa < KR_MIN] = cliplim
##########################################################################
    #plotting the polar projection of the image
    plt.figure(figsize=(7,6))
        
    ax = plt.subplot(projection='polar')
    radials = np.linspace(0,rlim,6,dtype='int')
    radials = np.arange(0,rlim,10,dtype='int')

    #shifting the image to have CML pointing southwards in the image
    if fixed == 'lon':
        image_centred = aa
        im_flip = np.flip(image_centred,0) #reverse the image along the longitudinal (x, theta) axis
    #cropping the image
        if hemis == "South" or hemis == "south" or hemis == "S" or hemis == "s":
            corte = im_flip[:(int((aa.shape[0])/crop)),:] #if crop=1, nothing changes # NO
            corte = np.roll(corte,180*4,axis=1)
            plt.plot(np.roll([np.radians(180-cml),np.radians(180-cml)],180*4),[0, 180], 'r--', lw=1.2) #cml
            plt.text(np.radians(180-cml), 3+rlim, 'CML', fontsize=11, color='r',
                     horizontalalignment='center', verticalalignment='center', fontweight='bold') 
        else: # if North
            corte = im_flip[:(int((aa.shape[0])/crop)),:] #if crop=1, nothing changes # NO
            plt.plot(np.roll([np.radians(360-cml),np.radians(360-cml)],180*4),[0, 180], 'r--', lw=1.2) #cml
            plt.text(np.radians(360-cml), 3+rlim, 'CML', fontsize=11, color='r',
                     horizontalalignment='center', verticalalignment='center', fontweight='bold') 
            
        if full == True:
            if hemis == "South" or hemis == "south" or hemis == "S" or hemis == "s":
                ax.set_xticklabels(['180°','','','','','','','','','90°','','','','','','','','',
                                    '0°','','','','','','','','','270°','','','','','','','',''], fontweight='bold')   
                ax.set_xticks(np.linspace(0,2*np.pi,37))                      # but set grid spacing#
                poshem = np.radians(45) #position of the "S" marker
    
            else: # if North
                ax.set_xticklabels(['0°','','','','','','','','','270°','','','','','','','','',
                                    '180°','','','','','','','','','90°','','','','','','','',''], fontweight='bold')   
                ax.set_xticks(np.linspace(0,2*np.pi,37))                      # but set grid spacing
                poshem = np.radians(45) #position of the "N" marker
                #ax.set_xticklabels(['0°','','270°','','180°','','90°'], fontweight='bold')
            
            ytickl = []
            for i in radials:
                ytickl.append(str(i)+'°')

            ax.set_yticklabels(ytickl,color='w',fontsize=10)#, weight='bold') # turn off auto lat labels
            ax.set_rticks(np.arange(radials[1],rlim,10,dtype='int'))           
            ax.set_rlabel_position(0)   #position of the radial labels  
            shrink = 1. #size of the colorbar
            possub = 1.05 #position in the y axis of the subtitle
                   
        else:
            if hemis == "South" or hemis == "south" or hemis == "S" or hemis == "s":
                ax.set_xticklabels(['90°','45°','0°','315°','270°'], fontweight='bold')
                ax.set_xticks(np.linspace(np.pi/2,3*np.pi/2,5))                       # but set grid spacing
                poshem = np.radians(-135) #position of the "S" marker
    
            else:  # if North
                ax.set_xticklabels(['270°','225°','180°','135°','90°'], fontweight='bold')
                ax.set_xticks(np.linspace(np.pi/2,3*np.pi/2,5))  
                poshem = np.radians(135) #position of the "N" marker
            
            shrink = 0.75 #size of the colorbar
            possub = 1.03 #position in the y axis of the subtitle
            ax.set_thetalim([np.pi/2,3*np.pi/2])
            ax.set_yticklabels(['',str(radials[1])+'°',str(radials[2])+'°',str(radials[3])+'°',
                                str(radials[4])+'°',str(radials[5])+'°'],color='w',fontsize=10)#, weight='bold') # turn off auto lat labels
            ax.set_rticks(np.linspace(radials[1],rlim,5,dtype='int'))                       # but set grid spacing
        ax.set_theta_zero_location("N")                   # set angle 0.0 to top of plot

    elif fixed == 'lt':
        image_centred = np.roll(aa,int(cml-180.)*4,axis=1)
        im_flip = np.flip(image_centred,0)
        corte = im_flip[:(int((aa.shape[0])/crop)),:] #if crop=1, nothing changes
            
        if full == True:
            ax.set_yticklabels(['',str(radials[1])+'°',str(radials[2])+'°',str(radials[3])+'°',
                str(radials[4])+'°',str(radials[5])+'°'], color='w', fontsize=10) # turn off auto lat labels
            ax.set_rticks(np.linspace(radials[1],rlim,5,dtype='int'))                       # but set grid spacing
            ax.set_xticklabels(['00','','','03','','','06','','','09','','','12',
                '','','15','','','18','','','21','',''], fontweight='bold')
            ax.set_xticks(np.linspace(0,2*np.pi,25))                      # but set grid spacing
            ax.set_rlabel_position(0)         
            shrink = 1. #size of the colorbar
            possub = 1.05 #position in the y axis of the subtitle

            if hemis == "South" or hemis == "south" or hemis == "S" or hemis == "s":
                ax.set_theta_zero_location("N")                   # set angle 0.0 to bottom of plot
                poshem = np.radians(45) #position of the "S" marker
            else:  # if North
                ax.set_theta_zero_location("N")                   # set angle 0.0 to bottom of plot
                poshem = np.radians(45) #position of the "N" marker
        else: #if not full, but half circle
            ax.set_xticklabels(['06','9','12','15','18'], fontweight='bold')
            ax.set_xticks(np.linspace(np.pi/2,3*np.pi/2,5))                       # but set grid spacing
            ax.set_thetalim([np.pi/2,3*np.pi/2])
            ax.set_yticklabels(['',str(radials[1]),'',str(radials[3]),'',str(radials[5])],color='w', fontsize=10) # turn off auto lat labels
            ax.set_rticks(np.linspace(radials[1],rlim,5,dtype='int'))                       # but set grid spacing
            shrink = 0.75 #size of the colorbar
            possub = 1.02 #position in the y axis of the subtitle
            poshem = np.radians(135) #position of the "S" marker

    ax.set_facecolor('k') #black background
    ax.set_rlim([0,rlim]) # max colat range
    ax.tick_params(axis='both',pad=2.)    # shift position of LT labels
    plt.rgrids(radials)#, color='white')
    filename = filename[-51:]
    

    #Naming variables
    v = filename[26:28] 
    visita = filename[-51:-5] 
    #if moonfp == True:
        #visita = 'mfp_'+visita
    doy = filename[7:10]
    tinti = int(filename[20:24])
    tint = str(tinti)
    hora = filename[11:19]
    year = filename[4:6]
    visit = 'v' + str(v) + '_20' + str(year)

    #one of the two titles for every plot
    plt.suptitle(f'Visit {prefix}{v} (DOY: {doy}/20{year}, {hora[0:2]}:{hora[3:5]}:{hora[6:]})', y=0.99, fontsize=14)
    cmlround = np.round(cml, decimals=1)
    
    #the other title + the 0° longitudinal meridian for the LT fixed case
    if fixed == 'lon':
        plt.title(f'Integration time={tint} seconds. CML: {cmlround}°', y=possub, fontsize=12)
    elif fixed == 'lt':
        plt.title(f'Fixed LT. Integration time={tint} s. CML: {cmlround}°', fontsize=12)
        plt.text(np.radians(cml)+np.pi, 4+rlim, '0°', color='coral', fontsize=12,
                 horizontalalignment='center', verticalalignment='bottom', fontweight='bold')
        plt.plot([np.radians(cml)+np.pi,np.radians(cml)+np.pi],[0, 180], color='coral', \
                  path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],\
                  linestyle='-.', lw=1) #prime meridian (longitude 0)

    #Actual plot and colorbar (change the vmin and vmax to play with the limits
    #of the colorbars, recommended to enhance/saturate certain features)
    if int(tint) < 30:
        plt.pcolormesh(theta,rho[:(int((aa.shape[0])/crop))],corte,norm=LogNorm(vmin=10., vmax=1500.), cmap='inferno')
        cbar = plt.colorbar(ticks=[10.,40.,100.,200.,400.,800.,1500.], shrink=shrink, pad=0.06)
        cbar.ax.set_yticklabels(['10','40','100','200','400','800','1500'])
    else:
        plt.pcolormesh(theta,rho[:(int((aa.shape[0])/crop))],corte,norm=LogNorm(vmin=10., vmax=3000.))
        cbar = plt.colorbar(ticks=[10.,40.,100.,200.,400.,1000.,3000.], shrink=shrink, pad=0.06)
        cbar.ax.set_yticklabels(['10','40','100','200','400','1000','3000'])

#####################################################
    #Title for the colorbar    
    cbar.ax.set_ylabel('Intensity [kR]', rotation=270.)
    
    #Grids (major and minor)
    plt.grid(True, which='major', color='w', alpha=0.6, linestyle='-')
    plt.minorticks_on()
    plt.grid(True, which='minor', color='w', alpha=0.2, linestyle='--')
    
    #stronger meridional lines for the 0, 90, 180, 270 degrees:
    plt.plot([np.radians(0),np.radians(0)],[0, 180], 'w', lw=0.9) 
    plt.plot([np.radians(90),np.radians(90)],[0, 180], 'w', lw=0.9) 
    plt.plot([np.radians(180),np.radians(180)],[0, 180], 'w', lw=0.9) 
    plt.plot([np.radians(270),np.radians(270)],[0, 180], 'w', lw=0.9) 

    #deprecated variable (maybe useful for the South/LT fixed definition?)
    shift = 0# cml-180.

    #regions delimitation (lat and lon)
    updusk = np.linspace(np.radians(205+shift),np.radians(170+shift),200)
    lon_dawn = np.linspace(np.radians(180+shift),np.radians(130+shift),200)
    uplat_dawn = np.linspace(33, 15, 200)
    downlat_dawn = np.linspace(39, 23, 200)
    
    lon_noon_a = np.linspace(np.radians(205+shift),np.radians(190+shift),100)
    lon_noon_b = np.linspace(np.radians(190+shift),np.radians(170+shift),100)
    downlat_noon_a = np.linspace(28, 32, 100)
    downlat_noon_b = np.linspace(32, 27, 100)
    
    #print which hemisphere are we in:
    plt.text(poshem, 1.3*rlim, str(hemis).capitalize(), fontsize=21, color='k', 
             horizontalalignment='center', verticalalignment='center', fontweight='bold') 
    
    #drawing the regions (if marked; only available for the North so far)
    if regions == True:
        if fixed == 'lon':
            if hemis == "North" or hemis == "north" or hemis == "N" or hemis == "n":  
                #dusk boundary
                plt.plot([np.radians(205+shift), np.radians(205+shift)], [20, 10], 'r-', lw=1.5)
                plt.plot([np.radians(170+shift), np.radians(170+shift)], [10, 20], 'r-', lw=1.5)
                plt.plot(updusk, 200*[10], 'r-', lw=1.5)
                plt.plot(updusk, 200*[20], 'r-', lw=1.5)
                #dawn boundary
                plt.plot([np.radians(130+shift), np.radians(130+shift)], [23, 15], 'k-', lw=1)
                plt.plot([np.radians(180+shift), np.radians(180+shift)], [33, 39], 'k-', lw=1)#changed from 54 to 39 to close the polygon
                plt.plot(lon_dawn, uplat_dawn, 'k-', lw=1)
                plt.plot(lon_dawn, downlat_dawn, 'k-', lw=1)
                #noon boundary
                plt.plot([np.radians(205+shift), np.radians(205+shift)], [22, 28], 'y-', lw=1.5)    
                plt.plot([np.radians(170+shift), np.radians(170+shift)], [27, 22], 'y-', lw=1.5)
                plt.plot(lon_noon_a, downlat_noon_a, 'y-', lw=1.5)
                plt.plot(lon_noon_b, downlat_noon_b, 'y-', lw=1.5)
                plt.plot(updusk, 200*[22], 'y-', lw=1.5)
                #polar boundary
                plt.plot([np.radians(205+shift), np.radians(205+shift)], [10, 28], 'w--', lw=1)
                plt.plot(lon_noon_a, downlat_noon_a, 'w--', lw=1)
                plt.plot(lon_noon_b, downlat_noon_b, 'w--', lw=1)
                plt.plot([np.radians(170+shift), np.radians(170+shift)], [27, 10], 'w--', lw=1)
                plt.plot(updusk, 200*[10], 'w--', lw=1)
            
            #elif hemis == "South" or hemis == "south" or hemis == "S" or hemis == "s":
            # !!! not defined yet
            
        #regions for the LT fixed case (only North so far)
        elif fixed == 'lt':
            #region delimitation
            updusk = np.linspace(np.radians(205+shift),np.radians(170+shift),200)
            lon_dawn = np.linspace(np.radians(180+shift),np.radians(130+shift),200)
            uplat_dawn = np.linspace(33, 15, 200)
            downlat_dawn = np.linspace(39, 23, 200)
            
            lon_noon_a = np.linspace(np.radians(205+shift),np.radians(190+shift),100)
            lon_noon_b = np.linspace(np.radians(190+shift),np.radians(170+shift),100)
            downlat_noon_a = np.linspace(28, 32, 100)
            downlat_noon_b = np.linspace(32, 27, 100)
            
            if hemis == "North" or hemis == "north" or hemis == "N" or hemis == "n":
                #dusk boundary
                plt.plot([np.radians(205+shift), np.radians(205+shift)], [20, 10], 'r-', lw=1.5)
                plt.plot([np.radians(170+shift), np.radians(170+shift)], [10, 20], 'r-', lw=1.5)
                plt.plot(updusk, 200*[10], 'r-', lw=1.5)
                plt.plot(updusk, 200*[20], 'r-', lw=1.5)
                #dawn boundary
                plt.plot([np.radians(130+shift), np.radians(130+shift)], [23, 15], 'b-', lw=1)
                plt.plot([np.radians(180+shift), np.radians(180+shift)], [33, 39], 'b-', lw=1)#changed from 54 to 39 to close the polygon
                plt.plot(lon_dawn, uplat_dawn, 'b-', lw=1)
                plt.plot(lon_dawn, downlat_dawn, 'b-', lw=1)
            
                #noon boundary
                plt.plot([np.radians(205+shift), np.radians(205+shift)], [22, 28], 'y-', lw=1.5)    
                plt.plot([np.radians(170+shift), np.radians(170+shift)], [27, 22], 'y-', lw=1.5)
                plt.plot(lon_noon_a, downlat_noon_a, 'y-', lw=1.5)
                plt.plot(lon_noon_b, downlat_noon_b, 'y-', lw=1.5)
                plt.plot(updusk, 200*[22], 'y-', lw=1.5)
                #polar boundary
                plt.plot([np.radians(205+shift), np.radians(205+shift)], [10, 28], 'w--', lw=1)
                plt.plot(lon_noon_a, downlat_noon_a, 'w--', lw=1)
                plt.plot(lon_noon_b, downlat_noon_b, 'w--', lw=1)
                plt.plot([np.radians(170+shift), np.radians(170+shift)], [27, 10], 'w--', lw=1)
                plt.plot(updusk, 200*[10], 'w--', lw=1)
            
            #elif hemis == "South" or hemis == "south" or hemis == "S" or hemis == "s":
            # !!! not defined yet
            
#################################################################################
                # MUST SHIFT THE MOONFP FOR THE LT FIXED CASE #
#################################################################################

#     #drawing the moon footprints 
#    # if moonfp == True:
#         #retrieve their expected longitude and latitude (from Hess et al., 2011)
#     #    nlonio, ncolatio, slonio, scolatio, nloneu, ncolateu, sloneu, scolateu, nlonga, ncolatga, slonga, scolatga = moonfploc(iolon,eulon,galon)
#      #   nlonio1, ncolatio1, slonio1, scolatio1, nloneu1, ncolateu1, sloneu1, scolateu1, nlonga1, ncolatga1, slonga1, scolatga1 = moonfploc(iolon1,eulon1,galon1)
#     #    nlonio2, ncolatio2, slonio2, scolatio2, nloneu2, ncolateu2, sloneu2, scolateu2, nlonga2, ncolatga2, slonga2, scolatga2 = moonfploc(iolon2,eulon2,galon2)
       
#         #plot a colored mark in their expected location, together with their name
#         if fixed == 'lon':
#             if hemis == "North" or hemis == "north" or hemis == "N" or hemis == "n":
#                 #we define some intervals for plotting the moon footprints because if they
#                 #are supposed to be way inside the "night" hemisphere (only within +-120degrees
#                 #from CML), if not, we do not plot them
#                 if abs(cml-nlonio1) < 120 or abs(cml-nlonio1) > 240:
#                     plt.plot([2*np.pi-(np.radians(nlonio1)),2*np.pi-(np.radians(nlonio2))],[ncolatio, ncolatio], 'k-', lw=4)
#                     plt.plot([2*np.pi-(np.radians(nlonio1)),2*np.pi-(np.radians(nlonio2))],[ncolatio, ncolatio], color='gold', linestyle='-', lw=2.5)
#                     plt.text(2*np.pi-(np.radians(nlonio)), 3.5+ncolatio, 'IO', color='gold', fontsize=10,  fontweight='bold',alpha=0.5,\
#                              path_effects=[patheffects.withStroke(linewidth=1, foreground='black')], horizontalalignment='center', verticalalignment='center')
                
#                 if abs(cml-nloneu1) < 120 or abs(cml-nloneu1) > 240:
#                     plt.plot([2*np.pi-(np.radians(nloneu1)),2*np.pi-(np.radians(nloneu2))],[ncolateu, ncolateu], 'k-', lw=4)
#                     plt.plot([2*np.pi-(np.radians(nloneu1)),2*np.pi-(np.radians(nloneu2))],[ncolateu, ncolateu], color='aquamarine', linestyle='-', lw=2.5)
#                     plt.text(2*np.pi-(np.radians(nloneu)), 3.5+ncolateu, 'EUR', color='aquamarine', fontsize=10, fontweight='bold',alpha=0.5,
#                              path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],\
#                              horizontalalignment='center', verticalalignment='center')
                
#                 if abs(cml-nlonga1) < 120 or abs(cml-nlonga1) > 250:
#                     plt.plot([2*np.pi-(np.radians(nlonga1)),2*np.pi-(np.radians(nlonga2))],[ncolatga, ncolatga], 'k-', lw=4)
#                     plt.plot([2*np.pi-(np.radians(nlonga1)),2*np.pi-(np.radians(nlonga2))],[ncolatga, ncolatga], 'w-', lw=2.5)
#                     plt.text(2*np.pi-(np.radians(nlonga)), 3.5+ncolatga, 'GAN', color='w', fontsize=10, fontweight='bold',alpha=0.5,
#                              path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],\
#                              horizontalalignment='center', verticalalignment='center')
#             else: #if we are in the Southern hemisphere
#                 if abs(cml-slonio1) < 120 or abs(cml-slonio1) > 240:
#                     plt.plot([(np.radians(180-slonio1)),(np.radians(180-slonio2))],[scolatio, scolatio], 'k-', lw=4)
#                     plt.plot([(np.radians(180-slonio1)),(np.radians(180-slonio2))],[scolatio, scolatio], color='gold', linestyle='-', lw=2.5)
#                     plt.text((np.radians(180-slonio)), 3.5+scolatio, 'IO', color='gold', fontsize=10, alpha=0.5,
#                              path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],\
#                              horizontalalignment='center', verticalalignment='center', fontweight='bold')
                
#                 if abs(cml-sloneu1) < 120 or abs(cml-sloneu1) > 240:
#                     plt.plot([(np.radians(180-sloneu1)),(np.radians(180-sloneu2))],[scolateu, scolateu], 'k-', lw=4)
#                     plt.plot([(np.radians(180-sloneu1)),(np.radians(180-sloneu2))],[scolateu, scolateu], color='aquamarine', linestyle='-', lw=2.5)
#                     plt.text((np.radians(180-sloneu)), 3.5+scolateu, 'EUR', color='aquamarine', fontsize=10, alpha=0.5,
#                              path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],\
#                              horizontalalignment='center', verticalalignment='center', fontweight='bold')
                
#                 if abs(cml-slonga1) < 120 or abs(cml-slonga1) > 240:
#                     plt.plot([(np.radians(180-slonga1)),(np.radians(180-slonga2))],[scolatga, scolatga], 'k-', lw=4)
#                     plt.plot([(np.radians(180-slonga1)),(np.radians(180-slonga2))],[scolatga, scolatga], 'w-', lw=2.5)
#                     plt.text((np.radians(180-slonga)), 3.5+scolatga, 'GAN', color='w', fontsize=10, alpha=0.5,
#                              path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],\
#                              horizontalalignment='center', verticalalignment='center', fontweight='bold')
    
#         elif fixed == 'lt':
#             if hemis == "North" or hemis == "north" or hemis == "N" or hemis == "n":
#                 if abs(cml-nlonio1) < 120 or abs(cml-nlonio1) > 240:
#                     plt.plot([(np.radians(180+cml-nlonio1)),(np.radians(180+cml-nlonio2))],[ncolatio, ncolatio], 'k-', lw=4)
#                     plt.plot([(np.radians(180+cml-nlonio1)),(np.radians(180+cml-nlonio2))],[ncolatio, ncolatio], color='gold', linestyle='-', lw=2.5)
#                     plt.text((np.radians(180+cml-nlonio)), 3.5+ncolatio, 'IO ', color='gold', fontsize=10,  fontweight='bold',alpha=0.5,
#                              path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],\
#                              horizontalalignment='center', verticalalignment='center')
                
#                 if abs(cml-nloneu1) < 120 or abs(cml-nloneu1) > 240:
#                     plt.plot([2*np.pi-(np.radians(180+cml-nloneu1)),2*np.pi-(np.radians(180+cml-nloneu2))],[ncolateu, ncolateu], 'k-', lw=4)
#                     plt.plot([2*np.pi-(np.radians(180+cml-nloneu1)),2*np.pi-(np.radians(180+cml-nloneu2))],[ncolateu, ncolateu], color='aquamarine', linestyle='-', lw=2.5)
#                     plt.text((np.radians(180+cml-nloneu)), 3.5+ncolateu, 'EUR', color='aquamarine', fontsize=10, fontweight='bold',alpha=0.5,
#                              path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],\
#                              horizontalalignment='center', verticalalignment='center')
                
#                 if abs(cml-nlonga1) < 120 or abs(cml-nlonga1) > 240:
#                     plt.plot([(np.radians(180+cml-nlonga1)),(np.radians(180+cml-nlonga2))],[ncolatga, ncolatga], 'k-', lw=4)
#                     plt.plot([(np.radians(180+cml-nlonga1)),(np.radians(180+cml-nlonga2))],[ncolatga, ncolatga], 'w-', lw=2.5)
#                     plt.text((np.radians(180+cml-nlonga)), 3.5+ncolatga, 'GAN', color='w', fontsize=10, fontweight='bold',alpha=0.5,
#                              path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],\
#                              horizontalalignment='center', verticalalignment='center')
       
#             else: # South hemisphere
#                 if abs(cml-slonio1) < 120 or abs(cml-slonio1) > 240:
#                     plt.plot([(np.radians(180+cml-slonio1)),(np.radians(180+cml-slonio2))],[scolatio, scolatio], 'k-', lw=4)
#                     plt.plot([(np.radians(180+cml-slonio1)),(np.radians(180+cml-slonio2))],[scolatio, scolatio], color='gold', linestyle='-', lw=2.5)
#                     plt.text((np.radians(180+cml-slonio)), 3.5+scolatio, 'IO ', color='gold', fontsize=10, alpha=0.5,
#                              path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],\
#                              horizontalalignment='center', verticalalignment='center', fontweight='bold')
                
#                 if abs(cml-sloneu1) < 120 or abs(cml-sloneu1) > 240:
#                     plt.plot([(np.radians(180+cml-sloneu1)),(np.radians(180+cml-sloneu2))],[scolateu, scolateu], 'k-', lw=4)
#                     plt.plot([(np.radians(180+cml-sloneu1)),(np.radians(180+cml-sloneu2))],[scolateu, scolateu], color='aquamarine', linestyle='-', lw=2.5)
#                     plt.text((np.radians(180+cml-sloneu)), 3.5+scolateu, 'EUR', color='aquamarine', fontsize=10,alpha=0.5,
#                              path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],\
#                              horizontalalignment='center', verticalalignment='center', fontweight='bold')
                
#                 if abs(cml-slonga1) < 120 or abs(cml-slonga1) > 240:
#                     plt.plot([(np.radians(180+cml-slonga1)),(np.radians(180+cml-slonga2))],[scolatga, scolatga], 'k-', lw=4)
#                     plt.plot([(np.radians(180+cml-slonga1)),(np.radians(180+cml-slonga2))],[scolatga, scolatga], 'w-', lw=2.5)
#                     plt.text((np.radians(180+cml-slonga)), 3.5+scolatga, 'GAN', color='w', fontsize=10,alpha=0.5,
#                              path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],\
#                              horizontalalignment='center', verticalalignment='center', fontweight='bold')
# #####################################################################
    #defining the final filename, adding sufixes depending n the stuff we are showing
   
    def ensure_dir(file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
    
    
    def savedataimg(file_path,fixed='detailtype'):
        ensure_dir(file_path)
        if fixed == 'lon':
            ensure_dir(file_path/lon)
            plt.savefig(fpath(f'file_path/lon/{tint}s/mo_{namesave}_fixlon.jpg'), dpi=dpi)
        elif fixed == 'lt':
            ensure_dir(fpath(f'file_path/lt'))
            plt.savefig(fpath(f'file_path/lt/{tint}s_fixedlt/mo_{visita}_fixedlt.jpg', dpi=300))
    namesave = str(filename)
    
    savedataimg('HST', fixed='lon')

#ORGINAL CODE
    #if fixed == 'lon':
    #    if not os.path.exists(fpath(f'pictures/polar/{prefix}{v}/fin/')):
    #        os.makedirs(fpath(f'pictures/polar/{prefix}{v}/fin/'))
    #    if not os.path.exists(fpath(f'pictures/polar/{prefix}{v}/fin/{tint}s/')):
    #        os.makedirs(fpath(f'pictures/polar/{prefix}{v}/fin/{tint}s/'))
    #    print('Name of the saved image is mo_'+str(namesave)+"_fixlon.jpg")
    #    plt.savefig(fpath(f'pictures/polar/{prefix}{v}/fin/{tint}s/mo_{namesave}_fixlon.jpg'), dpi=dpi)
    

    #elif fixed == 'lt':    
    #    if not os.path.exists(fpath(f'pictures/polar/{prefix}{v}/fin/')):
    #        os.makedirs(fpath(f'pictures/polar/{prefix}{v}/fin/'))
    #    if not os.path.exists(fpath(f'pictures/polar/{prefix}{v}/fin/{tint}s_fixedlt/')):
    #        os.makedirs(fpath(f'pictures/polar/{prefix}{v}/fin/{tint}s_fixedlt/'))
    #    print('Name of the saved image is mo_'+str(filename)+"_fixedlt.jpg")
    #    plt.savefig(fpath(f'pictures/polar/{prefix}{v}/fin/{tint}s_fixedlt/mo_{visita}_fixedlt.jpg', dpi=300)) # save location


    #plt.close()

#and this chunk is to call the function:
def multigif(lista, year, prefix, extra, time, radius, moonfp, full, fixed, mf, indf, polarf, secondf):
    for l in lista:
        l = str(l)
        print(f'VISIT {l} \n \n') #just to check out the visit we are plotting
        
        #we grab all the files we are interested in plotting
        arch = '*_v'+ l
        ti = str('/*0'+time+'*')
        ab = glob.glob(fpath(f'datasets/HST'))
        #ab = glob.glob(fpath(f'data/HST/Jupiter/{year}/extract/{extra}'+arch+'/nopolar'+time+ti))
        ab.sort()    
        print(ab)
        #print(f"Length of ab is: {len(ab)}")
        
        #define the image resolution for when there are many images so the 
        #final GIF is not too large
        if int(time) < 100:
            cal = 150
        else:
            cal = 300
            
        #and now we loop to apply the plotting function (moind) to every image
        #in the visit
        for  n,i in tqdm(enumerate(ab)):
            hdulist = fits.open(i)
            header = hdulist[1].header
            image = hdulist[1].data
            try:
                hemis = header['HEMISPH']
            except NameError:
                hemis = str(input('Input hemisphere manually:  ("north" or "south")  '))
            filename = str(i)[-51:-5]
            moind(image, header, filename, prefix, dpi = cal, crop=1, rlim = radius, fixed = fixed,
                      hemis = hemis, full=full, moonfp=moonfp, photo=n, mf=mf, indf=indf, polarf=polarf, secondf=secondf)
            hdulist.close()
            print(f'Image {n} of {str(i)[-51:-5]} created.')
        
        #and now we start creating the GIF
        visita = prefix+arch[-2:]
        imagesgif = []
        
        #durat = input('GIF duration (in seconds): ') #deprecated
        fps = 5
        if int(time) < 50:
            fps = 40
        
        #name of the GIF
        gifname = visita + '_t'+time
        
        print(f'On to the GIF ({gifname})')
        
        nam = str('*0'+time+'*')

        #optional suffixes to the GIF's name
        # + grabbing the correct images recently generated by moind
        if polarf == True:
            if fixed == 'lon':
                gifname += '_flon'
                gifphotos = glob.glob(fpath('pictures/polar/')+visita+'/fin/'+str(time)+'s/mopf'+nam)
            elif fixed == 'lt':
                gifname += '_flt'
                gifphotos = glob.glob(fpath('pictures/polar/')+visita+'/fin/'+str(time)+'s_fixedlt/mopf'+nam)

        else:
            if fixed == 'lon':
                gifname += '_flon'
                gifphotos = glob.glob(fpath('pictures/polar/')+visita+'/fin/'+str(time)+'s/mo_jup'+nam)
            elif fixed == 'lt':
                gifname += '_flt'
                gifphotos = glob.glob(fpath('pictures/polar/')+visita+'/fin/'+str(time)+'s_fixedlt/'+nam)
        
        if moonfp == True:
            gifname += '_mfp'
        
        if full == False:
            gifname += '_half'
        
        #final suffix (termination)
        gifname += '.gif'
        
        #GIF creation:
        gifphotos.sort()
        for file in gifphotos:
            imagesgif.append(imageio.imread(file))

        #saving the GIF
        if not os.path.exists(fpath(f'pictures/gifs/')):
            os.makedirs(fpath(f'pictures/gifs/'))
        imageio.mimsave(fpath('pictures/gifs/') + gifname, imagesgif, fps=fps)
    
#and this last part is the one that must be run every time:
#os.chdir(fpath('python/pypeline/')) ## Not sure why this exists 
#you have to input the year of the visits you are plotting (so cannot mix visits
#from different years in the same "run" of the code, for filepathing reasons)
def input_run():
    year = input("Year of the visit:  \n")
    if year == '2016':
        pre = input('Campaign from Jonny or Denis? (1/2)  ')
        if pre == '1' or pre == 'jonny' or pre == 'j' or pre == 'J' or pre == 'Jonny':
            prefix = 'ocx8'
            extra = 'nichols/' #this may be have to be removed if your directory system did not differentiate between Jonny's and Dennis's campaigns
        else:
            prefix = 'od8k'
            extra = 'grodent/' #this may be have to be removed if your directory system did not differentiate between Jonny's and Dennis's campaigns
    elif year == '2019':
        prefix = 'odxc'
        extra = ''
    elif year == '2021':
        prefix = 'oef4'
        extra = ''
    elif year == '2017' or  year == '2018':
        prefix = 'od8k'
        extra = ''
        
    time = str(input('Exposure time (in seconds: 10, 30, 100...): \n')) #usually 100
    radius = int(input('Max. radius (in degrees of colatitude): \n'))   #usually 40
    moonfp = not bool(input('Moon footprints printed? (Default: Enter for YES)\n')) #usually yes
    full = not bool(input('Show the whole hemisphere? (Default: Enter for YES)\n')) #usually yes
    fixed = str(input('Fix longitude (lon) or Local Time (lt):\n')) 

    #the only part you have to add manually is the particular set of visit numbers you want
    #to plot. If you want only one that is perfectly fine but it must be IN a list
    lista = ['01'] #for example, or lista = ['0v']

    multigif(lista, year, prefix, extra, time, radius, moonfp, full, fixed, mf=0, indf=0, polarf=True, secondf=0) # this is what I need to call


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
