__doc__="""
This module contains functions to generate polar projection plots of Jupiter's image data from FITS files. 
The main function, moind(), generates a polar projection plot of Jupiter's image data from a FITS file. 
The make_gif() function creates a GIF from a directory of FITS files.

adapted from dmoral's original code by the JAR:VIS team.
"""
#import all the necessary modules
#python standard libraries
import os
from dateutil.parser import parse
import datetime as dt
import glob
from typing import List, Tuple, Dict, Any, Union, Optional, Callable

#third party libraries
from astropy.io import fits
import imageio
import matplotlib as mpl
from matplotlib import patheffects as mpl_patheffects
import matplotlib.pyplot as plt
#import matplotlib.patheffects as patheffects
#import matplotlib.ticker as ticker
#from matplotlib.colors import LogNorm
import numpy as np
from tqdm import tqdm 
# local modules
from .const import fpath, fileInfo

try:
    from reading_mfp import moonfploc
except:
    moonfploc = print


def ensure_dir(file_path):
    '''this function checks if the file path exists, if not it will create one'''
    if not os.path.exists(file_path):
            os.makedirs(file_path)
def clock_format(x_rads, pos):
    # x_rads => 0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4, ...
    # returns=> 00, 03, 06, 09, 12, 15, 18, 21,..., 00,03,06,09,12,15,18,21,
    cnum= int(np.degrees(x_rads)/15)
    return f'{cnum:02d}' if cnum%24 != 0 else '00'



def moind(file_location:str=None,save_location:str=None,filename:str='auto', crop:float = 1, rlim:float = 40, fixed:str= 'lon', hemis:str='North', full:bool=True, regions:bool=False,fileinfo:fileInfo=None,fitsdataheader:Tuple[np.ndarray,Dict]=None,**kwargs)->Union[None,mpl.figure.Figure]:  
    """
        Generate a polar projection plot of Jupiter's image data.
        
        Args:
            file_location (str, optional): Path to the FITS file. Defaults to None.
            save_location (str, optional): Directory to save the generated plot. Defaults to None.
            filename (str, optional): Name of the output file. Defaults to 'auto'.
            crop (float, optional): Factor to crop the image. Defaults to 1.
            rlim (float, optional): latitudinal limit for the plot (in degrees). Defaults to 40.
            fixed (str, optional): Fixed parameter, either 'lon' or 'lt'. Defaults to 'lon'.
            hemis (str, optional): Hemisphere, either 'North' or 'South'. Defaults to 'North'.
            full (bool, optional): Whether to display the full plot. Defaults to True.
            regions (bool, optional): Whether to mark regions on the plot. Defaults to False.
            fileinfo (fileInfo, optional): File information object. Defaults to None.
            fitsdataheader (Tuple[np.ndarray, Dict], optional): FITS data and header if user wants to provide their own. Defaults to None.
            **kwargs: Additional keyword arguments passed to plt.savefig()..
        
        Returns:
            matplotlib.figure.Figure: The generated plot figure if 'return' is in kwargs.

        Keyword Args:
            cmap (str,mpl.colors.Colormap, optional): Colormap. Defaults to 'viridis'.
            norm (mpl.colors.Normalize, optional): Normalize object. Defaults to mpl.colors.LogNorm(vmin=ticks[0], vmax=ticks[-1]).
            ticks (List[float], optional): Colorbar ticks. Defaults to [10.,40.,100.,200.,400.,800.,1500.].
            shrink (float, optional): Colorbar shrink factor. Defaults to 1.
            pad (float, optional): Colorbar pad. Defaults to 0.06.
            other kwargs: Additional keyword arguments passed to plt.savefig().
       
        Notes:
            the function must be called with at least ONE of the following:
            
            - file_location (relative preffered)
            - fileinfo (fileInfo object pointing to the file)
            
            where fileinfo takes precedence over file_location.

            `hemis = 'South'` has not been fully implemented.
        
        Examples:
            >>> moind('datasets/HST/v09-may22/jup_16-143-18-41-06_0100_v09_stis_f25srf2_proj.fits', 'pictures/', 'jupiter.jpg')

            >>> moind(fileinfo=fileInfo('datasets/HST/v09-may22/jup_16-143-18-41-06_0100_v09_stis_f25srf2_proj.fits'), save_location='pictures/', filename='jupiter.jpg', cmap='inferno')
    """
    if fileinfo is None:
        f_abs = fpath(file_location)
    elif fileinfo is not None:
        f_abs = fileinfo.absolute_path
    with fits.open(f_abs) as hdulist:
            image_data = hdulist[1].data
            header = hdulist[1].header    
        


    # Scatter plot:
    cml = header['CML'] # store CML value
    dece = header['DECE']
    exp_time = header['EXPT']
    lon = {'io':(header['IOLON'],header['IOLON1'],header['IOLON2']), #! UNUSED
           'eu':(header['EULON'],header['EULON1'],header['EULON2']),
           'ga':(header['GALON'],header['GALON1'],header['GALON2'])}
    is_south = True if hemis.lower()[0] == 's' else False if hemis.lower()[0] == 'n' else 'Err'
    if is_south == 'Err': raise ValueError('Hemisphere not recognized. Please input "North" or "South"')
    is_lon = True if fixed == 'lon' else False if fixed == 'lt' else 'Err'
    if is_lon == 'Err': raise ValueError('Fixed parameter not recognized. Please input "lon" or "lt"')



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
    end_time_jup = start_time_jup + exposure      # end of exposure time #! UNUSED
    mid_ex_jup = start_time_jup + (exposure/2.)   # mid-exposure #! UNUSED
    
    #plot
    latbins = np.radians(np.linspace(-90, 90, num=image_data.shape[0]))
    lonbins = np.radians(np.linspace(0, 360, num=image_data.shape[1]))
    #polar variables
    rho   = np.linspace(0,180, num=int(image_data.shape[0]))# colatitude vector with image pixel resolution steps
    theta = np.linspace(0,2*np.pi,num=image_data.shape[1])
    if is_south:
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
    image_data[clipind == False] = np.nan #applying the mask
    #aa[aa < KR_MIN] = cliplim #   KR_MIN = cliplim

##########################################################################
    #plotting the polar projection of the image
    fig =plt.figure(figsize=(7,6))
    ax = plt.subplot(projection='polar')
    radials = np.arange(0,rlim,10,dtype='int')
    if is_lon: image_centred = image_data
    else: image_centred = np.roll(image_data,int(cml-180.)*4,axis=1) #shifting the image to have CML pointing southwards in the image
    im_flip = np.flip(image_centred,0) # reverse the image along the longitudinal (x, theta) axis
    corte = im_flip[:(int((image_data.shape[0])/crop)),:] # cropping image, if crop=1, nothing changes
    # plotting cml, only for lon
    if is_lon:
        if is_south:
            corte = np.roll(corte,180*4,axis=1)
            rot = 180
        else:
            rot = 360
        ax.plot(np.roll([np.radians(rot-cml),np.radians(rot-cml)],180*4),[0, 180], 'r--', lw=1.2) #cml
        ax.text(np.radians(rot-cml), 3+rlim, 'CML', fontsize=11, color='r', horizontalalignment='center', verticalalignment='center', fontweight='bold') 
    shrink = 1 if full else 0.75 # size of the colorbar
    possub = 1.05 if full else 1.03 if fixed == 'lt' else 1.02 # position in the y axis of the subtitle
    poshem = 45 if full else 135 if any([hemis.lower()[0]=='n', fixed=='lt']) else -135 #position of the "N/S" marker
    # set xticks/thetaticks
    # set radial ticks/ yticks
    ax.set_theta_zero_location("N")   
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=10))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: '{:.0f}°'.format(x))) # set radial labels
    ax.yaxis.set_tick_params(labelcolor='white', ) # set radial labels color
    if is_lon:
        if full:
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=np.pi/2))
            if is_south:    shift_t = lambda x: x # should be 180°, 90°, 0°, 270°
            else:           shift_t = lambda x: 2*np.pi-x # should be 0°, 90°, 180°, 270°
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: '{:.0f}°'.format(np.degrees(shift_t(x))%360)))
        else:
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=np.pi/4))
            #if is_south:    ax.set_xticklabels(['90°','45°','0°','315°','270°'], fontweight='bold') #not sure how to chnge or test this
            #else:           ax.set_xticklabels(['270°','225°','180°','135°','90°'], fontweight='bold')# if NortH
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=2*np.pi/36)) # 
    else:
        # clockticks
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=2*np.pi/8))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=2*np.pi/24))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(clock_format))    
        #if full:            ax.set_xticklabels(['00','03','06','09','12','15','18','21'], fontweight='bold')
        #else:               ax.set_xticklabels(['06','09','12','15','18'], fontweight='bold')
    if full:   
        ax.set_rlabel_position(0)   #position of the radial labels 
    else:
        ax.set_thetalim([np.pi/2,3*np.pi/2])

    ax.set_facecolor('k') #black background
    ax.set_rlim([0,rlim]) # max colat range
    ax.tick_params(axis='both',pad=2.)    # shift position of LT labels
    ax.set_rgrids(radials)#, color='white')

    #Naming variables
    finfo = fileInfo(file_location) if fileinfo is None else fileinfo

    # Titles
    plt.suptitle(f'Visit {finfo.visit} (DOY: {finfo.day}/{finfo.year}, {finfo.datetime})', y=0.99, fontsize=14)#one of the two titles for every plot
    plt.title(f'{"Fixed LT. " if not is_lon else ""}Integration time={finfo.exp} s. CML: {np.round(cml, decimals=1)}°',y=possub, fontsize=12)
        
        
    if not is_lon and full: # meridian line (0°)  
        plt.text(np.radians(cml)+np.pi, 4+rlim, '0°', color='coral', fontsize=12,horizontalalignment='center', verticalalignment='bottom', fontweight='bold')
        ax.plot([np.radians(cml)+np.pi,np.radians(cml)+np.pi],[0, 180], color='coral', path_effects=[mpl_patheffects.withStroke(linewidth=1, foreground='black')], linestyle='-.', lw=1) #prime meridian (longitude 0)

    #Actual plot and colorbar (change the vmin and vmax to play with the limits
    #of the colorbars, recommended to enhance/saturate certain features)
    if 'ticks' in kwargs:
        ticks = kwargs.pop('ticks')
    elif int(finfo.exp) < 30:
        ticks = [10.,40.,100.,200.,400.,800.,1500.]
    else:
        ticks = [10.,40.,100.,200.,400.,1000.,3000.]
    cmap = kwargs.pop('cmap') if 'cmap' in kwargs else 'viridis'
    norm = kwargs.pop('norm') if 'norm' in kwargs else mpl.colors.LogNorm(vmin=ticks[0], vmax=ticks[-1])
    shrink = kwargs.pop('shrink') if 'shrink' in kwargs else shrink
    pad = kwargs.pop('pad') if 'pad' in kwargs else 0.06
    plt.pcolormesh(theta,rho[:(int((image_data.shape[0])/crop))],corte,norm=norm, cmap=cmap)#!5 <- Color of the plot
    cbar = plt.colorbar(ticks=ticks, shrink=shrink, pad=0.06)
    cbar.ax.set_yticklabels([str(int(i)) for i in ticks])

#####################################################
    #Title for the colorbar    
    cbar.ax.set_ylabel('Intensity [kR]', rotation=270.)
    
    #Grids (major and minor)
    ax.grid(True, which='major', color='w', alpha=0.7, linestyle='-')
    plt.minorticks_on()
    ax.grid(True, which='minor', color='w', alpha=0.5, linestyle='-')
    
    #stronger meridional lines for the 0, 90, 180, 270 degrees:
    for i in range(0,4):
        ax.plot([np.radians(i*90),np.radians(i*90)],[0, 180], 'w', lw=0.9)

    #deprecated variable (maybe useful for the South/LT fixed definition?)
    shift = 0# cml-180.
    
    #print which hemisphere are we in:
    ax.text(poshem, 1.3*rlim, str(hemis).capitalize(), fontsize=21, color='k', 
             horizontalalignment='center', verticalalignment='center', fontweight='bold') 
    
    #drawing the regions (if marked; only available for the North so far)
    if regions == True:
        if is_south:
            pass
        else:
            #regions delimitation (lat and lon)
            updusk = np.linspace(np.radians(205),np.radians(170),200) 
            dawn = {k:np.linspace(v[0],v[1],200) for k,v in (('lon',(np.radians(180),np.radians(130))), ('uplat',(33, 15)), ('downlat',(39, 23)))}
            noon_a = {k:np.linspace(v[0],v[1],100) for k,v in (('lon',(np.radians(205),np.radians(190))),('downlat',(28, 32)))} 
            noon_b = {k:np.linspace(v[0],v[1],100) for k,v in (('lon',(np.radians(190),np.radians(170))),('downlat',(32, 27)))}
            # lon_noon_a = noon_a['lon']
            #dusk boundary
            ax.plot([np.radians(205), np.radians(205)], [20, 10], 'r-', lw=1.5) 
            ax.plot([np.radians(170), np.radians(170)], [10, 20], 'r-', lw=1.5)  
            ax.plot(updusk, 200*[10], 'r-', lw=1.5)
            ax.plot(updusk, 200*[20], 'r-', lw=1.5) 
            #dawn boundary
            c = 'k-' if is_lon else 'b-'
            ax.plot([np.radians(130), np.radians(130)], [23, 15], c, lw=1) 
            ax.plot([np.radians(180), np.radians(180)], [33, 39], c, lw=1)
            ax.plot(dawn['lon'], dawn['uplat'], c, lw=1)
            ax.plot(dawn['lon'], dawn['downlat'], c, lw=1)
            #noon boundary
            ax.plot([np.radians(205), np.radians(205)], [22, 28], 'y-', lw=1.5)   
            ax.plot([np.radians(170), np.radians(170)], [27, 22], 'y-', lw=1.5)    
            ax.plot(noon_a['lon'], noon_a['downlat'], 'y-', lw=1.5) 
            ax.plot(noon_b['lon'], noon_b['downlat'], 'y-', lw=1.5) 
            ax.plot(updusk, 200*[22], 'y-', lw=1.5) 
            #polar boundary
            ax.plot([np.radians(205), np.radians(205)], [10, 28], 'w--', lw=1)
            ax.plot([np.radians(170), np.radians(170)], [27, 10], 'w--', lw=1)
            ax.plot(noon_a['lon'], noon_a['downlat'], 'w--', lw=1) 
            ax.plot(updusk, 200*[10], 'w--', lw=1) 
            
        ### TOP
    from reading_mfp import moonfploc
    #   drawing the moon footprints 
        if moonfp == True:
        #retrieve their expected longitude and latitude (from Hess et al., 2011)
        nlonio, ncolatio, slonio, scolatio, nloneu, ncolateu, sloneu, scolateu, nlonga, ncolatga, slonga, scolatga = moonfploc(iolon,eulon,galon)
        nlonio1, ncolatio1, slonio1, scolatio1, nloneu1, ncolateu1, sloneu1, scolateu1, nlonga1, ncolatga1, slonga1, scolatga1 = moonfploc(iolon1,eulon1,galon1)
        nlonio2, ncolatio2, slonio2, scolatio2, nloneu2, ncolateu2, sloneu2, scolateu2, nlonga2, ncolatga2, slonga2, scolatga2 = moonfploc(iolon2,eulon2,galon2)

        #plot a colored mark in their expected location, together with their name
        if fixed == 'lon':
            if hemis == "North" or hemis == "north" or hemis == "N" or hemis == "n":
                #we define some intervals for plotting the moon footprints because if they
                #are supposed to be way inside the "night" hemisphere (only within +-120degrees
                #from CML), if not, we do not plot them
                if abs(cml-nlonio1) < 120 or abs(cml-nlonio1) > 240:
                    plt.plot([2*np.pi-(np.radians(nlonio1)),2*np.pi-(np.radians(nlonio2))],[ncolatio, ncolatio], 'k-', lw=4)
                    plt.plot([2*np.pi-(np.radians(nlonio1)),2*np.pi-(np.radians(nlonio2))],[ncolatio, ncolatio], color='gold', linestyle='-', lw=2.5)
                    plt.text(2*np.pi-(np.radians(nlonio)), 3.5+ncolatio, 'IO', color='gold', fontsize=10,  fontweight='bold',alpha=0.5,\
                             path_effects=[mpl_patheffects.withStroke(linewidth=1, foreground='black')], horizontalalignment='center', verticalalignment='center')

                if abs(cml-nloneu1) < 120 or abs(cml-nloneu1) > 240:
                    plt.plot([2*np.pi-(np.radians(nloneu1)),2*np.pi-(np.radians(nloneu2))],[ncolateu, ncolateu], 'k-', lw=4)
                    plt.plot([2*np.pi-(np.radians(nloneu1)),2*np.pi-(np.radians(nloneu2))],[ncolateu, ncolateu], color='aquamarine', linestyle='-', lw=2.5)
                    plt.text(2*np.pi-(np.radians(nloneu)), 3.5+ncolateu, 'EUR', color='aquamarine', fontsize=10, fontweight='bold',alpha=0.5,
                             path_effects=[mpl_patheffects.withStroke(linewidth=1, foreground='black')],\
                             horizontalalignment='center', verticalalignment='center')

                if abs(cml-nlonga1) < 120 or abs(cml-nlonga1) > 250:
                    plt.plot([2*np.pi-(np.radians(nlonga1)),2*np.pi-(np.radians(nlonga2))],[ncolatga, ncolatga], 'k-', lw=4)
                    plt.plot([2*np.pi-(np.radians(nlonga1)),2*np.pi-(np.radians(nlonga2))],[ncolatga, ncolatga], 'w-', lw=2.5)
                    plt.text(2*np.pi-(np.radians(nlonga)), 3.5+ncolatga, 'GAN', color='w', fontsize=10, fontweight='bold',alpha=0.5,
                             path_effects=[mpl_patheffects.withStroke(linewidth=1, foreground='black')],\
                             horizontalalignment='center', verticalalignment='center')
            else: #if we are in the Southern hemisphere
                if abs(cml-slonio1) < 120 or abs(cml-slonio1) > 240:
                    plt.plot([(np.radians(180-slonio1)),(np.radians(180-slonio2))],[scolatio, scolatio], 'k-', lw=4)
                    plt.plot([(np.radians(180-slonio1)),(np.radians(180-slonio2))],[scolatio, scolatio], color='gold', linestyle='-', lw=2.5)
                    plt.text((np.radians(180-slonio)), 3.5+scolatio, 'IO', color='gold', fontsize=10, alpha=0.5,
                             path_effects=[mpl_patheffects.withStroke(linewidth=1, foreground='black')],\
                             horizontalalignment='center', verticalalignment='center', fontweight='bold')

                if abs(cml-sloneu1) < 120 or abs(cml-sloneu1) > 240:
                    plt.plot([(np.radians(180-sloneu1)),(np.radians(180-sloneu2))],[scolateu, scolateu], 'k-', lw=4)
                    plt.plot([(np.radians(180-sloneu1)),(np.radians(180-sloneu2))],[scolateu, scolateu], color='aquamarine', linestyle='-', lw=2.5)
                    plt.text((np.radians(180-sloneu)), 3.5+scolateu, 'EUR', color='aquamarine', fontsize=10, alpha=0.5,
                             path_effects=[mpl_patheffects.withStroke(linewidth=1, foreground='black')],\
                             horizontalalignment='center', verticalalignment='center', fontweight='bold')

                if abs(cml-slonga1) < 120 or abs(cml-slonga1) > 240:
                    plt.plot([(np.radians(180-slonga1)),(np.radians(180-slonga2))],[scolatga, scolatga], 'k-', lw=4)
                    plt.plot([(np.radians(180-slonga1)),(np.radians(180-slonga2))],[scolatga, scolatga], 'w-', lw=2.5)
                    plt.text((np.radians(180-slonga)), 3.5+scolatga, 'GAN', color='w', fontsize=10, alpha=0.5,
                             path_effects=[mpl_patheffects.withStroke(linewidth=1, foreground='black')],\
                             horizontalalignment='center', verticalalignment='center', fontweight='bold')

        elif fixed == 'lt':
            if hemis == "North" or hemis == "north" or hemis == "N" or hemis == "n":
                if abs(cml-nlonio1) < 120 or abs(cml-nlonio1) > 240:
                    plt.plot([(np.radians(180+cml-nlonio1)),(np.radians(180+cml-nlonio2))],[ncolatio, ncolatio], 'k-', lw=4)
                    plt.plot([(np.radians(180+cml-nlonio1)),(np.radians(180+cml-nlonio2))],[ncolatio, ncolatio], color='gold', linestyle='-', lw=2.5)
                    plt.text((np.radians(180+cml-nlonio)), 3.5+ncolatio, 'IO ', color='gold', fontsize=10,  fontweight='bold',alpha=0.5,
                             path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],\
                             horizontalalignment='center', verticalalignment='center')

                if abs(cml-nloneu1) < 120 or abs(cml-nloneu1) > 240:
                    plt.plot([2*np.pi-(np.radians(180+cml-nloneu1)),2*np.pi-(np.radians(180+cml-nloneu2))],[ncolateu, ncolateu], 'k-', lw=4)
                    plt.plot([2*np.pi-(np.radians(180+cml-nloneu1)),2*np.pi-(np.radians(180+cml-nloneu2))],[ncolateu, ncolateu], color='aquamarine', linestyle='-', lw=2.5)
                    plt.text((np.radians(180+cml-nloneu)), 3.5+ncolateu, 'EUR', color='aquamarine', fontsize=10, fontweight='bold',alpha=0.5,
                             path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],\
                             horizontalalignment='center', verticalalignment='center')

                if abs(cml-nlonga1) < 120 or abs(cml-nlonga1) > 240:
                    plt.plot([(np.radians(180+cml-nlonga1)),(np.radians(180+cml-nlonga2))],[ncolatga, ncolatga], 'k-', lw=4)
                    plt.plot([(np.radians(180+cml-nlonga1)),(np.radians(180+cml-nlonga2))],[ncolatga, ncolatga], 'w-', lw=2.5)
                    plt.text((np.radians(180+cml-nlonga)), 3.5+ncolatga, 'GAN', color='w', fontsize=10, fontweight='bold',alpha=0.5,
                             path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],\
                             horizontalalignment='center', verticalalignment='center')

            else: # South hemisphere
                if abs(cml-slonio1) < 120 or abs(cml-slonio1) > 240:
                    plt.plot([(np.radians(180+cml-slonio1)),(np.radians(180+cml-slonio2))],[scolatio, scolatio], 'k-', lw=4)
                    plt.plot([(np.radians(180+cml-slonio1)),(np.radians(180+cml-slonio2))],[scolatio, scolatio], color='gold', linestyle='-', lw=2.5)
                    plt.text((np.radians(180+cml-slonio)), 3.5+scolatio, 'IO ', color='gold', fontsize=10, alpha=0.5,
                             path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],\
                             horizontalalignment='center', verticalalignment='center', fontweight='bold')

                if abs(cml-sloneu1) < 120 or abs(cml-sloneu1) > 240:
                    plt.plot([(np.radians(180+cml-sloneu1)),(np.radians(180+cml-sloneu2))],[scolateu, scolateu], 'k-', lw=4)
                    plt.plot([(np.radians(180+cml-sloneu1)),(np.radians(180+cml-sloneu2))],[scolateu, scolateu], color='aquamarine', linestyle='-', lw=2.5)
                    plt.text((np.radians(180+cml-sloneu)), 3.5+scolateu, 'EUR', color='aquamarine', fontsize=10,alpha=0.5,
                             path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],\
                             horizontalalignment='center', verticalalignment='center', fontweight='bold')

                if abs(cml-slonga1) < 120 or abs(cml-slonga1) > 240:
                    plt.plot([(np.radians(180+cml-slonga1)),(np.radians(180+cml-slonga2))],[scolatga, scolatga], 'k-', lw=4)
                    plt.plot([(np.radians(180+cml-slonga1)),(np.radians(180+cml-slonga2))],[scolatga, scolatga], 'w-', lw=2.5)
                    plt.text((np.radians(180+cml-slonga)), 3.5+scolatga, 'GAN', color='w', fontsize=10,alpha=0.5,
                             path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],\
                             horizontalalignment='center', verticalalignment='center', fontweight='bold')    
            
            
    sloc = fpath(save_location)
    ensure_dir(sloc)
    if filename == 'auto': # if a filename is not specified, it will be generated.
        filename = finfo._basename
        extras = []
        extras = "-".join(
            [i for i in [
                (f'crop{crop}') if crop !=1 else '', 
                (f'r{rlim}') if rlim !=90 else '',
                f'fix{fixed}', 
                'S' if is_south else 'N',
                'full' if full else '',
                'regions' if regions else ''] if i != ''])
        filename += '-'+extras + '.jpg'
    fig.savefig(f'{sloc}/{filename}', **kwargs) # kwargs are passed to savefig, (dpi, quality, bbox, etc.)
    if 'return' in kwargs:
        return fig
    plt.show()
    #plt.close()


def make_gif(fits_dir,fps=5,remove_temp=True,savelocation='auto',filename='auto',**kwargs)->None:
    """
        Create a GIF from a directory of FITS files.

        This function takes a directory of FITS files, processes each file to create
        images, and compiles these images into a GIF. The GIF can be saved to a specified
        location with a specified filename.

        Args:
            fits_dir (str or list): Directory or list of directories containing FITS files.
            fps (int, optional): Frames per second for the GIF. Defaults to 5.
            remove_temp (bool, optional): Whether to remove temporary files after creating the GIF. Defaults to True.
            savelocation (str, optional): Directory to save the GIF. Defaults to 'auto', which saves to 'pictures/gifs/'
            filename (str, optional): Filename for the GIF. Defaults to 'auto', which generates a filename based on the FITS files.
            **kwargs: Additional keyword arguments to pass to the plotting function.

        Returns:
            None

        Keyword Args:
            all keyword arguments are passed to the plotting function, moind().
            
        Example:
            >>> make_gif('/path/to/fits/files', fps=10, savelocation='/path/to/save/', filename='my_gif')
    """
    if isinstance(fits_dir,str):
          fits_dir = [fits_dir,]
    fits_file_list = []
    for f in fits_dir:
        for g in glob.glob(f + '/*.fits', recursive=True):
            fits_file_list.append(g)
    fits_file_list.sort()    
    ln = len(fits_file_list)
    print(f'Found {ln} files in the directory.')
    infolist = [fileInfo(f) for f in fits_file_list]
    imagesgif = []
    with tqdm(total=ln) as pbar:        
        for i,file in enumerate(infolist):
            moind(fileinfo=file,save_location='temp/',filename=f'gifpart_{i}.jpg', **kwargs)
            tqdm.write(f'Image {i+1} of {ln} created: {file._basename}')
            pbar.update(1)
            imagesgif.append(imageio.imread(fpath('temp/')+f'gifpart_{i}.jpg'))
            #saving the GIF
    if savelocation == 'auto':
        savelocation = fpath('pictures/gifs/')
    ensure_dir(savelocation)
    if filename == 'auto':
        filename = f"{infolist[0].obs}_{infolist[0].year}_v{infolist[0].visit}_f{fps}_{ln}frames" 
    imageio.mimsave(savelocation+filename+".gif" , imagesgif, fps=fps)
    if remove_temp:
        for file in glob.glob(fpath('temp/')+'*'):
            os.remove(file)
        os.rmdir(fpath('temp/'))
