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
from .const import fpath, fileInfo

    
#this is the funcion used for plotting the images
def moind(image_data:np.ndarray, header:np.ndarray, file_location:str,save_location:str,filename='auto', crop = 1, rlim = 30, fixed = 'lon', hemis='North', full=True, regions=False,**kwargs):
    """
            Generate a polar projection plot of an image with various customization options.
            Parameters:
            -----------
            image_data : numpy.ndarray
                2D array representing the image data, typically astropy.io.fits data
            header : dict
                Dictionary containing header information such as CML, DECE, EXPT, etc.
            file_location : str
                The path of the fits file 
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
    lon = {'io':(header['IOLON'],header['IOLON1'],header['IOLON2']), # Unused
           'eu':(header['EULON'],header['EULON1'],header['EULON2']),
           'ga':(header['GALON'],header['GALON1'],header['GALON2'])}

    
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
    fig = plt.figure(figsize=(7,6))
        
    ax = plt.subplot(projection='polar')
    radials = np.linspace(0,rlim,6,dtype='int')
    radials = np.arange(0,rlim,10,dtype='int') #TODO Double definition of radials

    #shifting the image to have CML pointing southwards in the image
    if fixed == 'lon':
        image_centred = image_data
        im_flip = np.flip(image_centred,0) #reverse the image along the longitudinal (x, theta) axis
    #cropping the image
        corte = im_flip[:(int((image_data.shape[0])/crop)),:]
        rot = 360
        if str(hemis).lower()[0] == "s":
            corte = np.roll(corte,180*4,axis=1)
            rot = 180
        ax.plot(np.roll([np.radians(rot-cml),np.radians(rot-cml)],180*4),[0, 180], 'r--', lw=1.2) #cml
        ax.text(np.radians(rot-cml), 3+rlim, 'CML', fontsize=11, color='r',
                     horizontalalignment='center', verticalalignment='center', fontweight='bold') 
           
        if full == True:
            ax.set_xticks(np.linspace(0,2*np.pi,37)) # set radial ticks #TODO currently minor grid is every 30deg, not 10.
            poshem = np.radians(45) #position of the "N/S" marker
            ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.pi/2))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(base=np.pi/20))
            # depending if north or south we flip the clock angle.
            if str(hemis).lower()[0] == "s":
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}°'.format(np.degrees(x)%360)))
                # should be 180°, 90°, 0°, 270°
            else:
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}°'.format(np.degrees(2*np.pi-x)%360)))
                # should be 0°, 270°, 180°, 90° (counterclockwise)
            ytickl = [str(i)+'°' for i in radials]
            ax.set_yticklabels(ytickl,color='w',fontsize=10)#, weight='bold') # turn off auto lat labels
            ax.set_rticks(np.arange(radials[1],rlim,10,dtype='int'))           
            ax.set_rlabel_position(0)   #position of the radial labels  
            shrink = 1. #size of the colorbar
            possub = 1.05 #position in the y axis of the subtitle
                   
        else:
            ax.set_xticks(np.linspace(np.pi/2,3*np.pi/2,5))  # but set grid spacing
            if str(hemis).lower()[0] == "s":
                ax.set_xticklabels(['90°','45°','0°','315°','270°'], fontweight='bold')
                poshem = np.radians(-135) #position of the "S" marker
    
            else:  # if North
                ax.set_xticklabels(['270°','225°','180°','135°','90°'], fontweight='bold')
                poshem = np.radians(135) #position of the "N" marker
            
            shrink = 0.75 #size of the colorbar
            possub = 1.03 #position in the y axis of the subtitle
            ax.set_thetalim([np.pi/2,3*np.pi/2])
            ax.set_yticklabels(['',str(radials[1])+'°',str(radials[2])+'°',str(radials[3])+'°',
                                str(radials[4])+'°',str(radials[5])+'°'],color='w',fontsize=10)#, weight='bold') # turn off auto lat labels
            ax.set_rticks(np.linspace(radials[1],rlim,5,dtype='int'))                       # but set grid spacing
        ax.set_theta_zero_location("N")                   # set angle 0.0 to top of plot

    elif fixed == 'lt':
        image_centred = np.roll(image_data,int(cml-180.)*4,axis=1)
        im_flip = np.flip(image_centred,0)
        corte = im_flip[:(int((image_data.shape[0])/crop)),:] #if crop=1, nothing changes
            
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
            ax.set_theta_zero_location("N")                   # set angle 0.0 to bottom of plot
            poshem = np.radians(45) #position of the "S" marker
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
    ax.set_rgrids(radials)#, color='white')

    #Naming variables
    finfo = fileInfo(file_location)
    #one of the two titles for every plot
    plt.suptitle(f'Visit {finfo.visit} (DOY: {finfo.day}/{finfo.year}, {finfo.datetime})', y=0.99, fontsize=14)
    cmlround = np.round(cml, decimals=1)
    

    #the other title + the 0° longitudinal meridian for the LT fixed case
    if fixed == 'lon':
        plt.title(f'Integration time={finfo.exp} seconds. CML: {cmlround}°', y=possub, fontsize=12)
    elif fixed == 'lt':
        plt.title(f'Fixed LT. Integration time={finfo.exp} s. CML: {cmlround}°', fontsize=12)
        plt.text(np.radians(cml)+np.pi, 4+rlim, '0°', color='coral', fontsize=12,
                 horizontalalignment='center', verticalalignment='bottom', fontweight='bold')
        plt.plot([np.radians(cml)+np.pi,np.radians(cml)+np.pi],[0, 180], color='coral',
                  path_effects=[patheffects.withStroke(linewidth=1, foreground='black')],
                  linestyle='-.', lw=1) #prime meridian (longitude 0)


    #Actual plot and colorbar (change the vmin and vmax to play with the limits
    #of the colorbars, recommended to enhance/saturate certain features)
    if int(finfo.exp) < 30:
        ticks = [10.,40.,100.,200.,400.,800.,1500.]
    else:
        ticks = [10.,40.,100.,200.,400.,1000.,3000.]
    plt.pcolormesh(theta,rho[:(int((image_data.shape[0])/crop))],corte,norm=LogNorm(vmin=ticks[0], vmax=ticks[-1]), cmap='inferno')#!5 <- Color of the plot
    cbar = plt.colorbar(ticks=ticks, shrink=shrink, pad=0.06)
    cbar.ax.set_yticklabels([str(int(i)) for i in ticks])     

#####################################################
    #Title for the colorbar    
    cbar.ax.set_ylabel('Intensity [kR]', rotation=270.)
    #Grids (major and minor)
    plt.grid(True, which='major', color='w', alpha=0.6, linestyle='-')
    plt.minorticks_on()
    plt.grid(True, which='minor', color='w', alpha=0.2, linestyle='--')
    #stronger meridional lines for the 0, 90, 180, 270 degrees:
    for i in range(0,4):
        plt.plot([np.radians(i*90),np.radians(i*90)],[0, 180], 'w', lw=1.5)
    #deprecated variable (maybe useful for the South/LT fixed definition?)
    shift = 0# cml-180.
    #print which hemisphere are we in:
    plt.text(poshem, 1.3*rlim, str(hemis).capitalize(), fontsize=21, color='k', 
             horizontalalignment='center', verticalalignment='center', fontweight='bold') 
  

    #drawing the regions (if marked; only available for the North so far)
    if regions == True:
        if str(hemis).lower()[0] == "n":
            #regions delimitation (lat and lon)
            updusk = np.linspace(np.radians(205),np.radians(170),200) 
            dawn = {k:np.linspace(v[0],v[2],200) for k,v in (('lon',(np.radians(180),np.radians(130))), ('uplat',(33, 15)), ('downlat',(39, 23)))}
            noon = {k:np.linspace(v[0],v[1],100) for k,v in (('lon_a',(np.radians(205),np.radians(190))), ('lon_b',(np.radians(190),np.radians(170))), ('downlat_a',(28, 32)), ('downlat_b',(32, 27)))}
            # lon_noon_a = noon['lon_a']

            #dusk boundary
            plt.plot([np.radians(205), np.radians(205)], [20, 10], 'r-', lw=1.5) 
            plt.plot([np.radians(170), np.radians(170)], [10, 20], 'r-', lw=1.5)  
            plt.plot(updusk, 200*[10], 'r-', lw=1.5)
            plt.plot(updusk, 200*[20], 'r-', lw=1.5) 
            #dawn boundary
            c = 'k-' if fixed == 'lon' else 'b-'
            plt.plot([np.radians(130), np.radians(130)], [23, 15], c, lw=1) 
            plt.plot([np.radians(180), np.radians(180)], [33, 39], c, lw=1)
            plt.plot(dawn['lon'], dawn['uplat'], c, lw=1)
            plt.plot(dawn['lon'], dawn['downlat'], c, lw=1)
            #noon boundary
            plt.plot([np.radians(205), np.radians(205)], [22, 28], 'y-', lw=1.5)   
            plt.plot([np.radians(170), np.radians(170)], [27, 22], 'y-', lw=1.5)    
            plt.plot(noon['lon_a'], noon['downlat_a'], 'y-', lw=1.5) 
            plt.plot(noon['lon_b'], noon['downlat_b'], 'y-', lw=1.5) 
            plt.plot(updusk, 200*[22], 'y-', lw=1.5) 
            #polar boundary
            plt.plot([np.radians(205), np.radians(205)], [10, 28], 'w--', lw=1)
            plt.plot([np.radians(170), np.radians(170)], [27, 10], 'w--', lw=1)
            plt.plot(noon['lon_a'], noon['downlat_a'], 'w--', lw=1) 
            plt.plot(updusk, 200*[10], 'w--', lw=1) 

        elif hemis == "South" or hemis == "south" or hemis == "S" or hemis == "s":
            pass
    def ensure_dir(file_path):
        '''this function checks if the file path exists, if not it will create one'''
        if not os.path.exists(file_path):
                os.makedirs(file_path)
  
    def savedataimg(file_path,fixed='detailtype', dpi=300):
        '''this function creates the image type of lt, or lon, into the desired file path, and what what dpi resolution'''
        ensure_dir(file_path)
        ensure_dir(f'{file_path}/{fixed}')
        mkpath = f'{file_path}/{fixed}/mo_{str(finfo._basename)}_fix{fixed}.jpg'
        plt.savefig(fpath(mkpath), dpi=dpi)

    
    savedataimg('HST', 'lon',dpi=300)
    savedataimg('pictures','lt',dpi=300)
    plt.close()

#and this chunk is to call the function:
def make_gif(fits_dir, **kwargs,):
    fits_file_list = glob.glob(fits_dir)
    #ab = glob.glob(fpath(f'data/HST/Jupiter/{year}/extract/{extra}'+arch+'/nopolar'+time+ti))
    fits_file_list.sort()    
    print(fits_file_list)
    #define the image resolution for when there are many images so the 
    #final GIF is not too large
    dpi = kwargs.get('dpi') if 'dpi' in kwargs else 150 
    #and now we loop to apply the plotting function (moind) to every image in the visit
    fixed = kwargs.get('fixed') if 'fixed' in kwargs else 'lon'
    full = kwargs.get('full') if 'full' in kwargs else True
    radius = kwargs.get('radius') if 'radius' in kwargs else 30
    fps = kwargs.get('fps') if 'fps' in kwargs else 5
    for i,file in tqdm(enumerate(fits_file_list)):
        with fits.open(file) as hdulist:
            header = hdulist[1].header
            image = hdulist[1].data
            moind(image, header, filename,  dpi = dpi, crop=1, rlim = radius, fixed = fixed,
                    hemis = header['HEMISPH'], full=full, photo=0, )#mf=mf, indf=indf, polarf=polarf, secondf=secondf
        print(f'Image {i} of {str(file)[-51:-5]} created.')
        
        #and now we start creating the GIF
        imagesgif = []        
        print(f'On to the GIF ')
        
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
def _multigif(lista, year, prefix, extra, time, radius, moonfp, full, fixed, mf, indf, polarf, secondf):
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
                      hemis = hemis, full=full, photo=n, )#mf=mf, indf=indf, polarf=polarf, secondf=secondf
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
#you have to input the year of the visits you are plotting (so cannot mix visits
#from different years in the same "run" of the code, for filepathing reasons)
def input_run():
    year = input("Year of the visit:  \n")
    if year == '2016':
        pre = input('Campaign from Jonny or Denis? (1/2)  ')
        if pre == '1' or pre == 'jonny' or pre == 'j' or pre == 'J' or pre == 'Jonny':
            prefix,ext = 'ocx8','nichols/'
        else:
            prefix,ext = 'od8k','grodent/' 
    elif year == '2019':
        prefix,ext = 'odxc',''
    elif year == '2021':
        prefix,ext = 'oef4',''
        
    elif year == '2017' or  year == '2018':
        prefix,ext = 'od8k',''
  
    time = str(input('Exposure time (in seconds: 10, 30, 100...): \n')) #usually 100
    radius = int(input('Max. radius (in degrees of colatitude): \n'))   #usually 40
    moonfp = not bool(input('Moon footprints printed? (Default: Enter for YES)\n')) #usually yes
    full = not bool(input('Show the whole hemisphere? (Default: Enter for YES)\n')) #usually yes
    fixed = str(input('Fix longitude (lon) or Local Time (lt):\n')) 

    #the only part you have to add manually is the particular set of visit numbers you want
    #to plot. If you want only one that is perfectly fine but it must be IN a list
    lista = ['01'] #for example, or lista = ['0v']

    multigif(lista, year, prefix, ext, time, radius, moonfp, full, fixed, mf=0, indf=0, polarf=True, secondf=0) # this is what I need to call
