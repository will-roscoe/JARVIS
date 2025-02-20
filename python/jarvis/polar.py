__doc__="""
This module contains functions to generate polar projection plots of Jupiter's image data from FITS files. 
The main function, moind(), generates a polar projection plot of Jupiter's image data from a FITS file. 
The make_gif() function creates a GIF from a directory of FITS files.

adapted from dmoral's original code by the JAR:VIS team.
"""
#python standard libraries
import os
from dateutil.parser import parse
import datetime as dt
import glob


from typing import Tuple, Dict, Union, Callable
import fastgif

#third party libraries
from astropy.io import fits
import imageio
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter, MultipleLocator, ScalarFormatter
from matplotlib import patheffects as mpl_patheffects
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm 
# local modules
from .const import FITSINDEX
from .utils import fpath, fitsheader, fits_from_parent, get_datetime, clock_format, ensure_dir, prepare_fits, make_filename, fits_from_glob, basename
from .reading_mfp import moonfploc


   
def process_fits_file(fitsobj: fits.HDUList) -> fits.HDUList:
    """
    Processes a FITS file to apply transformations for polar plotting of a hemisphere. Uses header information to adjust the image data.
    Args:
        fitsobj (fits.HDUList): The FITS file object to be processed.
    Returns:
        fits.HDUList: The processed FITS file object with updated data.
    The function performs the following steps:
    1. Extracts header information such as CML, DECE, EXPT, and south.
    2. Parses the start time from the header.
    3. Calculates light travel time based on the distance from the origin.
    4. Computes the start, end, and mid-exposure times adjusted for Jupiter.
    5. Generates latitude and longitude bins for the image data.
    6. Creates a mask based on the latitude and longitude bins.
    7. Applies the mask to the image data to filter out invalid regions.
    8. Returns a new FITS file object with the processed image data.
    """
    cml, dece, exp_time, is_south = fitsheader(fitsobj, 'CML', 'DECE', 'EXPT', 'south')
    is_lon = fitsheader(fitsobj, 'fixed_lon')
    start_time = parse(fitsheader(fitsobj, 'UDATE'))
    try:
        dist_org = fitsheader(fitsobj, 'DIST_ORG')
        ltime = dist_org * 1.495979e+11 / 299792458
        lighttime = dt.timedelta(seconds=ltime)
    except ValueError:
        lighttime = dt.timedelta(seconds=2524.42)
    exposure = dt.timedelta(seconds=exp_time)
    start_time_jup = start_time - lighttime
    end_time_jup = start_time_jup + exposure # noqa: F841
    mid_ex_jup = start_time_jup + (exposure / 2.) # noqa: F841
    image_data = fitsobj[FITSINDEX].data
    latbins = np.radians(np.linspace(-90, 90, num=image_data.shape[0]))
    lonbins = np.radians(np.linspace(0, 360, num=image_data.shape[1]))
    mask = np.zeros((int(image_data.shape[0]), image_data.shape[1]))
    cmlr = np.radians(cml)
    dec = np.radians(dece)
    for i in range(0, mask.shape[0]):
        mask[i, :] = np.sin(latbins[i]) * np.sin(dec) + np.cos(latbins[i]) * np.cos(dec) * np.cos(lonbins - cmlr)
    mask = np.flip(mask, axis=1)
    cliplim = np.cos(np.radians(89))
    clipind = np.squeeze([mask >= cliplim])
    image_data[clipind == False] = np.nan # noqa: E712
    return fits_from_parent(fitsobj, new_data=image_data, FIXED='LT' if not is_lon else 'LON')
        

def plot_polar(fitsobj:fits.HDUList, ax:mpl.projections.polar.PolarAxes,**kwargs)-> mpl.projections.polar.PolarAxes:
    """
    Plots a polar projection of the given FITS object data.
    Parameters:
    fitsobj (fits.HDUList): The FITS file object containing the data to be plotted.
    ax (matplotlib.axes._subplots.PolarAxesSubplot): The matplotlib axis object to plot on.
    **kwargs: Additional keyword arguments for customization.
    Keyword Arguments:
    title (str): Title for the plot. 
    suptitle (str): Supertitle for the plot.
    ticks (list): List of tick values for the colorbar.
    cmap (str): Colormap to be used for the plot.
    norm (matplotlib.colors.Normalize): Normalization for the colormap.
    shrink (float): Shrink factor for the colorbar.
    pad (float): Padding for the colorbar.
    draw_cbar, draw_grid, draw_ticks, ax_params, ml, hemis (bool): Flags to draw the colorbar, grid, ticks, axis parameters, meridian lines, and hemisphere text. all default to True.
    Returns:
    matplotlib.axes._subplots.PolarAxesSubplot: The axis object with the plot.
    """
    image_data=fitsobj[FITSINDEX].data
    cml, is_south, fixed_lon,crop, full, rlim = fitsheader(fitsobj, 'CML', 'south', 'fixed_lon', 'CROP', 'FULL', 'RLIM')
    if kwargs.pop('nodec', False):
        kwargs.update({'draw_cbar': False, 'draw_grid': False, 'draw_ticks': False, 'ax_params': False, 'ml': False, 'hemis': False, 'title':False, 'cax':True})
    draw_ticks,ax_params = kwargs.pop('draw_ticks', True), kwargs.pop('ax_params', True)
    ax.set(**dict(theta_zero_location="N", facecolor='k',rlabel_position=0 if full else 0, 
            thetalim=[np.pi/2,3*np.pi/2] if not full else None, 
            rlim=[0,rlim], rgrids=np.arange(0,rlim,10,dtype='int')) ) #set the polar plot
    ax.yaxis.set(**dict(major_locator=MultipleLocator(base=10), 
            major_formatter=FuncFormatter(lambda x, _: '{:.0f}°'.format(x))) if draw_ticks else dict(visible=False)) # set radial ticks
    ax.yaxis.set_tick_params(**dict(labelcolor='white') if draw_ticks else {}) # set radial ticks
    ax.xaxis.set(**dict(major_locator=MultipleLocator(base=np.pi/2 if full else np.pi/4),
            major_formatter=FuncFormatter(lambda x, _: '{:.0f}°'.format(
                np.degrees((lambda x: x if is_south else 2*np.pi-x)(x))%360) if fixed_lon 
                else clock_format(x)) ,
            minor_locator = MultipleLocator(base=np.pi/18 if fixed_lon else np.pi/12)) if draw_ticks else dict(visible=False))
    ax.tick_params(axis='both',pad=2.)    # shift position of LT labels
    # Titles
    tkw,stkw =kwargs.pop('title', True),kwargs.pop('suptitle', True)
    if tkw:
        t_ = dict(suptitle=f'Visit {fitsobj[1].header["VISIT"]} (DOY: {fitsobj[1].header["DOY"]}/{fitsobj[1].header["YEAR"]}, {get_datetime(fitsobj)})', 
                title=f'{"Fixed LT. " if not fixed_lon else ""}Integration time={fitsobj[1].header["EXPT"]} s. CML: {np.round(cml, decimals=1)}°')
        t_['title'] = tkw if isinstance(tkw, str) else t_['title'] 
        t_['suptitle'] = stkw if isinstance(stkw, str) else t_['suptitle'] 
        parentfig = ax.get_figure()
        parentfig.suptitle(t_['suptitle'], y=0.99, fontsize=14) #one of the two titles for every plot
        ax.set_title(t_['title'],y=1.05 if full else 1.03 if not fixed_lon else 1.02, fontsize=12)
    if kwargs.pop('ml', True):
        if fixed_lon: # plotting cml, only for lon
            rot = 180 if is_south else 360
            ax.plot(np.roll([np.radians(rot-cml),np.radians(rot-cml)],180*4),[0, 180], 'r--', lw=1.2) #cml
            ax.text(np.radians(rot-cml), 3+rlim, 'CML', fontsize=11, color='r', ha='center', va='center', fontweight='bold') 
        if not fixed_lon and full: # meridian line (0°)  
            ax.text(np.radians(cml)+np.pi, 4+rlim, '0°', color='coral', fontsize=12,ha='center', va='bottom', fontweight='bold')
            ax.plot([np.radians(cml)+np.pi,np.radians(cml)+np.pi],[0, 180], color='coral', path_effects=[mpl_patheffects.withStroke(linewidth=1, foreground='black')], linestyle='-.', lw=1) #prime meridian (longitude 0)

    #Actual plot and colorbar (change the vmin and vmax to play with the limits
    #of the colorbars, recommended to enhance/saturate certain features)
    ticks = kwargs.pop('ticks', [10.,40.,100.,200.,400.,800.,1500.] if int(fitsobj[1].header['EXPT']) < 30 else [10.,40.,100.,200.,400.,1000.,3000.])
    kwd = dict(cmap='viridis', norm=mpl.colors.LogNorm(vmin=ticks[0], vmax=ticks[-1]), shrink=1 if full else 0.75, pad=0.06)
    cmap, norm, shrink, pad = [kwargs.pop(k, v) for k, v in kwd.items()]
    rho = np.linspace(0,180,num=int(image_data.shape[0]))
    theta = np.linspace(0,2*np.pi, num=image_data.shape[1])
    if fixed_lon: 
        image_centred = image_data
    else: 
        image_centred = np.roll(image_data,int(cml-180.)*4,axis=1) #shifting the image to have CML pointing southwards in the image
    corte = np.flip(image_centred,0)[:(int((image_data.shape[0])/crop)),:]
    if is_south:
        rho = rho[::-1]
        corte = np.roll(corte,180*4,axis=1)
        
    cmesh = ax.pcolormesh(theta,rho[:(int((image_data.shape[0])/crop))],corte,norm=norm, cmap=cmap)#!5 <- Color of the plot

 
    if kwargs.pop('draw_cbar',True):
        cbar = plt.colorbar(ticks=ticks, shrink=shrink, pad=pad, ax=ax, mappable=cmesh)
        cbar.ax.set_yticklabels([str(int(i)) for i in ticks])
        cbar.ax.set_ylabel('Intensity [kR]', rotation=270.)
    #Grids (major and minor)
    if not kwargs.pop('draw_grid', True):
        ax.grid(False, which='both')
    else:
        ax.grid(True, which='major', color='w', alpha=0.7, linestyle='-')
        ax.minorticks_on()
        ax.grid(True, which='minor', color='w', alpha=0.5, linestyle='-')
        #stronger meridional lines for the 0, 90, 180, 270 degrees:
        for i in range(0,4):
            ax.plot([np.radians(i*90),np.radians(i*90)],[0, 180], 'w', lw=0.9)
    #deprecated variable (maybe useful for the South/LT fixed definition?)
    #shift = 0# cml-180. #! UNUSED
    #print which hemisphere are we in:
    if kwargs.pop('hemis', True):
        ax.text(45 if full else 135 if any([not is_south, not fixed_lon]) else -135 , 1.3*rlim, str(fitsheader(fitsobj, 'HEMISPH')).capitalize(), fontsize=21, color='k', 
             ha='center', va='center', fontweight='bold')   
    if kwargs.pop('cax', False): # get tightest layout possible
        plt.tight_layout()
    return ax




def plot_moonfp(fitsobj:fits.HDUList, ax:mpl.projections.polar.PolarAxes)->mpl.projections.polar.PolarAxes:
    """
    Plots the footprints of moons on a given axis based on the provided FITS object and the moonfploc function.
    Parameters:
    fitsobj (fits.HDUList): The FITS object containing the data.
    ax (mpl.projections.polar.PolarAxes): The matplotlib axis on which to plot the moon footprints.
    Returns:
    None
    The function extracts relevant data from the FITS header, calculates the positions of the moons,
    and plots their footprints on the provided axis. It handles both northern and southern hemispheres
    and adjusts the plotting based on whether the longitude is fixed or not.
    """
    cml, is_south, fixed_lon = fitsheader(fitsobj, 'CML', 'south', 'fixed_lon')
    lons = [fitsheader(fitsobj, f'IOLON{k}', f'EULON{k}', f'GALON{k}') for k in ('', 1, 2)]
    fp = np.array([moonfploc(*lon) for lon in lons])
    moon_list = [[[*fp[:,4*i], fp[0,4*i+1]], [*fp[:,4*i+2], fp[0,4*i+3]]] for i in range(3)]
    #LT:
    moonrange=[] #empty list, see if the coordinates of the moons are in range
    for x in moon_list:
        if not is_south: 
            moonrange.append(x[0]) #appends north hemisphere data to moon range, only use north hemis: list of northern hemisphere data of moons to the list
        else:
            moonrange.append(x[1]) #only using south hemisphere
    #moonrange has all north or all south now
    if not fixed_lon:  
        for i in range(3): #for IO first index, EUR second index, GAN third index
            x=np.radians(180+cml-moonrange[i][1]) #calculate the coordinates values, the second file
            y=np.radians(180+cml-moonrange[i][2]) #third file
            w=np.radians(180+cml-moonrange[i][0]) #first file
            v=moonrange[i][3] #last file
            if abs(cml-moonrange[i][1]) < 120 or abs(cml-moonrange[i][1]) > 240: #if the coordinates are in range of HST viewing, of each moon
                ax.plot([x,y],[v,v],'k-', lw=4)
                color,key = (('gold','IO'), ('aquamarine','EUR'), ('w','GAN'))[i]
                ax.plot([x,y],[v,v],color=color, linestyle='-', lw=2.5)
                ax.text(w, 3.5+v, key, color=color, fontsize=10,alpha=0.5,
                        path_effects=[mpl_patheffects.withStroke(linewidth=1, foreground='black')],\
                        horizontalalignment='center', verticalalignment='center', fontweight='bold')
    if fixed_lon:
        if not is_south: #process coordinates for north hemis of lon
            for i in range(3): #for IO first index, EUR second index, GAN third index
                x=2*np.pi-(np.radians(moonrange[i][1])) #calculate the coordinates values, the second file
                y=2*np.pi-(np.radians(moonrange[i][2])) #third file
                w=2*np.pi-(np.radians(moonrange[i][0])) #first file
                v=moonrange[i][3] #last file
                if abs(cml-moonrange[i][1]) < 120 or abs(cml-moonrange[i][1]) > 240: #if the coordinates are in range of HST viewing, of each moon
                    ax.plot([x,y],[v,v],'k-', lw=4)
                    color,key = (('gold','IO'), ('aquamarine','EUR'), ('w','GAN'))[i]
                    ax.plot([x,y],[v,v],color=color, linestyle='-', lw=2.5)
                    ax.text(w, 3.5+v, key, color=color, fontsize=10,alpha=0.5,
                            path_effects=[mpl_patheffects.withStroke(linewidth=1, foreground='black')],\
                            horizontalalignment='center', verticalalignment='center', fontweight='bold')     
        else:  #process coordinates for south hemis of lon
            for i in range(3): #for IO first index, EUR second index, GAN third index
                x=np.radians(180-moonrange[i][1]) #calculate the coordinates values, the second file
                y=np.radians(180-moonrange[i][2]) #third file
                w=np.radians(180-moonrange[i][0]) #first file
                v=moonrange[i][3] #last file      
                if abs(cml-moonrange[i][1]) < 120 or abs(cml-moonrange[i][1]) > 240: #if the coordinates are in range of HST viewing, of each moon
                    ax.plot([x,y],[v,v],'k-', lw=4)
                    color,key = (('gold','IO'), ('aquamarine','EUR'), ('w','GAN'))[i]
                    ax.plot([x,y],[v,v],color=color, linestyle='-', lw=2.5)
                    ax.text(w, 3.5+v, key, color=color, fontsize=10,alpha=0.5,
                            path_effects=[mpl_patheffects.withStroke(linewidth=1, foreground='black')],\
                            horizontalalignment='center', verticalalignment='center', fontweight='bold')            
        
        
        #ORGINALLY LON N, range is 250 idk why :abs(cml-nlonga1) < 120 or abs(cml-nlonga1) > 250:
        #LT, S/N: (np.radians(180+cml-X1)),(np.radians(180+cml-X2)), TEXT np.radians(180+cml-X0)
        #LT N EUR: (2*np.pi-(np.radians(180+cml-X1)) (weird)
        #LON N: 2*np.pi-(np.radians(X1)),2*np.pi-(np.radians(X2) TEXT 2*np.pi-(np.radians(X0))
        #LON S: (np.radians(180-X1)),(np.radians(180-X2)), TEXT np.radians(180-X0)
    return ax



def plot_regions(fitsobj: fits.HDUList, ax:mpl.projections.polar.PolarAxes)->mpl.projections.polar.PolarAxes:
    """
    Plots various regions on a polar plot using the provided FITS object and matplotlib axis.
    Parameters:
    fitsobj : object
        The FITS object containing the data and header information.
    ax : mpl.projections.polar.PolarAxes
        The matplotlib axis on which to plot the regions.
    The function plots the following regions:
    - Dusk boundary in red.
    - Dawn boundary in black or blue depending on the 'fixed_lon' header value.
    - Noon boundary in yellow.
    - Polar boundary in white dashed lines.
    """
    lon_fixed = fitsheader(fitsobj, 'fixed_lon')
    updusk = np.linspace(np.radians(205), np.radians(170), 200)
    dawn = { k: np.linspace(v[0], v[1], 200) for k, v in (("lon", (np.radians(180), np.radians(130))), ("uplat", (33, 15)), ("downlat", (39, 23))) }
    noon_a = {k: np.linspace(v[0], v[1], 100)for k, v in (("lon", (np.radians(205), np.radians(190))), ("downlat", (28, 32)))}
    noon_b = {k: np.linspace(v[0], v[1], 100)for k, v in (("lon", (np.radians(190), np.radians(170))), ("downlat", (32, 27)))}
            # lon_noon_a = noon_a['lon']
    regions = [[[[np.radians(205), np.radians(170)],[20,10]],[[np.radians(170), np.radians(205)],[10,20]],[updusk, 200 * [10]], [updusk, 200 * [20]]], # dusk boundary
             [[[np.radians(130), np.radians(130)],[23,15]],[[np.radians(180), np.radians(180)],[33,39]],[dawn["lon"], dawn["uplat"]],[dawn["lon"], dawn["downlat"]]], # dawn boundary
             [[[np.radians(205), np.radians(205)],[22,28]],[[np.radians(170), np.radians(170)],[27,22]],[noon_a["lon"], noon_a["downlat"]],[noon_b["lon"], noon_b["downlat"]]],  # noon boundary
             [[[np.radians(205), np.radians(205)],[10,28]],[[np.radians(170), np.radians(170)],[27,10]],[noon_a["lon"], noon_a["downlat"]],[updusk, 200 * [10]]]] # polar boundary
    for i,region in enumerate(regions):
        c = ["r-", "k-" if lon_fixed else "b-", "y-", "w--"][i]
        lw = [1.5, 1, 1.5, 1][i]
        for line in region:
            ax.plot(*line, c, lw=lw)
    return ax




def moind(fitsobj:fits.HDUList, crop:float = 1, rlim:float = 40, fixed:str= 'lon', hemis:str='North', full:bool=True, regions:bool=False,moonfp:bool=False,**kwargs)->mpl.figure.Figure:  
    """
    Process and plot a FITS file in polar coordinates 
    Parameters:
    fitsobj (fits.HDUList): The FITS file object to be processed.
    crop (float, optional): The crop factor for the FITS file. Default is 1.
    rlim (float, optional): The radial limit for the plot. Default is 40.
    fixed (str, optional): The fixed parameter for the FITS file. Default is 'lon'.
    hemis (str, optional): The hemisphere to be plotted ('North' or 'South'). Default is 'North'.
    full (bool, optional): Whether to plot the full FITS file. Default is True.
    regions (bool, optional): Whether to plot regions on the FITS file. Default is False.
    moonfp (bool, optional): Whether to plot the moon footprint. Default is False.
    **kwargs: Additional keyword arguments for plotting.
    Returns:
    Union[None, mpl.figure.Figure]: The matplotlib figure and axis objects if successful, otherwise None.
    """
    fits_obj = process_fits_file(prepare_fits(fitsobj, crop=crop, rlim=rlim, fixed=fixed, full=full, regions=regions, moonfp=moonfp))                                     
    fig =plt.figure(figsize=(7,6))
    ax = plt.subplot(projection='polar')
    plot_polar(fits_obj, ax, **kwargs)
    if regions and not fitsheader(fits_obj, 'south'):
        plot_regions(fits_obj,ax)
    if moonfp:
        plot_moonfp(fits_obj,ax,) 
    return fig, ax, fits_obj



def make_gif(fits_dir,fps=5,remove_temp=False,savelocation='auto',filename='auto',**kwargs)->None:
    """
    Create a GIF from a directory of FITS files.
    Parameters:
    fits_dir (str): The directory containing the FITS files.
    fps (int, optional): The frames per second for the GIF. Default is 5.
    remove_temp (bool, optional): Whether to remove temporary files. Default is False.
    savelocation (str, optional): The directory to save the GIF. Default is 'auto'.
    filename (str, optional): The name of the GIF file. Default is 'auto'.
    **kwargs: Additional keyword arguments for the moind function.
    Returns:
    None
    The function reads all FITS files in the specified directory, generates polar plots using the moind function,
    and creates a GIF from the images. The GIF is saved in the specified location with the given filename.
    """
    fitslist,fnames = fits_from_glob(fits_dir, names=True)
    imagesgif=[]
    with tqdm(total=len(fitslist)) as pb:        
        for i,file in enumerate(fitslist):
            pb.set_postfix(file=fnames[i])
            fig,ax,f = moind(file, **kwargs)
            fig.savefig(fpath('temp/')+f'gifpart_{i}.jpg', dpi=300)
            plt.close(fig)
            #tqdm.write(f'Image {i+1} of {len(fitslist)} created: {"IMPLEMENT"}')
            pb.update(1)
            
            imagesgif.append(imageio.imread(fpath('temp/')+f'gifpart_{i}.jpg'))
            #saving the GIF
    if savelocation == 'auto':
        savelocation = fpath('pictures/gifs/')
    ensure_dir(savelocation)
    if filename == 'auto':
        filename = make_filename(f)+f'{len(fitslist)}fr_{fps}fps' +'.gif'
    imageio.mimsave(savelocation+filename, imagesgif, fps=fps)
    if remove_temp:
        for file in glob.glob(fpath('temp/')+'*'):
            os.remove(file)
        os.rmdir(fpath('temp/'))



def makefast_gif(fitsobjs,initfunc=None,fps=5,showprogress=True,**kwargs)->None:
    """
    Create a GIF from a list of FITS files using the fastgif module.
    Parameters:
    fitsobjs (list): A list of FITS file objects.
    initfunc (function, optional): The initialization function for the GIF. this function should take an index as an argument and return a figure object. Default is None, which uses the moind function.
    fps (int, optional): The frames per second for the GIF. Default is 5.
    showprogress (bool, optional): Whether to show the progress bar. Default is True.
    **kwargs: Additional keyword arguments for the moind function. If 'saveto' is provided, the GIF is saved to the specified location.
    Returns:
    None
    The function generates a GIF from the list of FITS files using the fastgif module. It uses the initialization function to create the figure objects for each frame.
    The GIF is saved to the specified location if 'saveto' is provided in the keyword arguments.
    """
    if initfunc is None:
        def initfunc(idx):
            fits_obj = process_fits_file(prepare_fits(fitsobjs[idx], **kwargs.pop('fits',{})))                                     
            fig =plt.figure(figsize=(7,6))
            ax = plt.subplot(projection='polar')
            plot_polar(fits_obj, ax, **kwargs)
            if all([not fitsheader(fits_obj, 'south'), fitsheader(fits_obj, 'REGIONS')]):
                plot_regions(fits_obj,ax)
            if fitsheader(fits_obj, 'MOONFP'):
                plot_moonfp(fits_obj,ax,) 
            return fig
    if 'saveto' in kwargs:
        savelocation = kwargs.pop('saveto')
    else:
        savelocation = fpath('figures/gifs/')+make_filename(prepare_fits(fitsobjs[0], **kwargs))+'.gif'
    fastgif.make_gif(initfunc,num_calls=len(fitsobjs),filename=savelocation,show_progress=showprogress,writer_kwargs={'duration':1/fps})
    