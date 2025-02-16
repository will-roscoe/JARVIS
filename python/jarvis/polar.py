__doc__="""
This module contains functions to generate polar projection plots of Jupiter's image data from FITS files. 
The main function, moind(), generates a polar projection plot of Jupiter's image data from a FITS file. 
The make_gif() function creates a GIF from a directory of FITS files.

adapted from dmoral's original code by the JAR:VIS team.
"""
#import all the necessary modules
#python standard libraries
import os
import glob
from typing import Tuple, Dict, Union, Callable

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
from .const import fpath, FitsFile, FitsFileSeries
from .reading_mfp import moonfploc


def ensure_dir(file_path):
    '''this function checks if the file path exists, if not it will create one'''
    if not os.path.exists(file_path):
            os.makedirs(file_path)
def clock_format(x_rads, pos):
    # x_rads => 0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4, ...
    # returns=> 00, 03, 06, 09, 12, 15, 18, 21,..., 00,03,06,09,12,15,18,21,
    cnum= int(np.degrees(x_rads)/15)
    return f'{cnum:02d}' if cnum%24 != 0 else '00'


def load_fits_data(file_location: str = None, fileinfo: FitsFile = None) -> Tuple[np.ndarray, Dict]:
    if fileinfo is None:
        f_abs = fpath(file_location)
    else:
        f_abs = fileinfo._path
    with fits.open(f_abs) as hdulist:
        image_data = hdulist[1].data
        header = hdulist[1].header
    return image_data, header

def transform_to_polar(image_data: np.ndarray, header: Dict, crop: float = 1, rlim: float = 40, fixed: str = 'lon', hemis: str = 'North') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cml = header['CML']
    dece = header['DECE']
    latbins = np.radians(np.linspace(-90, 90, num=image_data.shape[0]))
    lonbins = np.radians(np.linspace(0, 360, num=image_data.shape[1]))
    rho = np.linspace(0, 180, num=int(image_data.shape[0]))
    theta = np.linspace(0, 2 * np.pi, num=image_data.shape[1])
    if hemis.lower()[0] == 's':
        rho = rho[::-1]
    mask = np.zeros((int(image_data.shape[0]), image_data.shape[1]))
    cmlr = np.radians(cml)
    dec = np.radians(dece)
    for i in range(0, mask.shape[0]):
        mask[i, :] = np.sin(latbins[i]) * np.sin(dec) + np.cos(latbins[i]) * np.cos(dec) * np.cos(lonbins - cmlr)
    mask = np.flip(mask, axis=1)
    cliplim = np.cos(np.radians(89))
    clipind = np.squeeze([mask >= cliplim])
    image_data[clipind == False] = np.nan #noqa: E712
    return image_data, rho, theta, mask, cml, dece

def create_figure(image_data: np.ndarray, rho: np.ndarray, theta: np.ndarray, cml: float, rlim: float, fixed: str, hemis: str, full: bool) -> Tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(7, 6))
    ax = plt.subplot(projection='polar')
    radials = np.arange(0, rlim, 10, dtype='int')
    if fixed == 'lon':
        image_centred = image_data
    else:
        image_centred = np.roll(image_data, int(cml - 180.) * 4, axis=1)
    im_flip = np.flip(image_centred, 0)
    corte = im_flip[:(int((image_data.shape[0]) / crop)), :]
    if fixed == 'lon':
        if hemis.lower()[0] == 's':
            corte = np.roll(corte, 180 * 4, axis=1)
            rot = 180
        else:
            rot = 360
        ax.plot(np.roll([np.radians(rot - cml), np.radians(rot - cml)], 180 * 4), [0, 180], 'r--', lw=1.2)
        ax.text(np.radians(rot - cml), 3 + rlim, 'CML', fontsize=11, color='r', horizontalalignment='center', verticalalignment='center', fontweight='bold')
    shrink = 1 if full else 0.75
    possub = 1.05 if full else 1.03 if fixed == 'lt' else 1.02
    poshem = 45 if full else 135 if any([hemis.lower()[0] == 'n', fixed == 'lt']) else -135
    ax.set_theta_zero_location("N")
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=10))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: '{:.0f}°'.format(x)))
    ax.yaxis.set_tick_params(labelcolor='white')
    if fixed == 'lon':
        if full:
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=np.pi / 2))
            if hemis.lower()[0] == 's':
                shift_t = lambda x: x # noqa: E731
            else:
                shift_t = lambda x: 2 * np.pi - x # noqa: E731
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: '{:.0f}°'.format(np.degrees(shift_t(x)) % 360)))
        else:
            ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=np.pi / 4))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=2 * np.pi / 36))
    else:
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=2 * np.pi / 8))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=2 * np.pi / 24))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(clock_format))
    if full:
        ax.set_rlabel_position(0)
    else:
        ax.set_thetalim([np.pi / 2, 3 * np.pi / 2])
    ax.set_facecolor('k')
    ax.set_rlim([0, rlim])
    ax.tick_params(axis='both', pad=2.)
    ax.set_rgrids(radials)
    return fig, ax

def plot_moon_footprints(ax: plt.Axes, lon: Dict[str, Tuple[float, float, float]], cml: float, hemis: str, fixed: str):
    '''
    moonfploc output:
                        00  01  02  03  04  05  06  07  08  09  10  11
        ________________________________________________________________               
        Longitudes  :  ■■■■    ■■■■    ■■■■    ■■■■    ■■■■    ■■■■     
        Latitudes   :      ■■■■    ■■■■    ■■■■    ■■■■    ■■■■    ■■■■
        ________________________________________________________________
        North       :  ■■■■■■■■        ■■■■■■■■        ■■■■■■■■         
        South       :          ■■■■■■■■        ■■■■■■■■        ■■■■■■■■        
        ________________________________________________________________
        IO          :  ■■■■■■■■■■■■■■■■
        EUR         :                  ■■■■■■■■■■■■■■■■
        GAN         :                                  ■■■■■■■■■■■■■■■■
    
    '''
    m = moonfploc(lon['io'][0], lon['eu'][0], lon['ga'][0])
    m1 = moonfploc(lon['io'][1], lon['eu'][1], lon['ga'][1])
    m2= moonfploc(lon['io'][2], lon['eu'][2], lon['ga'][2])
    io_dat = [[m[0], m1[0], m2[0], m[1]], [m[2], m1[2], m2[2], m[3]]]
    eur_dat = [[m[4], m1[4], m2[4], m[5]], [m[6], m1[6], m2[6], m[7]]]
    gan_dat = [[m[8], m1[8], m2[8], m[9]], [m[10], m1[10], m2[10], m[11]]]
    moon_list = [io_dat, eur_dat, gan_dat]
    moonrange = []
    for x in moon_list:
        if hemis.lower()[0] == 'n':
            moonrange.append(x[0])
        else:
            moonrange.append(x[1])
    if fixed == 'lt':
        for i in range(3):
            x = np.radians(180 + cml - moonrange[i][1])
            y = np.radians(180 + cml - moonrange[i][2])
            w = np.radians(180 + cml - moonrange[i][0])
            v = moonrange[i][3]
            if abs(cml - moonrange[i][1]) < 120 or abs(cml - moonrange[i][1]) > 240:
                ax.plot([x, y], [v, v], 'k-', lw=4)
                color, key = ('gold', 'IO') if i == 0 else ('aquamarine', 'EUR') if i == 1 else ('w', 'GAN')
                ax.plot([x, y], [v, v], color=color, linestyle='-', lw=2.5)
                ax.text(w, 3.5 + v, key, color=color, fontsize=10, alpha=0.5, path_effects=[mpl_patheffects.withStroke(linewidth=1, foreground='black')], horizontalalignment='center', verticalalignment='center', fontweight='bold')
    if fixed == 'lon':
        if hemis.lower()[0] == 'n':
            for i in range(3):
                x = 2 * np.pi - (np.radians(moonrange[i][1]))
                y = 2 * np.pi - (np.radians(moonrange[i][2]))
                w = 2 * np.pi - (np.radians(moonrange[i][0]))
                v = moonrange[i][3]
                if abs(cml - moonrange[i][1]) < 120 or abs(cml - moonrange[i][1]) > 240:
                    ax.plot([x, y], [v, v], 'k-', lw=4)
                    color, key = ('gold', 'IO') if i == 0 else ('aquamarine', 'EUR') if i == 1 else ('w', 'GAN')
                    ax.plot([x, y], [v, v], color=color, linestyle='-', lw=2.5)
                    ax.text(w, 3.5 + v, key, color=color, fontsize=10, alpha=0.5, path_effects=[mpl_patheffects.withStroke(linewidth=1, foreground='black')], horizontalalignment='center', verticalalignment='center', fontweight='bold')
        else:
            for i in range(3):
                x = np.radians(180 - moonrange[i][1])
                y = np.radians(180 - moonrange[i][2])
                w = np.radians(180 - moonrange[i][0])
                v = moonrange[i][3]
                if abs(cml - moonrange[i][1]) < 120 or abs(cml - moonrange[i][1]) > 240:
                    ax.plot([x, y], [v, v], 'k-', lw=4)
                    color, key = ('gold', 'IO') if i == 0 else ('aquamarine', 'EUR') if i == 1 else ('w', 'GAN')
                    ax.plot([x, y], [v, v], color=color, linestyle='-', lw=2.5)
                    ax.text(w, 3.5 + v, key, color=color, fontsize=10, alpha=0.5, path_effects=[mpl_patheffects.withStroke(linewidth=1, foreground='black')], horizontalalignment='center', verticalalignment='center', fontweight='bold')

def plot_regions(ax: plt.Axes, hemis: str, fixed: str, full: bool):
    if hemis.lower()[0] == 's':
        return
    updusk = np.linspace(np.radians(205), np.radians(170), 200)
    dawn = {k: np.linspace(v[0], v[1], 200) for k, v in (('lon', (np.radians(180), np.radians(130))), ('uplat', (33, 15)), ('downlat', (39, 23)))}
    noon_a = {k: np.linspace(v[0], v[1], 100) for k, v in (('lon', (np.radians(205), np.radians(190))), ('downlat', (28, 32)))}
    noon_b = {k: np.linspace(v[0], v[1], 100) for k, v in (('lon', (np.radians(190), np.radians(170))), ('downlat', (32, 27)))}
    ax.plot([np.radians(205), np.radians(205)], [20, 10], 'r-', lw=1.5)
    ax.plot([np.radians(170), np.radians(170)], [10, 20], 'r-', lw=1.5)
    ax.plot(updusk, 200 * [10], 'r-', lw=1.5)
    ax.plot(updusk, 200 * [20], 'r-', lw=1.5)
    c = 'k-' if fixed == 'lon' else 'b-'
    ax.plot([np.radians(130), np.radians(130)], [23, 15], c, lw=1)
    ax.plot([np.radians(180), np.radians(180)], [33, 39], c, lw=1)
    ax.plot(dawn['lon'], dawn['uplat'], c, lw=1)
    ax.plot(dawn['lon'], dawn['downlat'], c, lw=1)
    ax.plot([np.radians(205), np.radians(205)], [22, 28], 'y-', lw=1.5)
    ax.plot([np.radians(170), np.radians(170)], [27, 22], 'y-', lw=1.5)
    ax.plot(noon_a['lon'], noon_a['downlat'], 'y-', lw=1.5)
    ax.plot(noon_b['lon'], noon_b['downlat'], 'y-', lw=1.5)
    ax.plot(updusk, 200 * [22], 'y-', lw=1.5)
    ax.plot([np.radians(205), np.radians(205)], [10, 28], 'w--', lw=1)
    ax.plot([np.radians(170), np.radians(170)], [27, 10], 'w--', lw=1)
    ax.plot(noon_a['lon'], noon_a['downlat'], 'w--', lw=1)
    ax.plot(updusk, 200 * [10], 'w--', lw=1)

def save_figure(fig: plt.Figure, save_location: str, filename: str, **kwargs):
    sloc = fpath(save_location)
    ensure_dir(sloc)
    fig.savefig(f'{sloc}/{filename}', **kwargs)

def moind(file_location: str = None, save_location: str = None, filename: str = 'auto', crop: float = 1, rlim: float = 40, fixed: str = 'lon', hemis: str = 'North', full: bool = True, regions: bool = False, moonfp: bool = False, fileinfo: FitsFile = None, fitsdataheader: Tuple[np.ndarray, Dict] = None, preproj_func: Callable = None, **kwargs) -> Union[None, mpl.figure.Figure]:
    if fitsdataheader is None:
        image_data, header = load_fits_data(file_location, fileinfo)
    else:
        image_data, header = fitsdataheader
    if preproj_func is not None:
        image_data = preproj_func(image_data)
    image_data, rho, theta, mask, cml, dece = transform_to_polar(image_data, header, crop, rlim, fixed, hemis)
    fig, ax = create_figure(image_data, rho, theta, cml, rlim, fixed, hemis, full)
    if regions:
        plot_regions(ax, hemis, fixed, full)
    if moonfp:
        lon = {'io': (header['IOLON'], header['IOLON1'], header['IOLON2']), 'eu': (header['EULON'], header['EULON1'], header['EULON2']), 'ga': (header['GALON'], header['GALON1'], header['GALON2'])}
        plot_moon_footprints(ax, lon, cml, hemis, fixed)
    if filename == 'auto':
        finfo = FitsFile(file_location) if fileinfo is None else fileinfo
        filename = finfo._basename
        extras = "-".join([i for i in [(f'crop{crop}') if crop != 1 else '', (f'r{rlim}') if rlim != 90 else '', f'fix{fixed}', 'S' if hemis.lower()[0] == 's' else 'N', 'full' if full else '', 'regions' if regions else ''] if i != ''])
        filename += '-' + extras + '.jpg'
    save_figure(fig, save_location, filename, **kwargs)
    if 'return' in kwargs:
        return fig
    plt.close()


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
    infolist = [FitsFile(f) for f in fits_file_list]
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
