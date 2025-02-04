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
from .polar import ensure_dir

def mkfit(file_location:str=None,save_location:str=None,filename:str='auto', crop:float = 1, rlim:float = 40,fileinfo:fileInfo=None,fitsdataheader:Tuple[np.ndarray,Dict]=None,preproj_func:Callable=None,**kwargs)->Union[None,mpl.figure.Figure]:  
    if fileinfo is None:
        f_abs = fpath(file_location)
    elif fileinfo is not None:
        f_abs = fileinfo.absolute_path
    with fits.open(f_abs) as hdulist:
            image_data = hdulist[1].data
    if preproj_func is not None:
        image_data = preproj_func(image_data)
    fig =plt.figure(figsize=(7,6))
    ax = plt.subplot()
    ax.set_facecolor('k') #black background   # shift position of LT labels
    finfo = fileInfo(file_location) if fileinfo is None else fileinfo
    plt.suptitle(f'Visit {finfo.visit} (DOY: {finfo.day}/{finfo.year}, {finfo.datetime})', y=0.99, fontsize=14)#one of the two titles for every plot
    ticks = kwargs.pop('ticks') if 'ticks' in kwargs else None
    cmap = kwargs.pop('cmap') if 'cmap' in kwargs else 'viridis'
    norm = kwargs.pop('norm') if 'norm' in kwargs else 'log'
    shrink = kwargs.pop('shrink') if 'shrink' in kwargs else 1
    plt.imshow(image_data,norm=norm, cmap=cmap)#!5 <- Color of the plot
    cbar = plt.colorbar(ticks=ticks, shrink=shrink, pad=0.06)
    sloc = fpath(save_location)
    ensure_dir(sloc)
    if filename == 'auto': # if a filename is not specified, it will be generated.
        filename = finfo._basename
        extras = []
        extras = "-".join(
            [i for i in ['raw',
                (f'crop{crop}') if crop !=1 else '', 
                (f'r{rlim}') if rlim !=90 else ''] if i != ''])
        filename += '-'+extras + '.jpg'
    fig.savefig(f'{sloc}/{filename}', **kwargs) # kwargs are passed to savefig, (dpi, quality, bbox, etc.)
    if 'return' in kwargs:
        return fig
    plt.show()
    plt.close()