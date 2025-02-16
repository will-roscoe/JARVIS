import re
from jarvis.transforms import align_cmls, coadd, gradmap, dropthreshold
from jarvis.polar import process_fits_file, plot_polar, prepare_fits, fits_from_parent
from jarvis.utils import fpath, fitsheader

from astropy.io import fits
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import cmasher as cmr
import random
import cv2
import matplotlib as mpl
targetpaths = glob(fpath(r'datasets\HST\v06\*.fits'))
fitsfiles = [fits.open(x) for x in targetpaths]
print(len(fitsfiles))
aligned_data = [x[1].data for x in fitsfiles]
coadded_data = coadd(aligned_data)
graddat = np.where(gradmap(coadded_data)<200,np.nan,gradmap(coadded_data))#gradmap(coadded_data)#
coadd2 = coadd([graddat,coadded_data], weights=[1,1])
procorig = prepare_fits(fitsfiles[0], fixed='LT', full=False)
proc = process_fits_file(fits_from_parent(procorig, new_data=coadd2))

# import numpy as np
# from matplotlib import pyplot as plt

def mk_stripped_polar(fits_obj,path=None,**kwargs)->None:
    dpi = kwargs.pop('dpi',300)
    fig= plt.figure(figsize=(6,3), dpi=dpi,layout='none' )
    ax= fig.subplots(1,1,subplot_kw={'projection': 'polar'})
    cmap, nodec = kwargs.pop('cmap',cmr.neutral), kwargs.pop('nodec',True)
    facecolor, bg_color,thetaa,fit=  kwargs.pop('facecolor',kwargs.get('bg_color', 'black')),kwargs.pop('bg_color','#00000000'), kwargs.pop('theta_adjust',None), kwargs.pop('len_fit',0)
    # do normal plotting with no decorations, just the circle
    plot_polar(fits_obj, ax,cmap=cmap,nodec=nodec, **kwargs)
    # adjust the thetalimits to remove the artefacts

    if thetaa is not None: # directly set the offsets
        tl = [np.pi/2+thetaa[0],3*np.pi/2+thetaa[1]]
    else:
        if fit==0:# set the offsets based on the amount of fits used
            t=0
        else:
            t = np.radians(4.208+(fit-1)*0.233)
        tl = [np.pi/2+t,3*np.pi/2]
        ax.set_thetalim(tl)
    ax.set(facecolor=facecolor)
    fig.patch.set_facecolor(bg_color)
  
    ax.set_aspect(1)
    ax.set_position( [0, -0.45, 1, 1.8])
    for spine in ax.spines.values():
        spine.set_linewidth(0)
   
    tempdir = fpath(r"temp/"+f"temp_stripped_{random.randint(0,99999999):0<8}.png")
    fig.savefig(tempdir, dpi=dpi)
    img = cv2.imread(tempdir,cv2.IMREAD_GRAYSCALE)
    bg_l, fc_l = bg_color, facecolor
    col = [bg_l, fc_l]
    for i,c in enumerate(col):
        # turn into rgb, could be mpl string color, hex , rgb
        c_ = mcolors.to_rgba(c)[:3]
        # turn into luminance int
        col[i] = int(0.2126*c_[0]+0.7152*c_[1]+0.0722*c_[2])
    bg_l, fc_l = col
    # Find the bounding box of the non-black pixels
    coords = cv2.findNonZero(cv2.threshold(img, fc_l-1, fc_l, cv2.THRESH_BINARY)[1])
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]
    if path is not None:
        cv2.imwrite(path, img)
    else:
        return img
    
img = mk_stripped_polar(proc,bg_color='black', facecolor='white', cmap=cmr.neutral,len_fit=32)
impath = fpath(r"pictures\gifs\coadded gradmap&coadd_v06_with200kRgradthresh.png")
def get_limit_mask(imagearr, path=None)->np.ndarray:
    print(imagearr.shape[0]) # height
    # identify the background color and the side that has the facecolor in the top corner
    bg = imagearr[-1, :]
    topl,topr = imagearr[0,0],imagearr[0,-1]
    left,right = imagearr[:imagearr.shape[0]//2, :], imagearr[:imagearr.shape[0]//2,:]
    testside,fc = [left,topl] if all(topl==bg) else [right,topr]
    cutoff = 0
    # test the test side to find the index of the first row
    for i in range(testside.shape[0]):
        if not np.all([testside[j,i] == bg or testside[j,i]==fc for j in range(testside.shape[1])]):
            cutoff = i
            break
    cropped = imagearr[cutoff:,:]
    if path is not None:
        cv2.imwrite(path, cropped)
    else:
        return cropped
get_limit_mask(img,impath)