from math import e
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
import matplotlib.colors as mcolors
import cv2
import random
from jarvis.utils import fitsheader, fpath
from jarvis.polar import plot_polar

def halfxy_to_polar(x_pixel:float,y_pixel:float,x_0:float,rlim:float)-> np.ndarray:
    """transforms a coordinate of a point on an image (x,y=0,0 at top_left) to polar colatitude and longitude.
    x_pixel: x coordinate of the point
    y_pixel: y coordinate of the point
    x_0: x coordinate of the center of the image/ origin of the polar coordinate system.
    assumes the height=1/2 width of the image, and is calibrated to 90°<=colat<=270°
    """
    vec_p0 = np.array([x_0,0])
    vec_p = np.array([x_pixel,y_pixel])
    vec_pcal = vec_p - vec_p0
    pcal = np.linalg.norm(vec_pcal)
    colat = pcal/x_0 * rlim
    lon = np.degrees(np.arctan2(vec_pcal[1], vec_pcal[0])) + 270
    return np.array(colat,lon)
def halfxy_to_polar_arr(xy:np.ndarray,x_0:float,rlim:float)->np.ndarray:
    """transforms a list of coordinates of points on an image (x,y=0,0 at top_left) to polar colatitude and longitude.
    xy: list of x,y coordinates of the points
    x_0: x coordinate of the center of the image/ origin of the polar coordinate system.
    assumes the height=1/2 width of the image, and is calibrated to 90°<=colat<=270°
    """
    vec_ps = np.array(xy)
    vec_p0 = np.array([x_0,0])
    vec_pcal = vec_ps - vec_p0
    pcal = np.linalg.norm(vec_pcal,axis=1)
    colat = pcal/x_0 * rlim
    lon = np.degrees(np.arctan2(vec_pcal[:,1], vec_pcal[:,0])) + 270
    return np.array([colat,lon])


def mk_stripped_polar(fits_obj,path=None,**kwargs)->None:
    dpi = kwargs.pop('dpi',300)
    
    cmap, nodec = kwargs.pop('cmap',cmr.neutral), kwargs.pop('nodec',True)
    facecolor, bg_color=  kwargs.pop('facecolor',kwargs.get('bg_color', 'black')),kwargs.pop('bg_color','#00000000')
    full = fitsheader(fits_obj, 'full')
    thetalims = kwargs.pop('thetalims',None)
    # do normal plotting with no decorations, just the circle
    fig= plt.figure(figsize=(6,6 if full else 3) , dpi=dpi,layout='none' )
    ax= fig.subplots(1,1,subplot_kw={'projection': 'polar'})
    plot_polar(fits_obj, ax,cmap=cmap,nodec=nodec, **kwargs)
    ####################################################################################################
    # # adjust the thetalimits to remove the artefacts, currently manually needs to be adjusted
    if thetalims is not None:
        tl = thetalims
    else:
        tl = [0,2*np.pi] if full else [np.pi/2,3*np.pi/2]
    ax.set_thetalim(tl)
    ####################################################################################################
    ax.set(facecolor=facecolor)
    fig.patch.set_facecolor(bg_color)
    
    ax.set_aspect(1)
    ax.set_position( [0, -0.45, 1, 1.8] if full else [0, 0, 1, 1])
    for spine in ax.spines.values():
        spine.set_linewidth(0)
    tempdir = fpath(r"temp/"+f"temp_stripped_{random.randint(0,99999999):0<8}.png")
    fig.savefig(tempdir, dpi=dpi)
    return cropimg(tempdir,facecolor=facecolor,bg_color=bg_color)
   
def cropimg(inpath,outpath=None,facecolor='black',bg_color='white'): 
    img = cv2.imread(inpath,cv2.IMREAD_GRAYSCALE)
    bg_l, fc_l =mcolor_to_lum(bg_color, facecolor)
    # Find the bounding box of the non-black pixels
    coords = cv2.findNonZero(cv2.threshold(img, fc_l-1, fc_l, cv2.THRESH_BINARY)[1])
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]
    if outpath is not None:
        cv2.imwrite(outpath, img)
    else:
        return img
def mcolor_to_lum(*colors):
    col = [0]*len(colors)
    for i,c in enumerate(colors):
        # turn into rgb, could be mpl string color, hex , rgb
        c_ = mcolors.to_rgba(c)[:3]
        # turn into luminance int
        col[i] = int(0.2126*c_[0]+0.7152*c_[1]+0.0722*c_[2])
    return col