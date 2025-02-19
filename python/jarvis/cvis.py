from calendar import c
import os
import random
import re

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
import cv2

from jarvis.utils import fitsheader, fpath, mcolor_to_lum, fits_from_parent, prepare_fits, ensure_dir,  basename
from jarvis.polar import plot_polar, process_fits_file
from astropy.io import fits
from jarvis.const import FITSINDEX, DPR_IMXY, DPR_JSON
from typing import List, Union
from tqdm import tqdm
from jarvis.transforms import  coadd,gaussian_blur,fullxy_to_polar_arr
from glob import glob
import matplotlib as mpl
import json

def cropimg(inpath:str,outpath:str=None,ax_background=255,img_background=0)->Union[None,np.ndarray]: 
    """Crop an image to the bounding box of the non 'image_background' pixels.
    inpath: path to the image to crop
    outpath: path to save the cropped image
    ax_background: the luminance value of the axis background
    """
    img = cv2.imread(inpath,cv2.IMREAD_GRAYSCALE)
    ax_lum, ext_lum =mcolor_to_lum(ax_background, img_background)
    # Find the bounding box of the non-black pixels
    coords = cv2.findNonZero(cv2.threshold(img, max([ext_lum-1,0]), ax_lum, cv2.THRESH_BINARY)[1])
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]
    if outpath is not None:
        cv2.imwrite(outpath, img)
    else:
        return img
def mask_top_stripe(img:np.ndarray,threshold=0.01, trimfrac=0.3, bg_color=0, facecolor=255)->np.ndarray:
    """Makes a mask to remove the top stripe of the image."""
    bg_color, facecolor= [i if isinstance(i,int) else mcolor_to_lum(i) for i in [bg_color, facecolor]]
    imwidth, imheight = img.shape
    trimmedimg = img[:,int(trimfrac*imwidth):int((1-trimfrac)*imwidth)]
    # get the first row containing a non-white or black value#
    for i in range(imheight):
        # if more than 1% of the row is not white or black, we assume it is data
        if np.sum(np.logical_and(trimmedimg[i] != bg_color, trimmedimg[i] != facecolor)) > threshold*imwidth:
            datai = i

            break
    else:
        return img
    # make a mask where all values below datai row are allowed, and above are excluded
    mask = np.ones_like(img) * 255
    mask[:datai,:] = 0
    return cv2.bitwise_and(img, mask)

def mk_stripped_polar(fits_obj: fits.HDUList, ax_background='white',img_background='black',crop=True, cmap=cmr.neutral, dpi=300,**kwargs)->np.ndarray:
    full = fitsheader(fits_obj, 'full')
    # do normal plotting with no decorations, just the circle
    fig= plt.figure(figsize=(6,6 if full else 3) , dpi=dpi,layout='none' )
    ax= fig.subplots(1,1,subplot_kw={'projection': 'polar'})
    plot_polar(fits_obj, ax,cmap=cmap,nodec=kwargs.pop('nodec', True), **kwargs)
    ax.set(facecolor=ax_background)
    fig.patch.set_facecolor(img_background)
    ax.set_aspect(1)
    ax.set_position( [0, -0.45, 1, 1.8] if not full else [0.05, 0.05, 0.9, 0.9])
    for spine in ax.spines.values():
        spine.set_linewidth(0)
    tempdir = fpath(r"temp/"+f"temp_stripped_{random.randint(0,99999999):0<8}.png")
    fig.savefig(tempdir, dpi=dpi)
    plt.close()
    img = cropimg(tempdir,ax_background=ax_background,img_background=ax_background) if crop else cv2.imread(tempdir)
    os.remove(tempdir)
    return img

def gen_gaussian_coadded_fits():
    savedir = fpath(r"datasets/HST/custom/")
    ensure_dir(savedir)
    pbar = tqdm(total=20)
    for i in [f"{m:0>2}" for m in range(1,21)]:
        path = fpath(f'datasets/HST/v{i}/*.fits')
        tqdm.write(path)
        targetpaths = glob(path)
        fitsfiles = [fits.open(x) for x in targetpaths]
        bases = [basename(x) for x in targetpaths]
        print(bases)
        smoothed = [gaussian_blur(fitsfiles[i][1].data, 3, 1) for i in range(len(fitsfiles))]
        coadded = coadd(smoothed)
        cofitsd = coadd([fitsfiles[i][1].data for i in range(len(fitsfiles))])
        cogauss = gaussian_blur(coadded, 3, 1)
        cofitsd = fits_from_parent(fitsfiles[0], new_data=cogauss)
        cofitsd[1].header['HISTORY'] = f'Coadded from {len(smoothed)} files, then gaussian blurred with 3,1'
        cofitsd.writeto(savedir+f"v{i}_coadded_gaussian[3_1].fits", overwrite=True)
        pbar.update(1)

def generate_contours(fits_obj:fits.HDUList, lrange=(0.2,0.4))->List[List[float]]:
    # generate a stripped down, grey scale image of the fits file
    proc =process_fits_file(prepare_fits(fits_obj, fixed='LON', full=True))
    img = mk_stripped_polar(proc, cmap=cmr.neutral, ax_background='white', img_background='black') 
    # if a pixel is not provided, ask the user to provide one, and show the image
    # normalize image
    normed = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    # create a mask of the image using the provided luminance range
    mask = cv2.inRange(normed, lrange[0]*255, lrange[1]*255)
    # smooth out the mask to remove noise
    kernel = np.ones((5, 5), np.uint8)  # Small kernel to smooth edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small noise
    # find the contours of the mask
    contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    #contours = [cv2.convexHull(cnt) for cnt in contours] # Convex hull of the contours
    # Find the contour that encloses the given point
    return contours, hierarchy, img

def select_contour(contours, hierarchy, img:np.ndarray, id_pixel=None):
    if id_pixel is None or isinstance(id_pixel, str):
        def on_click(event):
            global click_coords
            if event.xdata is not None and event.ydata is not None:
                click_coords = (event.xdata, event.ydata)
                plt.close()
        fig, ax = plt.subplots()
        ax.imshow(img, cmap=cmr.neutral)
        ax.set_title("Click on a point")
        event_connection = fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()
        id_pixel = click_coords
    
    selected_contour = None
    for contour in contours:
        if cv2.pointPolygonTest(contour, id_pixel, False) > 0:  # >0 means inside
            selected_contour = contour
            break  # Stop searching once we find the correct contour
    # if a contour is found, convert the contour points to polar coordinates and return them
    if selected_contour is not None:
        paths= selected_contour.reshape(-1, 2)
        paths = fullxy_to_polar_arr(paths, img, 40)
        return paths
    else:
        raise ValueError("No contour found for the selected pixel at the given luminance range.")
def plot_pathpoints(clist):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, polar=True)
    ax2 = fig.add_subplot(122)
    for c in clist:
        ax.scatter(np.radians([x[1] for x in c]), [x[0] for x in c], s=1)
        ax2.scatter([x[1] for x in c], [x[0] for x in c], s=1)

    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax2.invert_yaxis()
    ax2.invert_xaxis()
    plt.show()       
def generate_contourpoints(fits_obj:fits.HDUList,id_pixel=None, lrange=(0.2,0.4), ):
    contours, hierarchy, img = generate_contours(fits_obj, lrange)
    return select_contour(contours, hierarchy, img, id_pixel)
  

def pathtest():
    test = fits.open(fpath(r"datasets/HST/custom/v04_coadded_gaussian[3_1].fits"))
    clist = [generate_contourpoints(test, DPR_IMXY['04'], (0.25,0.35),)]
    plot_pathpoints(clist)
    return clist[0]

def savecontourpath(contours, path_id, **kwargs):
    with open(DPR_JSON, 'r') as f:
        data = json.load(f)

    # if only one contour is input, save it as a single path (only one if shape is (n,2)))
    c1 = contours[0]
    if isinstance(c1[0], (int, float)):
        data[path_id] = {'path':contours, **kwargs}
    else:
        for i, contour in enumerate(contours):
            data[f"{path_id}_{i}"] = {'path':contour, **kwargs}
    # if more than one contour is input, save them all as separate paths (if shape is (m,n,2))
    with open(DPR_JSON, 'w') as f:
        json.dump(data, f, indent=4)
def loadcontourpath(path_id, *args, regex=False):
    if not regex:
        with open(DPR_JSON, 'r') as f:
            data = json.load(f)
            contourd=data[path_id]
            if len(args) == 0:
                return contourd['path']
        return contourd['path'], [contourd[arg] for arg in args]
    # use regex to see if there are multiple paths with the same id
    with open(DPR_JSON, 'r') as f:
        data = json.load(f)
        contourd = {k:v for k,v in data.items() if re.match(path_id, k)}
        if len(contourd) == 0:
            raise ValueError(f"No paths found matching the regex {path_id}")
        if len(args) == 0:
            return [contourd[k]['path'] for k in contourd.keys()]
        return [contourd[k]['path'] for k in contourd.keys()], [[contourd[k][arg] for arg in args] for k in contourd.keys()]
        
    
    

    

  


