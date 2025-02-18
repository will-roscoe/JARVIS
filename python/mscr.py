import tqdm
from jarvis.transforms import align_cmls, coadd, gradmap, dropthreshold, gaussian_blur
from jarvis.polar import process_fits_file,  prepare_fits, fits_from_parent
from jarvis.utils import fpath, ensure_dir, make_filename, debug_fitsdata, basename
import os
from astropy.io import fits
from glob import glob
from tqdm import tqdm

import numpy as np
import cmasher as cmr

import cv2

from ml import mk_stripped_polar, mcolor_to_lum


ensure_dir(fpath(r"temp/")) 
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
        cofits = fits_from_parent(fitsfiles[0], new_data=coadded)
        cofits[1].header['HISTORY'] = f'Coadded from {len(smoothed)} files with gaussian blur 3,1 pre-applied'
        cofits.writeto(savedir+f"v{i}_gaussian[3_1]_coadded.fits", overwrite=True)

        cofitsd = coadd([fitsfiles[i][1].data for i in range(len(fitsfiles))])
        cogauss = gaussian_blur(coadded, 3, 1)
        cofitsd = fits_from_parent(fitsfiles[0], new_data=cogauss)
        cofitsd[1].header['HISTORY'] = f'Coadded from {len(smoothed)} files, then gaussian blurred with 3,1'
        cofitsd.writeto(savedir+f"v{i}_coadded_gaussian[3_1].fits", overwrite=True)
        pbar.update(1)
    
for i in [f"{m:0>2}" for m in range(1,21)]:
    fitsfile =  fits.open(fpath(f'datasets/HST/custom/v{i}_coadded_gaussian[3_1].fits'))
    d = fitsfile[1].data 
    grads = gradmap(d)
    proc =process_fits_file(prepare_fits(fitsfile, fixed='LT', full=False))
    img = mk_stripped_polar(proc, cmap=cmr.neutral, facecolor='white', bg_color='black') 
    cv2.imwrite(fpath(f"temp/v{i}.png"), img)

    # # # make a mask removing the top stripe of the image which will only contain black and white (0,255) values
    masked_img = mask_top_stripe(img, facecolor='white', bg_color='black')
    mask = cv2.inRange(masked_img, 0.3*255, 0.6*255)
    kernel = np.ones((3, 3), np.uint8)  # Small kernel to smooth edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small noise
    
    #ret, thresh = cv2.threshold(masked_img, 255*0.4, 255*0.7, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    #contours = [cv2.convexHull(cnt) for cnt in contours]
    # # # save the image
    
    cv2.drawContours(image=img, contours=contours, contourIdx=-1,  color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imwrite(fpath(f"temp/v{i}proc.jpg"), img)


rmglob = glob(fpath(r"temp/temp*.png"))
for f in rmglob:
    os.remove(f)




    #cv2.imwrite(fpath(f"pictures/bw/{make_filename(proc)}_bw_stripped.jpg"), img)



