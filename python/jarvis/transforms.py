from typing import List, Optional
import numpy as np
import scipy
import astropy.io.fits as fits
from .const import FITSINDEX
from .utils import fitsheader, fits_from_parent

##########################################################################################################
#                            COORDINATE SYSTEM TRANSFORMS
##########################################################################################################
def fullxy_to_polar(x,y, img, rlim=40)->tuple:
    r0 = img.shape[1]/2
    x_ = x - r0
    y_ = y - r0
    r = np.sqrt(x_**2 + y_**2)
    colat = r/r0 * rlim
    lon = np.degrees(np.arctan2(y_, x_)) +  90
    while lon < 0:
        lon += 360
    while lon > 360:
        lon -= 360
    return [colat,lon]
def fullxy_to_polar_arr(xys, img, rlim=40)->np.ndarray:
    """transforms a list of coordinates of points on an image (x,y=0,0 at top_left) to polar colatitude and longitude.
    xy: list of x,y coordinates of the points [[x1,y1],[x2,y2],...]
    img: the image the coordinates are from
    rlim: the maximum colatitude value
    """
    cl = []
    r0 = img.shape[1]/2
    
    for i,(x,y) in enumerate(xys): 
        x_ = x - r0
        y_ = y - r0
        r = np.sqrt(x_**2 + y_**2)
        colat = r/r0 * rlim
        lon = np.degrees(np.arctan2(y_, x_)) +  90
        while lon < 0:
            lon += 360
        while lon > 360:
            lon -= 360
        cl.append((colat,lon))
    return cl

def polar_to_fullxy(colat, lon, img, rlim)->np.ndarray:
    """transforms polar colatitude and longitude to full image coordinates.
    colat: the colatitude value
    lon: the longitude value
    img: the image the coordinates are from
    rlim: the maximum colatitude value
    """
    r0 = img.shape[1]/2
    x_ = r0 * np.cos(np.radians(lon-90))
    y_ = r0 * np.sin(np.radians(lon-90))
    rxy = colat/rlim * r0
    x = x_ + rxy
    y = y_ + rxy
    return [x,y]

##########################################################################################################
#                             IMAGE ARRAY TRANSFORMS
##########################################################################################################
def gaussian_blur(input_arr:np.ndarray, radius, amount,boundary:str='wrap',mode:str='same')->np.ndarray:
      '''Return the input array convolved with the kernel2d convolution kernel.
      radius = pixel radius of the kernel
      amount = standard deviation of the gaussian kernel
      '''
      kernel2d = np.exp(-np.arange(-radius,radius+1)**2/(2*amount**2))
      kernel2d = np.outer(kernel2d, kernel2d)
      kernel2d /= np.sum(kernel2d)
      return scipy.signal.convolve2d(input_arr, kernel2d, mode=mode, boundary=boundary)
      
def gradmap(input_arr:np.ndarray, kernel2d:np.array=np.array([[-1-1j,-2j,1-1j],[-2,0,2],[-1+1j,2j,1+1j]]),boundary:str='wrap',mode:str='same')->np.ndarray:
      '''Return the gradient of the input array using the kernel2d convolution kernel.
convolution kernels used are from Swithenbank-Harris, B.G., Nichols, J.D., Bunce, E.J. 2019 
Gx = [[-1  0  1]    Gy = [[-1 -2 -1]                      G= [[-1-1j 0-2j 1-1j]
      [-2  0  2]          [ 0  0  0]  ==> G = Gx + j*Gy =     [-2    0     2  ]
      [-1  0  1]]         [ 1  2  1]]                         [-1+1j 0+2j  1+1j]
'''
      complexret = scipy.signal.convolve2d(input_arr, kernel2d, mode='same', boundary='wrap')
      return np.abs(complexret)
def dropthreshold(input_arr:np.ndarray, threshold:float)->np.ndarray:
      '''Return a the input array, but with valuesbelow the threshold set to 0.'''
      return np.where(input_arr<threshold, 0, input_arr)
def coadd(input_arrs:List[np.ndarray], weights:Optional[List[float]]=None)->np.ndarray:
      '''Coadd a list of arrays with optional weights.
      input arrs N arrays of x*y shape
      '''
      if weights is None:
            weights = [1 for i in range(len(input_arrs))]
      combined = np.stack(input_arrs, axis=0)
      return np.average(combined, axis=0, weights=weights)
def normalize(input_arr: np.ndarray) -> np.ndarray:
    '''Normalize the input array to the range [0, 1].'''
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)
    return (input_arr - min_val) / (max_val - min_val)
##########################################################################################################
#                         FITS DATA TRANSFORMS
##########################################################################################################
def adaptive_coadd(input_fits:List[fits.HDUList], eff_ang=90)-> fits.HDUList:
      # each array has better resolution closer to cml, and worse at the edges. we need to identify how 
      cmls = [fitsheader(f,'CML') for f in input_fits]
      effective_lims = [[cml-eff_ang, cml+eff_ang] for cml in cmls]
      for i in range(len(effective_lims)):
            if effective_lims[i][0] < 0:
                  effective_lims[i][0] += 360
            if effective_lims[i][1] > 360:
                  effective_lims[i][1] -= 360
      #for each longitude, we need to identify which of the input fits objects are relevant
def align_cmls(input_fits:List[fits.HDUList], primary_index:int)-> List[fits.HDUList]:
      """Align a list of fits objects to a primary fits object by rolling the images to match the CML positions."""
      primary_fits = input_fits[primary_index] # align all images to this one
      data0,header0 = primary_fits[FITSINDEX].data, primary_fits[FITSINDEX].header # 'zero point' data and fits
      cml0 = header0['CML'] # 'zero point' cml
      assert all([f[FITSINDEX].data.shape == data0.shape for f in input_fits]), 'All images must have the same shape.'
      height,width = data0.shape # get shape, so we can identify index to roll by
      diffs = [cml0 - fitsheader(f,'CML') for f in input_fits] # angle differences
      dwidths =[int(d/360*width) for d in diffs] # index/pixel differences
      aligned = [np.roll(f[FITSINDEX].data, d, axis=1) for f,d in zip(input_fits,dwidths)] # roll each image
      return [fits_from_parent(f,new_data=arr) for f,arr in zip(input_fits,aligned)] # return new fits objects.
