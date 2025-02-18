from typing import List, Optional
import numpy as np
import scipy
import astropy.io.fits as fits
from .const import FITSINDEX
from .utils import fitsheader, fits_from_parent
#third party libraries
'''
convolution kernels used are from Swithenbank-Harris, B.G., Nichols, J.D., Bunce, E.J. 2019 
Gx = [[-1  0  1]    Gy = [[-1 -2 -1]                      G= [[-1-1j 0-2j 1-1j]
      [-2  0  2]          [ 0  0  0]  ==> G = Gx + j*Gy =     [-2    0     2  ]
      [-1  0  1]]         [ 1  2  1]]                         [-1+1j 0+2j  1+1j]
'''
def gradmap(input_arr:np.ndarray, kernel2d:np.array=np.array([[-1-1j,-2j,1-1j],[-2,0,2],[-1+1j,2j,1+1j]]),boundary:str='wrap',mode:str='same')->np.ndarray:
      '''Return the gradient of the input array using the kernel2d convolution kernel.'''
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
def adaptive_coadd(input_fits:List[fits.HDUList])-> fits.HDUList:
      # each array has better resolution closer to cml, and worse at the edges. we need to identify how 
      cmls = [fitsheader(f,'CML') for f in input_fits]
      latlen, lonlen = input_fits[0][FITSINDEX].data.shape
      weights = []
      for i in cmls:
            indcml = int(i / 360 * lonlen)
            # create pdf distribution of len width, with peak at centre
            exp_ = np.exp(-0.5 * ((np.arange(lonlen) - lonlen // 2) / (lonlen / 12)) ** 2)
            # shift the peak to the cml
            shifted = np.roll(exp_, indcml - lonlen // 2)
            stacked = np.stack([shifted for _ in range(latlen)], axis=0)
            weights.append(stacked)
      weights3d = np.stack(weights, axis=0)
      coadded = coadd([f[FITSINDEX].data for f in input_fits], weights=weights3d)
      return fits_from_parent(input_fits[0], new_data=coadded)




def normalize(input_arr: np.ndarray) -> np.ndarray:
    '''Normalize the input array to the range [0, 1].'''
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)
    return (input_arr - min_val) / (max_val - min_val)

      
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
