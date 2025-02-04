import os
from dateutil.parser import parse
import datetime as dt
import glob
from typing import List, Tuple, Dict, Any, Union, Optional, Callable
import numpy as np
import scipy
#third party libraries
from astropy.io import fits
'''
convolution kernels used are from Swithenbank-Harris, B.G., Nichols, J.D., Bunce, E.J. 2019 
Gx = [[-1  0  1]    Gy = [[-1 -2 -1]                      G= [[-1-1j 0-2j 1-1j]
      [-2  0  2]          [ 0  0  0]  ==> G = Gx + j*Gy =     [-2    0     2  ]
      [-1  0  1]]         [ 1  2  1]]                         [-1+1j 0+2j  1+1j]
'''
def gradmap(input_arr:np.ndarray, kernel2d:np.array=np.array([[-1-1j,-2j,1-1j],[-2,0,2],[-1+1j,2j,1+1j]]),boundary:str='wrap',mode:str='same')->np.ndarray:
      complexret = scipy.signal.convolve2d(input_arr, kernel2d, mode='same', boundary='wrap')
      return np.abs(complexret)


def coadd(input_arrs:List[np.ndarray], weights:Optional[List[float]]=None)->np.ndarray:
    '''Coadd a list of arrays with optional weights.
    '''
    raise NotImplementedError('coadd not implemented yet')