from typing import List, Optional
import numpy as np
import scipy
#third party libraries
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

