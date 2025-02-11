#import collections
import collections.abc as collections # If using Python version 3.10 and above
from datetime import datetime
import numpy as np
from pathlib import Path
import spiceypy as spice
from jarvis import fpath
# load lsk if not loaded already
lsk = 'python\\Kernels\\naif0012.tls' #'/Users/sarah/OneDrive - Lancaster University/Prog/Python/Galileo/kernels/naif0012.tls'
try:
    spice.kinfo(lsk)
except spice.stypes.SpiceyError:
    spice.kclear()
    spice.furnsh(lsk)

# Turns spice/ET time into python datetime
# Input: ET 1-D array/list or single value
def et2datetime(ettimes):
    isscalar = False
    if not isinstance(ettimes, collections.Iterable):
        isscalar = True
        ettimes = [ettimes]
    utctimes = spice.timout(ettimes, 'YYYY-MM-DD, HR:MN:SC.###')
    pytimes = np.array([datetime.strptime(iii, '%Y-%m-%d, %H:%M:%S.%f') for iii in utctimes])
    if isscalar:
        # return np.asscalar(pytimes)  # np.asscalar depricated in Numpy v1.16
        return np.ndarray.item(pytimes)
    else:
        return pytimes

# Turns python datetime into spice/ET time
# Input: datetime 1-D array/list or single value
def datetime2et(pytimes):
    isscalar = False
    if not isinstance(pytimes, collections.Iterable):
        isscalar = True
        pytimes = [pytimes]
    utctimes = np.array([datetime.strftime(iii, '%Y-%m-%d, %H:%M:%S.%f') for iii in pytimes])
    ettimes = np.array([spice.utc2et(iii) for iii in utctimes])
    if isscalar:
        # return np.asscalar(ettimes)
        return np.ndarray.item(ettimes)
    else:
        return ettimes

# Turns string into ET time
# list of strings or single string, datetime-compliant format string
def str2et(strs, fmt):
    if not np.shape(strs):
        strs = [strs]
    pytimes = np.array([datetime.strptime(iii, fmt) for iii in strs])
    if len(pytimes)==1:
        # return datetime2et(np.asscalar(pytimes))
        return datetime2et(np.ndarray.item(pytimes))
    else:
        return datetime2et(pytimes)

# Turns string into ET time
# list of ET times or single ET time, datetime-compliant format string
def et2str(et, fmt):
    if not isinstance(et, collections.Iterable):
        et = [et]
    pytimes = et2datetime(et)
    strs = np.array([datetime.strftime(iii, fmt) for iii in pytimes])
    if len(strs)==1:
        # return np.asscalar(strs)
        return np.ndarray.item(strs)
    else:
        return strs

# Turns python datetime into days after 2004-01-01 time
# 2004-01-01 is doy2004 0
def et2doy2004(ettimes):
    ettimes = np.array(ettimes)
    deltasec = ettimes - datetime2et(datetime(2004,1,1,0,0,0))
    doy2004 = deltasec/3600/24
    if np.size(doy2004)==1:
        # return np.asscalar(doy2004)
        return np.ndarray.item(doy2004)
    else:
        return doy2004

# Turns days after 2004-01-01 time into python datetime
# 2004-01-01 is doy2004 0
def doy20042et(doy2004):
    doy2004 = np.array(doy2004)
    deltasec = doy2004*3600*24
    ettimes = deltasec + datetime2et(datetime(2004,1,1,0,0,0))
    if np.size(ettimes)==1:
        # return np.asscalar(ettimes)
        return np.ndarray.item(ettimes)
    else:
        return ettimes

def datetime2doy2004(pytimes):
    ettimes = datetime2et(pytimes)
    doy2004 = et2doy2004(ettimes)
    return doy2004

def doy20042datetime(doy2004):
    ettimes = doy20042et(doy2004)
    pytimes = et2datetime(ettimes)
    return pytimes
