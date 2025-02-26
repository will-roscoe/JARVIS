import os
import datetime
#import re
import numpy as np
import astropy.io.fits as fits
from typing import List
import glob

#from regex import P
from .const import GHROOT, FITSINDEX
from matplotlib import colors as mcolors
from tqdm import tqdm
from pathlib import Path

#################################################################################
#                   PATH/DIRECTORY MANAGMENT
#################################################################################
def ensure_dir(file_path):
    '''this function checks if the file path exists, if not it will create one'''
    if not os.path.exists(file_path):
            os.makedirs(file_path)

# test, prints the README at the root of the project
def __testpaths():
    for x in os.listdir(GHROOT):
        print(x)
def fpath(x):
    return os.path.join(GHROOT, x)
def rpath(x):
    return os.path.relpath(x, GHROOT)
def basename(x, ext=False):
    parts = x.split('/') 
    parts = parts[-1].split('\\') if '\\' in parts[-1] else parts
    base = parts[-1].split('.')[0] if not ext else parts[-1]
    return base
def split_path(x, include_sep=True):
    parts = Path(x).parts
    if include_sep:
        parts = [p + os.sep for p in parts[0:-1]] + [parts[-1]]
    return parts
    
#################################################################################
#                   FITS FILE INTERFACING
#################################################################################
def fits_from_glob(fits_dir:str, suffix:str='/*.fits', recursive=True, sort=True, names=False)->List[fits.HDUList]:
    """Returns a list of fits objects from a directory."""
    if isinstance(fits_dir,str):
          fits_dir = [fits_dir,]
    fits_file_list = []
    for f in fits_dir:
        for g in glob.glob(f + '/*.fits', recursive=True):
            fits_file_list.append(g)
    if sort:
        fits_file_list.sort()   
    tqdm.write(f'Found {len(fits_file_list)} files in the directory.')
    if names:
        return [fits.open(f) for f in fits_file_list] , [basename(f) for f in fits_file_list]
    return [fits.open(f) for f in fits_file_list] 
#################################################################################
#                  HDUList/FITS object FUNCTIONS
#################################################################################
def fitsheader(fits_object, *args, ind=FITSINDEX,cust=True):
    """Returns the header value of a fits object.
    Args:
        fits_object: fits.HDUList
        *args: list of header keys to return
        ind: index of the fits object in the HDUList
        cust: if True, returns custom values for 'south', 'fixed_lon', 'fixed', else will return the header value corresponding to the key."""
    #print(fits_object[ind].header.__dict__)
    ret = []
    #assert all(isinstance(arg, str) for arg in args), 'All arguments must be strings.'
    for arg in args:
        if arg.lower() =='south': # returns if_south as bool
            obj_l=fits_object[ind].header['HEMISPH'].lower()[0]
            obj = True if obj_l == 's' else False if obj_l == 'n' else ''
        elif arg.lower()=='fixed_lon': # returns if fixed_lon as bool
            obj_=fits_object[ind].header['FIXED']
            obj = False if obj_.lower() == 'lt' else True if obj_.lower() == 'lon' else ''
        elif arg.lower() in ['fixed',]: # returns fixed as so
            obj=''
        else:
            try:
                obj=fits_object[ind].header[arg.upper()]
            except: #noqa: E722
                obj=fits_object[ind].header[arg]

        ret.append(obj)
    return ret if len(ret) > 1 else ret[0]
def fits_from_parent(original_fits, new_data=None, **kwargs):
    """Returns a new fits object with the same header as the original fits object, but with new data and/or new header values."""
    orig_header = [original_fits[i].header.copy() for i in [0,1]]
    orig_data = [original_fits[i].data for i in [0,1]]
    
    for k,v in kwargs.items():
        orig_header[FITSINDEX][k] = v
    if new_data is not None:
        try:
            orig_header[FITSINDEX]['VERSION'] +=1
        except KeyError:
            orig_header[FITSINDEX]['VERSION'] = 1
    else:
        new_data = orig_data[FITSINDEX]
    fits_new = fits.HDUList([fits.PrimaryHDU(orig_data[0], header=orig_header[0]), fits.ImageHDU(new_data, header=orig_header[1])])
    #print(fits_new.info(), original_fits.info())
   
    return fits_new          
def get_datetime(fits_object):
    """Returns a datetime object from the fits header."""
    udate = fitsheader(fits_object, 'UDATE')         
    return datetime.datetime.strptime(udate, '%Y-%m-%d %H:%M:%S') # '2016-05-19 20:48:59'                     
def prepare_fits(fits_obj:fits.HDUList, regions=False, moonfp=False, fixed='lon', rlim=40,full=True, crop=1, **kwargs)->fits.HDUList:
    """Returns a fits object with specified header values, which can be used for processing."""
    kwargs.update({'REGIONS':bool(regions), 'MOONFP':bool(moonfp), 'FIXED':str(fixed).upper(), 'RLIM':int(rlim), 'FULL':bool(full), 'CROP': float(crop) if abs(float(crop)) <=1 else 1}) 
    return fits_from_parent(fits_obj,  **kwargs)
def make_filename(fits_obj:fits.HDUList):
    """Returns a filename based on the fits header values."""
    # might be better to alter the filename each time we process or plot something.
    args = ['CML', 'HEMISPH', 'FIXED', 'RLIM', 'FULL', 'CROP', 'REGIONS', 'MOONFP', 'VISIT', 'DOY', 'YEAR', 'EXPTIME']
    udate = get_datetime(fits_obj)
    cml, hem, fixed, rlim, full, crop, regions, moonfp, visit, doy, year, expt = [fitsheader(fits_obj, arg) for arg in args]
    extras = ",".join([m for m in [
    hem[0].upper(),
    fixed.upper(),
    f'{rlim}deg' if rlim != 40 else "",
    'full' if full else 'half',
    f'crop{crop}' if crop != 1 else "",
    'regions' if regions else "",
    'moonfp' if moonfp else ""] if m != ""])
    filename = f'jup_v{visit:0<2}_{doy:0<3}_{year}_{udate.strftime("%H%M%S")}_{expt:0>4}({extras})'
    return filename
def update_history(fits_object, *args):
    """Updates the HISTORY field of the fits header with the current time."""
    curr_hist=fitsheader(fits_object, 'HISTORY')
    if args is None:
        return curr_hist
    else:
        for arg in args:
            fits_object[FITSINDEX].header['HISTORY'] = arg + '@' + datetime.datetime.now().strftime('%y%m%d_%H%M%S') 
        
def debug_fitsheader(fits_obj:fits.HDUList):
    header = fits_obj[FITSINDEX].header
    print("\n".join([f"{k}: {v}"for k,v in header.items()]))
def debug_fitsdata(fits_obj:fits.HDUList):
    data = fits_obj[FITSINDEX].data
    print(data.shape)
    print(f"[0,:]:{data[0].shape} {data[0]}")
    print(f"[1,:]: {data[1].shape} {','.join([str(round(d,3)) for d in data[1,:5]])} ... {','.join([str(round(d,3)) for d in data[1,-5:]])}")
#################################################################################
#                    IMAGE UTILITIES
#################################################################################
def mcolor_to_lum(*colors):
    col = []
    for i,c in enumerate(colors):
        if not isinstance(c, int):
            # turn into rgb, could be mpl string color, hex , rgb
            c_ = mcolors.to_rgba(c)
            # turn into luminance int
            col.append(int((0.2126*c_[0]+0.7152*c_[1]+0.0722*c_[2])*255))
        else:
            col.append(c)
    if len(col) == 1:
        return col[0]
    return col






#################################################################################
#                   MISC/UNUSED FUNCTIONS
#################################################################################

def clock_format(x_rads, pos=None):
    """Converts radians to clock format."""
    # x_rads => 0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4, ...
    # returns=> 00, 03, 06, 09, 12, 15, 18, 21,..., 00,03,06,09,12,15,18,21,
    cnum= int(np.degrees(x_rads)/15)
    return f'{cnum:02d}' if cnum%24 != 0 else '00'



class Jfits:
    ind = FITSINDEX
    def __init__(self, fits_loc:str=None, fits_obj: fits.HDUList=None, **kwargs):
        if fits_loc is not None:
            self.loc = fits_loc
            self.hdul= fits.open(fits_loc)
        else:
            self.hdul = fits_obj
        self.header.update(kwargs)
    @property
    def data(self):
        return self.hdul[self.ind].data
    @property
    def header(self):
        return self.hdul[self.ind].header
    def update(self,data=None, **kwargs):
        if data is not None:
            self.hdul = fits_from_parent(self.hdul, new_data=data)
        for k,v in kwargs.items():
            self.hdul = fits_from_parent(self.hdul, **{k:v})
    def writeto(self, path:str):
        self.hdul.writeto(path)
    def close(self):
        self.hdul.close()
    def data_apply(self, func, *args, **kwargs):
        self.update(data=func(self.data, *args, **kwargs))
    def apply(self, func, *args, **kwargs):
        self = func(self, *args, **kwargs)
    def __del__(self):
        self.close()

gv_translation = dict(
 group =  [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], #group numbers
 visit =  [ 1,  2,  5,  4,  3,  8,  9, 10, 13, 15, 11, 12, 16, 18, 19, 20, 21, 23, 24, 29])

def group_to_visit(*args):
    ret = []
    for arg in args:
        arg = int(arg) 
        if arg in gv_translation['group']:
            ret.append(gv_translation['visit'][gv_translation['group'].index(arg)])
        else:
            ret.append(None)
    if len(ret) == 1:
        return ret[0]
    return ret

def visit_to_group(*args):
    ret = []
    for arg in args:
        arg = int(arg) 
        if arg in gv_translation['visit']:
            ret.append(gv_translation['group'][gv_translation['visit'].index(arg)])
        else:
            ret.append(None)
    if len(ret) == 1:
        return ret[0]
    return ret

def fitsdir(sortby='visit', full=False):
    fitspaths = []
    for g in gv_translation['group']:
        fitspaths.append(fpath(f'datasets/HST/group_{g:0>2}'))
    if sortby == 'visit':
        # sort the file paths by their respective visit number
        fitspaths = [fitspaths[i] for i in np.argsort(gv_translation['visit'])]
    if full:
        for i, f in enumerate(fitspaths):
            files = os.listdir(f)
            fitspaths[i] = [os.path.join(f, file) for file in files]
    return fitspaths

def hdulinfo(fits_obj:fits.HDUList):
    ret = []
    for i,hdu in enumerate(fits_obj):
        name = hdu.name
        type = hdu.__class__.__name__
        ret.append(dict(name=name,type=type))
    return ret
        
            
    