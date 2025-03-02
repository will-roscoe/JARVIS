"""
This module provides various utility functions for managing paths and directories, interfacing with FITS files, 
handling FITS headers and data, and performing time series operations.
"""
import os
import datetime
import numpy as np
import astropy.io.fits as fits
from astropy.table import Table, vstack
from typing import List
import glob
import astropy
from matplotlib import colors as mcolors
from tqdm import tqdm
from pathlib import Path
from .const import GHROOT, FITSINDEX


#################################################################################
#                   PATH/DIRECTORY MANAGMENT
#################################################################################



def ensure_dir(file_path):
    '''this function checks if the file path exists, if not it will create one'''
    if file_path[-1] != os.sep:
        file_path += os.sep
    if not os.path.exists(file_path):
            os.makedirs(file_path)


def ensure_file(file_path):
    '''this function checks if the file path exists, if not it will create one'''
    if file_path[-1] == os.sep:
        file_path = file_path[:-1]
    if not os.path.isfile(file_path):
            with open(file_path, 'w') as f:
                f.write('')


# test, prints the README at the root of the project
def __testpaths():
    for x in os.listdir(GHROOT):
        print(x)


def fpath(x):
    """Returns the absolute path of a file or directory in the project root."""
    return os.path.join(GHROOT, x)


def rpath(x):
    """Returns the relative path of a file or directory in the project root."""
    return os.path.relpath(x, GHROOT)


def filename_from_path(x, ext=False):
    """Returns the filename from a path, if ext is True, returns the filename with the extension, else returns the base filename."""
    parts = x.split('/') 
    parts = parts[-1].split('\\') if '\\' in parts[-1] else parts
    base = parts[-1].split('.')[0] if not ext else parts[-1]
    return base


def split_path(x, include_sep=True):
    """Splits a path into its parts, returns a list of the parts."""
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
        return [fits.open(f) for f in fits_file_list] , [filename_from_path(f) for f in fits_file_list]
    return [fits.open(f) for f in fits_file_list]


def hst_fitsfile_paths(sortby='visit', full=False):
    """Convienence function to return a list of file paths for the HST fits files.
    if full is True, returns a list of lists of file paths, else returns a list of file paths."""
    fitspaths = []
    for g in gv_translation['group']:
        fitspaths.append(fpath(f'datasets/HST/group_{g:0>2}'))
    if sortby == 'visit':
        # sort the file paths by their respective visit number
        fitspaths = [fitspaths[i] for i in np.argsort(gv_translation['visit'])]
    if full:
        for i, f in enumerate(fitspaths):
            files = os.listdir(f)
            fitspaths[i] = [os.path.join(f, file) for file in files if file.endswith('.fits')]
    return fitspaths


def hst_segmented_paths(segments=4,byvisit=True):
    """Returns a dictionary of file paths for the HST fits files, segmented into the specified number subgroups."""
    fitspaths = hst_fitsfile_paths(full=True)
    fdict = {}
    for i,f in enumerate(fitspaths,1):
        parts = np.array_split(f, segments)
        if not byvisit:
            for j,p in enumerate(parts):
                fdict[f'g{i:0>2}s{chr(65+j)}'] = p
        else:
            for j,p in enumerate(parts):
                fdict[f'v{group_to_visit(i):0>2}s{chr(65+j)}'] = p
    return fdict



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
            obj='' #TODO: implement this
        else:
            try:
                obj=fits_object[ind].header[arg.upper()]
            except: #noqa: E722
                obj=fits_object[ind].header[arg]
        ret.append(obj)
    return ret if len(ret) > 1 else ret[0]


def silent_fitsheader(fits_object, *args, ind=FITSINDEX):
    """Returns the header value of a fits object, if the key does not exist, returns an empty string. This method does not raise an error if the key does not exist."""
    try:
        return fitsheader(fits_object, *args, ind=ind)
    except KeyError:
        ret = []
        for arg in args:
            try:
                ret.append(fitsheader(fits_object, arg, ind=ind))
            except KeyError:
                ret.append('')
        return ret if len(ret) > 1 else ret[0]
    

def adapted_fits(original_fits, new_data=None, **kwargs):
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


def assign_params(fits_obj:fits.HDUList, regions=False, moonfp=False, fixed='lon', rlim=40,full=True, crop=1, **kwargs)->fits.HDUList:
    """Returns a fits object with specified header values, which can be used for processing."""
    kwargs.update({'REGIONS':bool(regions), 'MOONFP':bool(moonfp), 'FIXED':str(fixed).upper(), 'RLIM':int(rlim), 'FULL':bool(full), 'CROP': float(crop) if abs(float(crop)) <=1 else 1}) 
    return adapted_fits(fits_obj,  **kwargs)


def filename_from_fits(fits_obj:fits.HDUList):
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


def hdulinfo(fits_obj:fits.HDUList):
    """Returns a list of dictionaries containing the name and type of each HDU in the fits object."""
    ret = []
    for i,hdu in enumerate(fits_obj):
        name = hdu.name
        type = hdu.__class__.__name__
        ret.append(dict(name=name,type=type))
    return ret



#################################################################################
#                            IMAGE UTILITIES
#################################################################################



def mcolor_to_lum(*colors):
    """Converts color(s) to luminance values using matplotlib's color converter."""
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



################################################################################
#                         TIME (SERIES) UTILITIES
################################################################################



def get_obs_interval(fits_dirs):
    """Returns the observation interval from a list of fits directories or filepaths"""
    if isinstance(fits_dirs, str):
        fits_dirs = [fits_dirs,]
    fobjs = []
    for f in fits_dirs:
        if os.path.isfile(f):
            fit = fits.open(f)
            fobjs.append(fit)
        else:
            ffits = fits_from_glob(f)
            fobjs.extend(ffits)
    dates = [get_datetime(f) for f in fobjs]
    for f in fobjs:
        f.close
    return min(dates), max(dates)
        

def get_datetime(fits_object: fits.HDUList)->datetime.datetime:
    """Returns a datetime object from the fits header."""
    udate = fitsheader(fits_object, 'UDATE')         
    return datetime.datetime.strptime(udate, '%Y-%m-%d %H:%M:%S') # '2016-05-19 20:48:59'
    


def get_datetime_interval(fits_object: fits.HDUList)->tuple[datetime.datetime]:
    """Returns a tuple of the start and end datetimes of the observation interval from the fits header."""
    date_obs = fitsheader(fits_object, 'DATE-OBS')
    if len(date_obs)<19:
        date_obs += f"T{fitsheader(fits_object, 'TIME-OBS')}" 
    mindate = datetime.datetime.strptime(date_obs, '%Y-%m-%dT%H:%M:%S')
    try:
     date_end = fitsheader(fits_object, 'DATE-END')
     maxdate = datetime.datetime.strptime(date_end, '%Y-%m-%dT%H:%M:%S')
    except KeyError:
        maxdate = mindate + datetime.timedelta(seconds=fitsheader(fits_object,'EXPT'))    
    return mindate, maxdate


def get_timedelta(fits_object:fits.HDUList)->datetime.timedelta:
    """Returns the timedelta of the observation interval from the fits header."""
    start_time, end_time = get_datetime_interval(fits_object)
    return end_time - start_time


def datetime_to_yrdoysod(dt:datetime.datetime)-> tuple[int]:
    """Converts a datetime object to a tuple of year, day of year, and seconds of day."""
    dtt = dt.timetuple()
    return (dtt.tm_year, dtt.tm_yday, dtt.tm_hour*3600 + dtt.tm_min*60 + dtt.tm_sec)


def yrdoysod_to_datetime(year:int, doy:int, sod:int)->datetime.datetime:
    """Converts a tuple of year, day of year, and seconds of day to a datetime object."""
    return datetime.datetime(year, 1, 1) + datetime.timedelta(days=doy-1, seconds=sod)





#################################################################################
#                   MISC/UNUSED FUNCTIONS
#################################################################################



def clock_format(x_rads, pos=None):
    """Converts radians to clock format."""
    # x_rads => 0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4, ...
    # returns=> 00, 03, 06, 09, 12, 15, 18, 21,..., 00,03,06,09,12,15,18,21,
    cnum= int(np.degrees(x_rads)/15)
    return f'{cnum:02d}' if cnum%24 != 0 else '00'




gv_translation = dict(
 group =  [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], #group numbers
 visit =  [ 1,  2,  5,  4,  3,  8,  9, 10, 13, 15, 11, 12, 16, 18, 19, 20, 21, 23, 24, 29])



def group_to_visit(*args):
    """Converts group number to visit number. If multiple arguments are passed, returns a list of visit numbers."""
    ret = []
    for arg in args:
        arg = int(arg) 
        if arg in gv_translation['group']:
            ret.append(gv_translation['visit'][gv_translation['group'].index(arg)])
        else:
            ret.append(None)
    if len(ret) == 1:
        return ret[0]
    elif len(ret) == 0:
        raise ValueError(f'None found for {args}')
    return ret

def visit_to_group(*args):
    """Converts visit number to group number. If multiple arguments are passed, returns a list of group numbers."""
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

