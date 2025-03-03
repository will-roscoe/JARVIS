"""time utility functions for the jarvis package, specifically dealing with frequently needed conversions and FITS interfacing."""
from astropy.io.fits import open as fopen, HDUList, BinTableHDU, TableHDU
from datetime import datetime, timedelta
from os.path import isfile
from astropy.table import Table, vstack
from .utils import fits_from_glob, fitsheader
# defining __all__ stops any indirect imports from importing everything in the module, or circular imports.
__all__ = ['get_obs_interval', 'get_datetime', 'get_datetime_interval', 'get_timedelta', 'datetime_to_yrdoysod', 'yrdoysod_to_datetime', 'get_data_over_interval']
def get_obs_interval(fits_dirs):
    """Returns the observation interval from a list of fits directories or filepaths"""
    if isinstance(fits_dirs, str):
        fits_dirs = [fits_dirs,]
    fobjs = []
    for f in fits_dirs:
        if isfile(f):
            fit = fopen(f)
            fobjs.append(fit)
        else:
            ffits = fits_from_glob(f)
            fobjs.extend(ffits)
    dates = [get_datetime(f) for f in fobjs]
    for f in fobjs:
        f.close
    return min(dates), max(dates)
        

def get_datetime(fits_object: HDUList)->datetime:
    """Returns a datetime object from the fits header."""
    udate = fitsheader(fits_object, 'UDATE')         
    return datetime.strptime(udate, '%Y-%m-%d %H:%M:%S') # '2016-05-19 20:48:59'
    


def get_datetime_interval(fits_object: HDUList)->tuple[datetime]:
    """Returns a tuple of the start and end datetimes of the observation interval from the fits header."""
    date_obs = fitsheader(fits_object, 'DATE-OBS')
    if len(date_obs)<19:
        date_obs += f"T{fitsheader(fits_object, 'TIME-OBS')}" 
    mindate = datetime.strptime(date_obs, '%Y-%m-%dT%H:%M:%S')
    try:
     date_end = fitsheader(fits_object, 'DATE-END')
     maxdate = datetime.strptime(date_end, '%Y-%m-%dT%H:%M:%S')
    except KeyError:
        maxdate = mindate +timedelta(seconds=fitsheader(fits_object,'EXPT'))    
    return mindate, maxdate


def get_timedelta(fits_object:HDUList)->timedelta:
    """Returns the timedelta of the observation interval from the fits header."""
    start_time, end_time = get_datetime_interval(fits_object)
    return end_time - start_time


def datetime_to_yrdoysod(dt:datetime)-> tuple[int]:
    """Converts a datetime object to a tuple of year, day of year, and seconds of day."""
    dtt = dt.timetuple()
    return (dtt.tm_year, dtt.tm_yday, dtt.tm_hour*3600 + dtt.tm_min*60 + dtt.tm_sec)


def yrdoysod_to_datetime(year:int, doy:int, sod:int)->datetime:
    """Converts a tuple of year, day of year, and seconds of day to a datetime object."""
    return datetime(year, 1, 1) + timedelta(days=doy-1, seconds=sod)


def get_data_over_interval(fitsobjs, chosen_interval:list[datetime], data_index=2,include_parts=False)->Table:
    """Combines the data from the fits objects, and returns a table with the data over the given interval.
    fitsobjs: list of fits objects
    chosen_interval: list of two datetime objects, the interval to extract the data from
    data_index: the index of the data table in the fits object
    include_parts: if True, includes the parts of the each dataset that overlap with the interval, if False, only includes the data from fits that are completely within the interval."""
    valids = []
    interval_yds = {k: (st, en) for k, st, en in zip(['YEAR', 'DAYOFYEAR','SECOFDAY'], *[datetime_to_yrdoysod(i) for i in chosen_interval])}
    # sort fitsobjs by datetime (not strictly necessary, but means this might be able to be improved by a bisecting approach)
    fitsobjs = sorted(fitsobjs, key=lambda x: get_datetime_interval(x)[0])
    # first pick the fits objects with an interval that overlaps with the given interval
    # then extract the data.
    if include_parts:
        for f in fitsobjs:
            segment_interval = get_datetime_interval(f)
            if segment_interval[0] <= chosen_interval[1] and segment_interval[1] >= chosen_interval[0]:
                valids.append(f)
    else:
        for f in fitsobjs:
            segment_interval = get_datetime_interval(f)
            if segment_interval[0] >= chosen_interval[0] and segment_interval[1] <= chosen_interval[1]:
                valids.append(f)
    assert all([isinstance(f[data_index], (BinTableHDU, TableHDU))]), f"The HDU at index {data_index} is not a BinTableHDU or TableHDU for all fits objects."
    data = vstack([Table(f[data_index].data) for f in valids])
    if len(data) == 0:
        return None
    if include_parts:
        for key,(minv, maxv) in interval_yds.items(): # filter the data by the interval given
            data = data[(data[key]>=minv) & (data[key]<=maxv)]
    data.sort(list(interval_yds.keys()))   
    # finally add the epoch collumn, 0 = J2000
    data['EPOCH'] = data['YEAR'] + (data['DAYOFYEAR'] + data['SECOFDAY']/86400)/365.25
    return data

