from pathlib import Path
import os
import datetime
import numpy as np
#! defines the project root directory (as the root of the gh repo) 
GHROOT = Path(__file__).parents[2]
#! if you move this file/folder, you need to change this line to match the new location. 
#! index matches the number of folders to go up from where THIS file is located: /[2] python/[1] jarvis/[0] const.py
# takes a relative path within repo and returns an absolute path.
def fpath(x):
    return os.path.join(GHROOT, x)
def rpath(x):
    return os.path.relpath(x, GHROOT)

def ensure_dir(file_path):
    '''this function checks if the file path exists, if not it will create one'''
    if not os.path.exists(file_path):
            os.makedirs(file_path)
def clock_format(x_rads, pos):
    # x_rads => 0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2, 7pi/4, ...
    # returns=> 00, 03, 06, 09, 12, 15, 18, 21,..., 00,03,06,09,12,15,18,21,
    cnum= int(np.degrees(x_rads)/15)
    return f'{cnum:02d}' if cnum%24 != 0 else '00'

# test, prints the README at the root of the project
def __testpaths():
    for x in os.listdir(GHROOT):
        print(x)
        

    



def fitsheader(fits_object, *args, ind=1,cust=True):
    ret = []
    
    for arg in args:
        if not isinstance(arg, str):
            raise TypeError('All arguments must be strings.')
        if arg.lower() =='south': # returns if_south as bool
            obj=fits_object[ind].header['HEMISPH']
            if obj.lower()[0] == 'n':
                obj=False
            elif obj.lower()[0] == 's':
                obj=True
            else:
                obj=''   
        elif arg.lower()=='fixed_lon': # returns if fixed_lon as bool
            obj=fits_object[ind].header['FIXED']
            if obj == 'LON':
                obj=True
            elif obj == 'LT':
                obj=False
            else:
                obj = ''  
        elif arg.lower() in ['fixed',]: # returns fixed as so
            obj=''
        else:
            obj=fits_object[ind].header[arg.upper()]
        ret.append(obj)
    return ret if len(ret) > 1 else ret[0]
def fits_from_parent(original_fits, new_data=None, **kwargs):
    fits_new = original_fits.copy()
    if new_data is not None:
        fits_new[1].data = new_data
    for k,v in kwargs.items():
        fits_new[1].header[k] = v
    return fits_new
        
           
def get_datetime(fits_object): # returns a datetime object from the fits header ##TODO: not implemented
    udate = fitsheader(fits_object, 'UDATE')         
    return datetime.datetime.strptime(udate, '%Y-%m-%d %H:%M:%S') 
#                                             '2016-05-19 20:48:59'   





#test = fileInfo(r'datasets\HST\v02\jup_16-138-18-48-16_0100_v02_stis_f25srf2_proj.fits')
#print(test.split)
#datasets\HST\v01\jup_16-137-23-43-30_0100_v01_stis_f25srf2_proj.fits
