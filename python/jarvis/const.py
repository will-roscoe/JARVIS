from pathlib import Path
import os
import datetime
from astropy.io import fits
import numpy as np
from typing import List, Union
import glob
#! defines the project root directory (as the root of the gh repo) 
GHROOT = Path(__file__).parents[2]
#! if you move this file/folder, you need to change this line to match the new location. 
#! index matches the number of folders to go up from where THIS file is located: /[2] python/[1] jarvis/[0] const.py
# takes a relative path within repo and returns an absolute path.
fpath = lambda x: os.path.join(GHROOT, x)
rpath = lambda x: os.path.relpath(x, GHROOT)


# test, prints the README at the root of the project
def __testpaths():
    for x in os.listdir(GHROOT):
        print(x)
        
class fileInfo():
    strOnly = False
    fits_index = 1
    def __init__(self, rel_path:str,):
         # define relative path
        if not os.path.exists(fpath(rel_path)): # raise if file does not exist, to catch errors early
            if os.path.exists(rel_path):
                # treat as absolute, remove the GHROOT from the beginning
                rel_path = rpath(rel_path)
            else:
                raise FileNotFoundError(f'File: {rel_path} not found by fileInfo. Check the path and try again.')
        self._rel_path = rel_path
            
        self._filename = os.path.split(rel_path)[1] # filename= basename.extension
        self._basename = os.path.basename(rel_path).split('.')[0]
        
        split = self._basename.split('_')
        split = [split[0]] + split[1].split('-') + split[2:]
        keys = ['observation', 'year','days','hours','minutes','seconds', 'exposuretime','visit','instrument','filter','type']
        self._dict = {k:v for k,v in zip(keys, split)}
        self._dict['year']= "20" + self._dict['year'] # contains only last two digits of year, so add 20
        if not self.strOnly: # if not string only, convert numeric values to int
            for k,v in self._dict.items():
                if v.isnumeric():
                    self._dict[k] = int(v)
            self._dict['visit'] = int(self._dict['visit'][1:]) # remove leading v
            
            
    def __getattr__(self, _name): # _dict items are accessible as self.xyz instead of self._dict['xyz']
        name = _name.lower()
        if name in self._dict.keys():
            return self._dict[name]
        else:
            for k,v in self._dict.items():
                if k.startswith(name):
                    return v
        return object.__getattr__(self, _name)

    @property
    def datetime(self): # returns a datetime object from the file name
        if self.strOnly:
            return datetime.datetime.strptime(self._dict['year'] + self._dict['days'] + self._dict['hours'] + self._dict['minutes'] + self._dict['seconds'], '%Y%j%H%M%S')
        return datetime.datetime(self.year, 1, 1, self.hours, self.minutes, self.seconds) + datetime.timedelta(days=self.days-1)
    def __getitem__(self, *args):
        if len(args) == 1:
            if isinstance(args[0], (int, slice)):
                return list(self._dict.values())[args[0]]
            return self.__getattr__(args[0])
        return [self.__getattr__(x) for x in args]
    



def fitsheader(fits_object, ind=1,cust=True, *args):
    ret = []
    for arg in args:
        if arg.lower() =='south':
            obj=fits_object[ind].header['HEMIS']
            if obj.lower()[0] == 'n':
                obj=False
            elif obj.lower()[0] == 's':
                obj=True
            else:
                obj=''   
        if arg.lower()=='fixed_lon':
            obj=fits_object[ind].header['FIXED']
            if obj == 'LON':
                obj=True
            elif obj == 'LT':
                obj=False
            else:
                obj = ''  
        elif arg.lower() in ['fixed',]:
            obj=''
        else:
            obj=fits_object[ind].header[arg.upper()]
        ret.append(obj)
def fits_from_parent(original_fits, new_data=None, **kwargs):
    fits_new = original_fits.copy()
    if new_data is not None:
        fits_new[1].data = new_data
    for k,v in kwargs.items():
        fits_new[1].header[k] = v
    return fits_new
        
           
def get_datetime(fits_object): # returns a datetime object from the fits header ##TODO: not implemented
    udate = fitsheader(fits_object, 'UDATE')         
    return datetime.datetime.strptime(udate, '%Y-%m-%dT%H:%M:%S.%f') 






#test = fileInfo(r'datasets\HST\v02\jup_16-138-18-48-16_0100_v02_stis_f25srf2_proj.fits')
#print(test.split)
#datasets\HST\v01\jup_16-137-23-43-30_0100_v01_stis_f25srf2_proj.fits
