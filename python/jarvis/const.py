from pathlib import Path
import os
import datetime
#! defines the project root directory (as the root of the gh repo) 
GHROOT = Path(__file__).parents[2]
#! if you move this file/folder, you need to change this line to match the new location. 
#! index matches the number of folders to go up from where THIS file is located: /[2] python/[1] jarvis/[0] const.py
# takes a relative path within repo and returns an absolute path.
fpath = lambda x: os.path.join(GHROOT, x)



# test, prints the README at the root of the project
def __testpaths():
    for x in os.listdir(GHROOT):
        print(x)
        
class fileInfo():
    strOnly = False
    def __init__(self, rel_path:str,):
        self._rel_path = rel_path # define relative path
        if not os.path.exists(fpath(rel_path)): # raise if file does not exist, to catch errors early
            raise FileNotFoundError(f'File {rel_path} not found by fileInfo')
        self._filename = os.path.split(rel_path)[1] # filename= basename.extension
        self._basename = os.path.basename(rel_path)
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
        return datetime.datetime(self.year, 1, 1, self.hours, self.minutes, self.seconds) + datetime.timedelta(days=self.days)
    def __getitem__(self, *args):
        if len(args) == 1:
            if isinstance(args[0], (int, slice)):
                return list(self._dict.values())[args[0]]
            return self.__getattribute__(args[0])
        return [self.__getattribute__(x) for x in args]


test = fileInfo(r'datasets\HST\v02\jup_16-138-18-48-16_0100_v02_stis_f25srf2_proj.fits')
#print(test.split)
#datasets\HST\v01\jup_16-137-23-43-30_0100_v01_stis_f25srf2_proj.fits
