from pathlib import Path
import os
import datetime
from astropy.io import fits
import numpy as np
from typing import List, Union, Dict

GHROOT = Path(__file__).parents[2]

def fpath(x): 
    return os.path.join(GHROOT, x)

def rpath(x): 
    return os.path.relpath(x, GHROOT)

def relpath(x):
    if not os.path.exists(fpath(x)):
        if os.path.exists(x):
            x = rpath(x)
        else:
            raise FileNotFoundError(f'File: {x} not found/couldn\'t be turned into a relative path. Check the path and try again.')
    return x

def abspath(x):
    if not os.path.exists(x):
        if os.path.exists(rpath(x)):
            x = rpath(x)
        else:
            raise FileNotFoundError(f'File: {x} not found/couldn\'t be turned into an absolute path. Check the path and try again.')
    return x

def __testpaths():
    for x in os.listdir(GHROOT):
        print(x)

class FitsFile:
    def __init__(self, path: str):
        self._path = fpath(path)
        self._data = fits.open(self._path)[1].data
        self._header = fits.open(self._path)[1].header

    @property
    def data(self):
        return self._data

    @property
    def header(self):
        return self._header

    def __getattr__(self, name):
        name = name.upper()
        if name in self._header:
            return self._header[name]
        raise AttributeError(f"'FitsFile' object has no attribute '{name}'")

    def __getitem__(self, key):
        return self._data[key]

    def __repr__(self):
        return f"FitsFile(path={self._path})"

class FitsFileSeries:
    def __init__(self, filepaths: List[str] = None, filedir: str = None, glob_pattern: str = None, expect: int = None, homogenous: Union[List[str], str] = False, timesort: bool = True):
        if filepaths is not None:
            self.filepaths = filepaths
        elif filedir is not None:
            self.filepaths = [fpath(x) for x in os.listdir(fpath(filedir))]
        elif glob_pattern is not None:
            self.filepaths = glob.glob(fpath(glob_pattern))
        if expect is not None:
            assert len(self.filepaths) == expect, f'Expected {expect} files, found {len(self.filepaths)}'
        self._files = [FitsFile(x) for x in self.filepaths]
        if timesort:
            self._files.sort(key=lambda x: x.header['DATE-OBS'])
        if homogenous:
            if isinstance(homogenous, str):
                homogenous = [homogenous]
            for attr in homogenous:
                values = [getattr(file, attr) for file in self._files]
                assert all(value == values[0] for value in values), f'Values for {attr} are not homogenous: {values}'

    @property
    def files(self):
        return self._files

    @property
    def data_arrays(self):
        return [file.data for file in self._files]

    @property
    def data_array(self):
        return np.stack(self.data_arrays, axis=0)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index):
        return self._files[index]

    def __iter__(self):
        return iter(self._files)

    def __repr__(self):
        return f"FitsFileSeries(files={self._files})"
