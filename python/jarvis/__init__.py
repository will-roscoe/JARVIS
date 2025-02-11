'''jarvis package

This package contains modules for image processing and analysis of jupiter datasets.

Available submodules
---------------------
- polar: Module for processing HST FITS data and generating polar projections.
- const: package-wide constants and directory management.

Contributors
--------------
- Will Roscoe
- Samuel Courthold
- Yifan Chen
- Ronan Szeto
- polar module functions adapted from dmoral's original code.

const: 
    fpath, rpath, fitsheader, fits_from_parent, ensure_dir, clock_format
polar:
    




'''


from .const import *
from .polar import *
from .transforms import *

__all__ = ['moind', 'make_gif', 'fpath','rpath', 'fileInfo', 'const', 'polar', 'transforms', 'gradmap']
