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
    proc




'''


from .utils import fpath, rpath, fitsheader, fits_from_parent, ensure_dir, clock_format, get_datetime, prepare_fits, make_filename, update_history, fits_from_glob
from .polar import plot_moonfp, moind, make_gif,plot_polar,plot_regions,process_fits_file
from .transforms import gradmap, coadd, normalize, align_cmls
from .const import GHROOT, FITSINDEX

__all__ = ['const', 
           'polar', 
           'transforms',
           'utils',
           'fpath', 
           'rpath', 
           'fitsheader', 
           'fits_from_parent', 
           'ensure_dir',
           'clock_format', 
           'plot_moonfp', 
           'moind', 
           'make_gif', 
           'plot_polar', 
           'plot_regions', 
           'process_fits_file', 
           'gradmap', 'coadd', 
           'normalize', 
           'align_cmls',
            'mkgif',
            'fits_from_glob', 
            'get_datetime',
            'prepare_fits',
            'make_filename',
            'update_history',
            'GHROOT',
            'FITSINDEX',
           ]
