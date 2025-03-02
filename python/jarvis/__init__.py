'''jarvis package

This package contains modules for image processing and analysis of jupiter datasets.

Available submodules
---------------------
- polar: Module for processing HST FITS data and generating polar projections.
- const: package-wide constants and directory management.
- utils: utility functions for file management and data manipulation.
- transforms: functions for transforming image data.
- cvis: functions for contour generation and plotting using OpenCV.
- power: functions for calculating power spectra of image data.
- extensions: functions for extending the functionality of the package or low-importance functions.

Primary Contributors
--------------
- Will Roscoe
- Samuel Courthold
- Yifan Chen
- Ronan Szeto

'''
import spiceypy as spice
from .const import KERNELDIR, FITSINDEX, GHROOT, DATADIR, PYDIR, PKGDIR
try:
    spice.furnsh(KERNELDIR+'jupiter.mk')
    from .power import powercalc
    
except:  #noqa: E722
    def powercalc(*args, **kwargs):
        """Function requires SPICE kernels to be loaded."""
        raise NotImplementedError('Power calculations require SPICE kernels to be loaded.')
    print('Kernel may need to be extracted for full functionality')

import logging
import os
from .utils import (fpath, fitsheader, ensure_dir, fits_from_glob, hst_fitsfile_paths, 
                assign_params, adapted_fits, group_to_visit, visit_to_group)
from .polar import (moind, make_gif, plot_moonfp, plot_regions, plot_polar, 
                    prep_polarfits)

from .cvis import (pathtest, generate_contours, identify_contour, 
                plot_contourpoints, save_contour)
__version__ = '0.1.0-alpha'
__all__ = ['polar', 'const', 'utils', 'cvis', 'power', 
    'fpath', 'fitsheader', 'ensure_dir', 'fits_from_glob', 'hst_fitsfile_paths', 
    'assign_params', 'adapted_fits', 'group_to_visit', 'visit_to_group',
    'moind', 'make_gif', 'plot_moonfp', 'plot_regions', 'plot_polar', 
    'prep_polarfits', 'powercalc', 'plotcontours', 'pathtest', 'generate_contours', 
    'identify_contour', 'plot_contourpoints', 'save_contour', 'KERNELDIR', 
    'FITSINDEX', 'GHROOT', 'DATADIR', 'PYDIR', 'PKGDIR',]
AUTHORS = {
    'Will Roscoe': '@will-roscoe',
    'Samuel Courthold': '@samoo2000000',
    'Yifan Chen': '@Yeefhan',
    'Ronan Szeto': '@RonanSzeto'}
SOURCES = {'dmoral': 'polar module functions adapted from dmoral\'s original code.',
           'J Nichols, Joe Kinrade,': 'power module/HST_emission_power.py adapted from'}
   
BIBTEX = '''
@software{Courthold_Jarvis_Jovian_DPR,
author = {Courthold, Samuel and Roscoe, Will and Chen, Yifan and Szeto, Ronan},
license = {CC-BY-NC-SA-4.0},
title = {{Jarvis: Jovian DPR Analytical Library}},
url = {https://github.com/will-roscoe/JARVIS}
}
'''
if not os.path.exists(DATADIR):
    raise FileNotFoundError(f"Neccesary directory not found at {DATADIR}, containing data files.")

JLOGGER = logging.getLogger(__name__)
JLOGGER.setLevel(logging.INFO)
JLOGGER.info("JAR:VIS package loaded.")



