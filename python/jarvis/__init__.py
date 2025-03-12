#!/usr/bin/env python3
"""jarvis package.

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

"""

from warnings import warn

from .const import Dirs

try:
    import spiceypy as spice

    spice.furnsh(Dirs.KERNEL + "jupiter.mk")
    from .power import powercalc

except:  # noqa: E722

    def powercalc(*args, **kwargs):
        """Function requires SPICE kernels to be loaded."""  # noqa: D401
        raise NotImplementedError("Power calculations require SPICE kernels to be loaded.")

    warn("Kernel may need to be extracted for full functionality", ImportWarning)

import logging
import os

from .cvis import generate_contours, identify_contour, pathtest, plot_contourpoints, save_contour
from .polar import make_gif, moind, plot_moonfp, plot_polar, plot_regions, prep_polarfits
from .utils import (
    datetime_to_yrdoysod,
    ensure_dir,
    fits_from_glob,
    fitsheader,
    fpath,
    get_data_over_interval,
    get_datetime,
    get_datetime_interval,
    get_obs_interval,
    get_timedelta,
    group_to_visit,
    hdulinfo,
    hst_fpath_list,
    hst_fpath_segdict,
    visit_to_group,
    yrdoysod_to_datetime,
)

__version__ = "0.1.0-alpha"
__all__ = (
    ["cvis", "extensions", "stats", "transforms", "utils", "fpath", "fitsheader", "ensure_dir", "fits_from_glob", "hst_fitsfile_paths", "group_to_visit", "visit_to_group", "hst_segmented_paths", "hdulinfo", "get_obs_interval", "get_datetime", "get_datetime_interval", "get_timedelta", "datetime_to_yrdoysod", "yrdoysod_to_datetime", "get_data_over_interval", "polar", "moind", "make_gif", "plot_moonfp", "plot_regions", "plot_polar", "prep_polarfits", "cvis", "pathtest", "generate_contours", "identify_contour", "plot_contourpoints", "save_contour", "power", "powercalc", "const", "KERNELDIR", "FITSINDEX", "GHROOT", "Dirs.DATA", "PYDIR", "PKGDIR"]
)
AUTHORS = {
    "Will Roscoe": "@will-roscoe",
    "Samuel Courthold": "@samoo2000000",
    "Yifan Chen": "@Yeefhan",
    "Ronan Szeto": "@RonanSzeto",
}
SOURCES = {
    "dmoral": "polar module functions adapted from dmoral's original code.",
    "J Nichols, Joe Kinrade,": "power module/HST_emission_power.py adapted from",
}

BIBTEX = """
@software{Courthold_Jarvis_Jovian_DPR,
author = {Courthold, Samuel and Roscoe, Will and Chen, Yifan and Szeto, Ronan},
license = {CC-BY-NC-SA-4.0},
title = {{Jarvis: Jovian DPR Analytical Library}},
url = {https://github.com/will-roscoe/JARVIS}
}
"""
if not os.path.exists(Dirs.DATA):
    raise FileNotFoundError(f"Neccesary directory not found at {Dirs.DATA}, containing data files.")

# JLOGGER = logging.getLogger(__name__)
# JLOGGER.setLevel(logging.INFO)
# JLOGGER.info("JAR:VIS package loaded.")
