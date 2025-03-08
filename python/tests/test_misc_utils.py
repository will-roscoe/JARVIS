#!/usr/bin/env python3
import pytest  # noqa: F401
from astropy.io import fits
from jarvis.utils import assign_params, fpath, mcolor_to_lum

fits_obj = fits.open(fpath(r"datasets\HST\v04\jup_16-140-20-48-59_0103_v04_stis_f25srf2_proj.fits"))

prepared_fits = assign_params(fits_obj, regions=False, moonfp=False, fixed="lon", rlim=40, full=True, crop=1)


def test_mcolor_to_lum():
    assert mcolor_to_lum(1) == 1
    assert mcolor_to_lum("white") == 255
    assert mcolor_to_lum("black") == 0
    assert mcolor_to_lum("red", (0, 1, 0), "blue") == [54, 182, 18]
