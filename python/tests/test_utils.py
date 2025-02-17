import os
import datetime
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import astropy.io.fits as fits
from jarvis.utils import (
    ensure_dir, clock_format, fitsheader, fits_from_parent, get_datetime,
    prepare_fits, make_filename, update_history, fits_from_glob, Jfits
)

def test_ensure_dir(tmp_path):
    test_dir = tmp_path / "test_dir"
    ensure_dir(test_dir)
    assert os.path.exists(test_dir)

def test_clock_format():
    assert clock_format(0) == '00'
    assert clock_format(np.pi / 4) == '03'
    assert clock_format(np.pi / 2) == '06'
    assert clock_format(3 * np.pi / 4) == '09'
    assert clock_format(np.pi) == '12'
    assert clock_format(5 * np.pi / 4) == '15'
    assert clock_format(3 * np.pi / 2) == '18'
    assert clock_format(7 * np.pi / 4) == '21'

def test_fitsheader():
    header = {'HEMISPH': 'NORTH', 'FIXED': 'LON', 'UDATE': '2016-05-19 20:48:59'}
    fits_object = MagicMock()
    fits_object[0].header = header
    assert fitsheader(fits_object, 'south', ind=0) == False
    assert fitsheader(fits_object, 'fixed_lon', ind=0) == True
    assert fitsheader(fits_object, 'UDATE', ind=0) == '2016-05-19 20:48:59'

def test_fits_from_parent():
    header = {'VERSION': 1}
    data = np.zeros((10, 10))
    fits_object = fits.HDUList([fits.PrimaryHDU(data), fits.ImageHDU(data, header=header)])
    new_fits = fits_from_parent(fits_object, new_data=np.ones((10, 10)))
    assert new_fits[1].header['VERSION'] == 2
    assert np.array_equal(new_fits[1].data, np.ones((10, 10)))

def test_get_datetime():
    header = {'UDATE': '2016-05-19 20:48:59'}
    fits_object = MagicMock()
    fits_object[0].header = header
    assert get_datetime(fits_object) == datetime.datetime(2016, 5, 19, 20, 48, 59)

def test_prepare_fits():
    header = {'VERSION': 1}
    data = np.zeros((10, 10))
    fits_object = fits.HDUList([fits.PrimaryHDU(data), fits.ImageHDU(data, header=header)])
    prepared_fits = prepare_fits(fits_object, regions=True, moonfp=True, fixed='lon', rlim=40, full=True, crop=1)
    assert prepared_fits[1].header['REGIONS'] == True
    assert prepared_fits[1].header['MOONFP'] == True

def test_make_filename():
    header = {
        'CML': 'CML', 'HEMISPH': 'NORTH', 'FIXED': 'LON', 'RLIM': 40, 'FULL': True,
        'CROP': 1, 'REGIONS': True, 'MOONFP': True, 'VISIT': 1, 'DOY': 150, 'YEAR': 2020, 'EXPTIME': 1000, 'UDATE': '2016-05-19 20:48:59'
    }
    fits_object = MagicMock()
    fits_object[0].header = header
    filename = make_filename(fits_object)
    assert filename == 'jup_v1_150_2020_204859_E1000(N,LON,40deg,full,regions,moonfp)'

def test_update_history():
    header = {'HISTORY': 'Initial history'}
    fits_object = MagicMock()
    fits_object[0].header = header
    update_history(fits_object, 'New entry')
    assert 'New entry@' in fits_object[0].header['HISTORY']

def test_fits_from_glob(tmp_path):
    test_file = tmp_path / "test.fits"
    hdu = fits.PrimaryHDU()
    hdu.writeto(test_file)
    fits_list = fits_from_glob(str(tmp_path))
    assert len(fits_list) == 1

def test_jfits():
    header = {'VERSION': 1}
    data = np.zeros((10, 10))
    fits_object = fits.HDUList([fits.PrimaryHDU(data), fits.ImageHDU(data, header=header)])
    jfits = Jfits(fits_obj=fits_object)
    assert np.array_equal(jfits.data, data)
    assert jfits.header['VERSION'] == 1
    jfits.update(data=np.ones((10, 10)))
    assert np.array_equal(jfits.data, np.ones((10, 10)))
    jfits.writeto('/tmp/test.fits')
    jfits.close()