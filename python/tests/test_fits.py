#!/usr/bin/env python3
import numpy as np
import pytest  # noqa: F401
from astropy.io import fits
from jarvis.utils import (
    adapted_hdul,
    assign_params,
    debug_fitsdata,
    debug_fitsheader,
    filename_from_hdul,
    fitsheader,
    fpath,
    get_datetime,
    update_history,
)

fits_obj = fits.open(fpath(r"datasets\HST\v04\jup_16-140-20-48-59_0103_v04_stis_f25srf2_proj.fits"))

prepared_fits = assign_params(fits_obj, regions=False, moonfp=False, fixed="lon", rlim=40, full=True, crop=1)


def test_fits_from_parent():
    new_fits = adapted_hdul(fits_obj, new_data=np.ones_like(fits_obj[1].data))
    assert new_fits[1].data.shape == fits_obj[1].data.shape


def test_prepare_fits():
    new_fits = assign_params(fits_obj, regions=False, moonfp=False, fixed="lon", rlim=40, full=True, crop=1)
    assert new_fits[1].header["FIXED"] == "LON"
    assert new_fits[1].header["RLIM"] == 40
    assert new_fits[1].header["FULL"]
    assert new_fits[1].header["CROP"] == 1
    assert not new_fits[1].header["REGIONS"]
    assert not new_fits[1].header["MOONFP"]


def test_fitsheader():
    assert fitsheader(prepared_fits, "rlim") == 40
    assert fitsheader(prepared_fits, "full")
    assert fitsheader(prepared_fits, "crop") == 1
    assert not fitsheader(prepared_fits, "regions")
    assert not fitsheader(prepared_fits, "moonfp")
    assert not fitsheader(prepared_fits, "south")
    assert fitsheader(prepared_fits, "fixed_lon")


def test_get_datetime():
    assert get_datetime(fits_obj).strftime("%d-%m-%y %H:%M:%S") == "19-05-16 20:48:59"


def test_make_filename():
    expect = "jup_v04_140_2016_204859_0103"
    assert filename_from_hdul(prepared_fits)[0 : len(expect)] == expect


def test_update_history():
    assert update_history(prepared_fits, "test1", "test2")[0:13] == "test1@test2@"


def test_debug_fitsheader(capsys):
    debug_fitsheader(prepared_fits)
    captured = capsys.readouterr()
    assert "FIXED: LON" in captured.out


def test_debug_fitsdata(capsys):
    debug_fitsdata(prepared_fits)
    captured = capsys.readouterr()
    assert "(720, 1440)" in captured.out
