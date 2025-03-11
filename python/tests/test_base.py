#!/usr/bin/env python3
import os

import pytest  # noqa: F401
from astropy.io import fits
from jarvis import fpath, make_gif, moind


class TestFileStructure:

    """Testing class independent functions"""

    def test_nothing(self):
        assert True

    def test_fpath(self):
        assert os.path.exists(fpath("python/tests/test_base.py"))
        # assert os.path.exists(fpath('python\\tests\\test_base.py'))

    def test_should_pass(self):
        assert 1 + 2 == 3


file = fits.open("datasets/HST/v01/jup_16-137-23-43-30_0100_v01_stis_f25srf2_proj.fits")
temp = "temp/tests"


class TestImageGen:
    def test_default_img(self):
        assert moind(file) is not None


class TestGIFGeneration:
    def test_gif_gen(self):
        make_gif("datasets/HST/v01/", savelocation=temp, filename="test_norm", dpi=300)
        assert os.path.exists(fpath(f"{temp}/test_norm.gif"))

    def test_gif_moonfp(self):
        make_gif("datasets/HST/v01/", savelocation=temp, filename="test_mfp", dpi=300, moonfp=True)
        assert os.path.exists(fpath(f"{temp}/test_mfp.gif"))
