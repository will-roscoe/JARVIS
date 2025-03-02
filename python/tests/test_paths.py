import pytest
from jarvis.utils import fpath,rpath, filename_from_path, ensure_dir
import os

relative_path =  'datasets/HST/v04/jup_16-140-20-48-59_0103_v04_stis_f25srf2_proj.fits'

def test_fpath():
    assert os.path.isfile(fpath(relative_path))
def test_rpath():
    assert os.path.samefile(rpath(fpath(relative_path)), relative_path)
def test_basename():
    assert filename_from_path(relative_path) == 'jup_16-140-20-48-59_0103_v04_stis_f25srf2_proj'
    assert filename_from_path(relative_path, ext=True) == 'jup_16-140-20-48-59_0103_v04_stis_f25srf2_proj.fits'
def test_ensure_dir():
    ensure_dir('tempdir')
    assert os.path.isdir('tempdir')
    os.rmdir('tempdir')
    assert not os.path.isdir('tempdir')
    