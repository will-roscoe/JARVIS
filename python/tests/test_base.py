import pytest
from jarvis import fpath
import os
class TestFileStructure:
        '''Testing class independent functions
        '''
        def test_nothing(self):
            assert True
        def test_fpath(self):
              assert os.path.exists(fpath('python/tests/test_base.py'))
              assert os.path.exists(fpath('python\\tests\\test_base.py'))

class Test_Image_Generation:
      pass

class Test_GIF_Generation:
    pass