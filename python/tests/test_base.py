import pytest
from jarvis import fpath, moind, make_gif
import os
import itertools
file = 

class TestFileStructure:
        '''Testing class independent functions
        '''
        def test_nothing(self):
            assert True
        def test_fpath(self):
            assert os.path.exists(fpath('python/tests/test_base.py'))
            #assert os.path.exists(fpath('python\\tests\\test_base.py'))
        def test_should_pass(self):
            assert 1+2 == 3


file = 'datasets/HST/v03/jup_16-139-23-27-06_0100_v05_stis_f25srf2_proj.fits'
temp = 'temp/tests'
class TestImageGen:
    def test_default_img(self):
        moind(file, temp,'test.jpg')
        assert os.path.exists(fpath(f'{temp}/test.jpg'))
    def test_img_param_matrix(self):
        testp_ = ['crop','rlim','fixed','hemis','full','regions']
        testvals = [(1,0.7),(30,90), ('lon','lt'),('n',),(True,False),(False,False)] 
        test_params = [dict(zip(testp_, x)) for x in itertools.product(*testvals)]
        for i,x in enumerate(test_params):
            moind(file, temp, f'matrixtest{i}.jpg', **x)
            assert os.path.exists(fpath(f'{temp}/matrixtest{i}.jpg'))

class Test_GIF_Generation:
    def test_gif_gen(self):
        make_gif('datasets\\HST\\v01\\',savelocation=temp,filename='test', dpi=300)
        assert os.path.exists(fpath(f'{temp}/test.gif'))