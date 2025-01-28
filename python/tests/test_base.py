import pytest

class TestFileStructure:
        '''Testing class independent functions
        '''
        def test_(self):
            assert True
        

thing = 'something'
var = 'v1'

if var == 'v1':
    print(thing + 'v1.x')
elif var == 'v2':
    print(thing + 'v2.b')


suffix = 'v1.x' if var == 'v1' else 'v2.b'
print(thing + suffix)