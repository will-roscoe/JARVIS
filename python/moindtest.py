from jarvis import moind, fpath
from astropy.io import fits
import itertools
file = 'datasets/HST/v09-may22/jup_16-143-18-41-06_0100_v09_stis_f25srf2_proj.fits'
saveto = 'pictures/testing'


testp_ = ['crop','rlim','fixed','hemis','full','regions']
testvals = [(1,0.7),(30,90), ('lon','lt'),('n',),(True,False),(False,False)] 
test_params = [dict(zip(testp_, x)) for x in itertools.product(*testvals)]

ignore_errors = True
for x in test_params:
    if ignore_errors:
        try:
            moind(file, saveto, **x)
        except Exception as e:
            print(e, x)
    moind(file, saveto, **x)

