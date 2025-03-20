import glob

import numpy as np
import pandas as pd
from astropy.io import fits
from jarvis import fpath
from jarvis.stats import correlate
from scipy import special

torus_counter = 0
torus_fitspath = glob.glob(r"C:datasets/Hisaki/Torus Power/*.fits")
torus_groups_list = torus_fitspath[12:-30]
power_dawn_all = np.array([])
power_dusk_all = np.array([])
torus_time_all = np.array([])

for l in torus_groups_list:
    hdul = fits.open(fpath(l))
    time_series_data = hdul[2].data
    torus_time = time_series_data.field('SECOFDAY') + 86400 * torus_counter
    if l == r'C:datasets/Hisaki/Torus Power\exeuv_torus_20160525_lv03_LT00-24_dt00010_vr01_00.fits':
        torus_counter += 2
    torus_counter += 1
    power_dawn = time_series_data.field('TPOW0710ADAWN')
    power_dusk = time_series_data.field('TPOW0710ADUSK')
    for m in power_dawn:
        power_dawn_all = np.append(power_dawn_all, m)
    for n in power_dusk:
        power_dusk_all = np.append(power_dusk_all, n)
    for o in torus_time:
        torus_time_all = np.append(torus_time_all, o)

power_dawn_average = []
for i in range(0, len(power_dawn_all), 20):
    power_dawn_average.append(np.mean(power_dawn_all[i:i+20]))
time_average = [torus_time_all[i] for i in range(0, len(torus_time_all), 20)]



data1 = pd.read_csv('2025-03-06_17-30-59.txt', sep=' ', header=None)
average_dpr_power=[]
visitnum = np.array(data1[0])
sum = 0
visitloops = 1
for i in range(len(visitnum)):
    if visitnum[i] == visitnum[i-1]:
        sum += data1[4][i]
        visitloops += 1
    else:
        average_dpr_power.append(sum/visitloops)
        sum = 0
        visitloops = 0
date = data1[1]
data1 = average_dpr_power
data2 = power_dawn_average[0:len(data1)]


def get_sigma(p_value):
    # the special.erfc(x) is the complementary error function, where x = sigma/sqrt(2)
    # i.e. the table of p-value-to-sigma at
    # https://en.wikipedia.org/wiki/Normal_distribution#Standard_deviation_and_tolerance_intervals
    # is actually a table of erf(x), erfc(x), 1./erfc(x) for sigma values from 1 to 6
    # and special.erfcinv(alpha)*np.sqrt(2.) will return the significance level in sigma.
    #
    # however, WARNING - this sigma is only valid assuming the distributions are Normal
    # which a lot of this utils file assumes they aren't.
    # so use as guidance but not gospel, and very much with caution.
    return special.erfcinv(p_value)*np.sqrt(2.)

print('sigma:', get_sigma(correlate(data1, data2)[1]))
