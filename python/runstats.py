from jarvis.stats import correlate
from Hisaki_Torus_Power import torus_plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from astropy.io import fits
from jarvis import fpath

torus_counter = 0
torus_fitspath = glob.glob(r"C:datasets/Hisaki/Torus Power/*.fits")
torus_groups_list = torus_fitspath[2:42]
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

data1 = pd.read_csv('2025-03-06_17-30-59.txt', sep=' ', header=None)[4]
data2 = power_dawn_all
correlate(data1, data2)