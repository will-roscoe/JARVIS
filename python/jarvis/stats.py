import numpy as np
from scipy import stats as st
import scipy.signal as si
import pandas as pd
import matplotlib.pyplot as plt
#does all the stats we will need for analysis
def stats(data,mean=True,median=True,std=True,max=False,min=False):
    '''Calculates common statistcal values for a given dataset.'''
    stats = {}
    if mean:
        stats['mean'] = np.mean(data)
    if median:
        stats['median'] = np.median(data)
    if std:
        stats['std'] = np.std(data)
    if max:
        stats['max'] = np.max(data)
    if min:
        stats['min'] = np.min(data)
    return stats

def correlate(data1, data2):
    '''Calculates the correlation between two datasets.'''
    plt.scatter(data1, data2)
    plt.show()
    correlation = si.correlate(data1, data2, mode='valid')
    print (correlation)
    lags = si.correlation_lags(len(data1), len(data2), mode='valid')
    print (lags)
    regression = st.linregress(data1, data2)
    print (regression)
    return correlation


    
    