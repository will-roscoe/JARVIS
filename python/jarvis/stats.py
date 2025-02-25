import numpy as np
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


    