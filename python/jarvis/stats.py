#!/usr/bin/env python3
"""This module contains functions for statistical analysis of data."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as si
from scipy import stats as st


# does all the stats we will need for analysis
def stats(data, mean=True, median=True, std=True, vmax=False, vmin=False):
    """Calculate common statistcal values for a given dataset."""
    stats = {}
    if mean:
        stats["mean"] = np.mean(data)
    if median:
        stats["median"] = np.median(data)
    if std:
        stats["std"] = np.std(data)
    if vmax:
        stats["max"] = np.max(data)
    if vmin:
        stats["min"] = np.min(data)
    return stats


def correlate(data1, data2):
    """Calculate the correlation between two datasets."""
    correlation = si.correlate(data1, data2, mode="full")
    correlation /= np.max(correlation)
    print(correlation)
    lags = si.correlation_lags(len(data1), len(data2), mode="full")
    print(lags)
    plt.plot(lags, correlation)
    plt.xlabel("Lags")
    plt.ylabel("Correlation")
    plt.title("Correlation between data sets")
    plt.show()
    if len(data1) == len(data2):
        pearson = st.pearsonr(data1, data2)
        print(pearson)
        return pearson
    print("Data sets are not the same length. Data set 1 length: ", len(data1), "Data set 2 length: ", len(data2))
    return None



# Not sure but log or gamma (with θ/α >2) distribution may match data? ~~Will
def get_statstext(
    hist, mode, symb, skewed=False,
):  # Function i pulled out of another project with skewed distribution analysis. kurtosis only works if there are two sides of the distribution though, not maximum at one of the extrema.
    """Return a string with the mode, average, standard deviation, kurtosis and skewness of a histogram.

    Args:
    hist: A histogram tuple. (generate from np.histogram)
    mode: The mode of the histogram.
    symb: symbol representing the quantity.
    skewed: If True, returns kurtosis and skewness.

    """
    freqs, bins = hist[:2]
    mids = 0.5 * (bins[1:] + bins[:-1]).flatten()
    x = [mids[i] for i in range(len(mids)) for _ in range(int(freqs[i]))]
    if skewed:
        return ",\n".join(
            [
                f"Mo[{symb}]: {mode:.1f}",
                f"Avg[{symb}]: {np.mean(x):.1f}",
                f"$\\kappa$: {st.kurtosis(x):.1f}",
                f"$\\gamma$: {st.skew(x):.1f}",
            ],
        )
    return ",\n".join(
        [f"Mo[{symb}]: {mode:.1f}", f"Avg[{symb}]: {np.mean(x)::.1f}", f"$\\sigma$: {std_dev(hist)::.1f}"],
    )


def std_dev(hist):
    """Return the standard deviation of a histogram (normally distributed)."""
    counts, bins = hist[:2]
    mids = 0.5 * (bins[1:] + bins[:-1])
    probs = counts / np.sum(counts)
    mean = np.sum(probs * mids)
    return np.sqrt(np.sum(probs * (mids - mean) ** 2))


def intensity_distribution(roidata, bins=100, irange=None):
    """Return the intensity distribution of a given roidata."""
    return np.histogram(roidata, bins=bins, range=irange)

