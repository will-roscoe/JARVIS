#!/usr/bin/env python3
"""Created on Tue Jan 25 11:15:23 2022.

This code uses both the Hess 2011 table of ephemeris and the longitude of
each Jovian moon (read in the header of every fits file) to output the
expected latitude and longitude of the four Galilean moons.

@author: diegomp
"""

import numpy as np

from .utils import fpath


def moonfploc(iolon, eulon, galon):
    """Use both the Hess 2011 table of ephemeris and the longitude of each Jovian moon (read in the header of every fits file) to output the expected latitude and longitude of the four Galilean moons."""
    tablename = fpath("datasets/hess2011_table1.txt")
    lista = []
    with open(tablename) as table:
        for line in table:
            element = line.split()
            elemen = [float(o) for o in element]
            lista.append(elemen)

    lista = [x for x in lista if x != []]

    lon = np.arange(36)
    satlon = [iolon, eulon, galon]

    for n, i in enumerate(satlon):
        for lo in lon:
            if i > lista[lo][0]:
                pre = lo
                pos = lo + 1
                if lo == lon[-1]:
                    pos = 0
                sat_i = lista[pre][0]
                sat_f = lista[pos][0]

                if n == 0:  # io
                    nlonio = np.interp(i, [sat_i, sat_f], [lista[pre][1], lista[pos][1]])
                    nlatio = np.interp(i, [sat_i, sat_f], [lista[pre][2], lista[pos][2]])
                    slonio = np.interp(i, [sat_i, sat_f], [lista[pre][3], lista[pos][3]])
                    slatio = np.interp(i, [sat_i, sat_f], [lista[pre][4], lista[pos][4]])

                    ncolatio = 90.0 - abs(nlatio)
                    scolatio = 90.0 - abs(slatio)

                elif n == 1:  # europa
                    nloneu = np.interp(i, [sat_i, sat_f], [lista[pre][5], lista[pos][5]])
                    nlateu = np.interp(i, [sat_i, sat_f], [lista[pre][6], lista[pos][6]])
                    sloneu = np.interp(i, [sat_i, sat_f], [lista[pre][7], lista[pos][7]])
                    slateu = np.interp(i, [sat_i, sat_f], [lista[pre][8], lista[pos][8]])

                    if lista[pos][5] == 0.0:
                        nloneu = lista[pre][5]
                        nlateu = lista[pre][6]
                    elif lista[pre][5] == 0.0:
                        nloneu = lista[pos][5]
                        nlateu = lista[pos][6]

                    ncolateu = 90.0 - abs(nlateu)
                    scolateu = 90.0 - abs(slateu)

                elif n == 2:  # ganymede
                    nlonga = np.interp(i, [sat_i, sat_f], [lista[pre][9], lista[pos][9]])
                    nlatga = np.interp(i, [sat_i, sat_f], [lista[pre][10], lista[pos][10]])
                    slonga = np.interp(i, [sat_i, sat_f], [lista[pre][11], lista[pos][11]])
                    slatga = np.interp(i, [sat_i, sat_f], [lista[pre][12], lista[pos][12]])

                    if lista[pos][9] == 0.0:
                        nlonga = lista[pre][9]
                        nlatga = lista[pre][10]
                    elif lista[pre][9] == 0.0:
                        nlonga = lista[pos][9]
                        nlatga = lista[pos][10]

                    ncolatga = 90.0 - abs(nlatga)
                    scolatga = 90.0 - abs(slatga)

    return nlonio, ncolatio, slonio, scolatio, nloneu, ncolateu, sloneu, scolateu, nlonga, ncolatga, slonga, scolatga


# np.savetxt('/home/diego/Pictures/finding_MO/gaussfit/averageoval_allv_median_filt2.5.txt', mean_oval)
