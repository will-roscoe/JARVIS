#=== HST_emission_power.py =====================================================
# Script to read in projected/unprojected HST STIS .fits images, define a
# spatial region, and calculate the UV emission power.
#
# Translation between unprojected/projected images is partly handled by IDL
# pipeline legacy code from Boston University, and partly from python code
# provided by Jonny Nichols at Leicester. (broject function)

# Emission power computed as per Gustin+ 2012.

# Should allow e.g., calculation of total auroral oval UV emission power using
# statistical auroral boundaries, planetary auroral comparisons, or
# application to Voronoi image segmentations, etc.

# === Joe Kinrade - 8 January 2025 =============================================
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
#from matplotlib.colors import LogNorm
#import pandas as pd
from dateutil.parser import parse
import datetime as dt
#import matplotlib.patheffects as pe
#from matplotlib.path import Path
import spiceypy as spice
import collections.abc as collections # If using Python version 3.10 and above
#from scipy import signal
#?import scipy.constants as c
from matplotlib import path
from .cvis import pathtest
from .const import KERNELDIR, FITSINDEX
from .utils import fpath, fitsheader, get_datetime
#?from . import time_conversions as tc
def powercalc(fits_obj="", pat=""):
    spice.furnsh(KERNELDIR+'jupiter.mk') # SPICE kernels
    fs = 12   # font size for plots
    ctrs  = pathtest()# dictionary of contour paths 
    print(ctrs)
    #dusk_active_region = [[20,192.25],[30,200],[20,220],[15,230],[15,220]] # high cml, dusk
    #?dusk_active_region = [[20,167.75],[30,160],[20,140],[15,130],[15,140]] # SIII longitudes
    #?swirl_region = [[2.75,111],[17,155],[18,185],[4,190]] # high cml, swirl 360-SIII longitudes here
    #?noon_active_region = [[18,154],[24,154],[28,192],[22,192]] # high cml, noon 360-SIII longitudes here
    dark_region = ctrs
    dark_boundary = path.Path(ctrs) 
    testlons = np.arange(0,360,0.25)
    testcolats = np.arange(0,40,0.25)
    llons, llats = np.meshgrid(testlons, testcolats) # checked correct
    coords = np.vstack([llats.flatten(), llons.flatten()]).T
    dark_mask = dark_boundary.contains_points(coords) # ([[llats], [llons]])
    dark_mask_2d = dark_mask.reshape(160,1440)
    # mask needs to be applied to un-rolled image

    # ------------------------------------------------------------------------------
    # Nichols constants:
    au_to_km = 1.495978707e8

    # Nichols spice stuff:
    planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn',
            'Uranus', 'Neptune', 'Pluto', 'Vulcan']
    inx = planets.index('Jupiter')
    naifobj = 99 + (inx + 1) * 100
    #?frame = 'IAU_' + 'Jupiter'
    radii = spice.bodvcd(naifobj, 'RADII', 3)
    print(radii)
    rpeqkm = radii[1][0]
    rpplkm = radii[1][2]
    oblt = 1. - rpplkm / rpeqkm
    #?obt = oblt

    # Hardwire epoch
    #?epoch = 'J2000'
    #?corr = 'LT'


    # Load fits file and image/header ----------------------------------------------
    proj_filename = fpath(r'datasets\HST\group_04\jup_16-140-20-07-19_0100_v04_stis_f25srf2_proj.fits')


    hdu_list = fits.open(proj_filename)  # opens the FITS files, accessing data plus header info
    hdu_list.info()                    # print file information
    
    # accessing specific header info entries:
    #?exp_time  = hdu_list[0].header['EXPT']
    #ltime     = hdu_list[0].header['LIGHTIME']
    #?aperture  = hdu_list[0].header['FILTER']   # filter type - important as it determines the Gustin conversion factors for intensity/counts/powers etc.
    cml       = hdu_list[0].header['CML']
    #?cml_start = hdu_list[0].header['CML1']     # cml at start of exposure
    #?cml_end   = hdu_list[0].header['CML2']     # cml at end of exposure
    dece      = hdu_list[0].header['DECE']
    #hem       = hdu_list[0].header['HEMISPH']
    #?obt     = hdu_list[0].header['OBLT']       # can't see oblateness in the fits header?
    #?dist_org  = hdu_list[0].header['DIST_ORG']   # Earth-planet distance in AU before reduction
    pcx       = hdu_list[0].header['PCX']        # Planet centre pixel
    pcy       = hdu_list[0].header['PCY']        # Planet centre pixel
    #?nppa_org  = hdu_list[0].header['NPPA_ORG']   # North pole position angle before reduction
    nppa      = hdu_list[0].header['NPPA']       # North pole position angle
    pixsize   = hdu_list[0].header['PXSEC']    # Pixel size in arc seconds 'PIXSIZE' for IDL pprocessed Saturn files
    pxsec     = pixsize                          # just matching a variable name here that's used in the broject function
    dist      = hdu_list[0].header['DIST']       # Standard (scaled) Earth-planet distance in AU
    #?dmeq_orig = hdu_list[0].header['DMEQ_ORG']   # Original diameter of planet equator in arcsecond
    # ------------------------------------------------------------------------------
    cts2kr    = hdu_list[0].header['CTS2KR']     # conversion factor in counts/sec/kR
    print(1./cts2kr)                           # reciprocal of cts2kr to match in Gustin+2012 Table 1.
    #?ltime = dist_org*c.au/c.c
    #?lighttime = dt.timedelta(seconds=ltime)
    # If 1 / conversion factor is ~3994, this implies a colour ratio of 1.10. for Saturn with a STIS SrF2 image (see Gustin+ 2012 Table 1):
    #?colour_ratio = 2.5 # 1.10
    # And this in turn means that the counts-per-second to total emitted power (Watts). conversion factor is 9.04e-10 (Gustin+2012 Table 2), for STIS SrF2:
    gustin_conv_factor =  9.04e-10 # 1.02e-9
    # "Conversion factor to be multiplied by the squared HST-planet distance (km) to determine the total emitted power (Watts) from observed counts per second."
    # ------------------------------------------------------------------------------
    # In some fits files (Jupiter), these 'delta' values for auroral emission altitude are listed as DELRPKM in the header:
    delrpkm = hdu_list[0].header['DELRPKM']    # auroral emission altitudes at homopause in km, a.k.a 'deltas'
    # If not (Saturn), it's hard-wired in here depending on the target planet (probably Saturn!):
    deltas = {'Mars': 0., 'Jupiter': 240., 'Saturn': 1100., 'Uranus': 0.}
    delrpkm = deltas['Jupiter']
    rpkm = rpeqkm            # matching a variable name used in the broject function
    # ------------------------------------------------------------------------------
    # convert HST timestamp to time at Saturn using light travel time:
    start_time = get_datetime(hdu_list)     # create datetime object
    #lighttime = dt.timedelta(seconds=hdu_list[0].header['LIGHTIME'])
    #?exposure = dt.timedelta(seconds=exp_time)
    #?start_time_saturn = start_time - lighttime          # correct for light travel time
    #?end_time_saturn = start_time_saturn + exposure      # end of exposure time
    #?mid_ex_saturn = start_time_saturn + (exposure/2.)   # mid-exposure time at Saturn
    #?mid_ex = start_time + (exposure/2.)                 # mid-exposure time at HST
    # ------------------------------------------------------------------------------
    # def datetime2et(pytimes):
    #     """This function turns python datetime into spice/ET time - From time_conversions.py
    #     Args:
    #         pytimes: datetime 1-D array/list or single value"""
    #     isscalar = False
    #     if not isinstance(pytimes, collections.Iterable):
    #         isscalar = True
    #         pytimes = [pytimes]
    #     utctimes = np.array([dt.strftime(iii, '%Y-%m-%d, %H:%M:%S.%f') for iii in pytimes])
    #     ettimes = np.array([spice.utc2et(iii) for iii in utctimes])
    #     if isscalar:
    #         return np.ndarray.item(ettimes)
    #     else:
    #         return ettimes
    #?et = tc.datetime2et(mid_ex)   # mid-exposure ephemeris time at HST (variable for bproject function)
    # ------------------------------------------------------------------------------
    image_data = hdu_list[1].data   # extract image data from first fits extension (ext=0)


    def _plot_raw():#?
        # make a quick-look plot to check image array content:

        plt.figure()
        plt.title('Image array in fits file.')
        plt.imshow(image_data, cmap='cubehelix',origin='lower', vmin=1, vmax=1000)
        plt.xlabel('longitude pixels')
        plt.ylabel('co-latitude pixels')
        cbar = plt.colorbar(pad=0.05)
        cbar.ax.set_ylabel('Intensity [kR]',fontsize=12)
        plt.show()

    hdu_list.close()                # close file once you're done with it
    # ------------------------------------------------------------------------------
    # perform limb trimming based on angle of surface vector normal to the sun
    lonm = np.radians(np.linspace(0.,360.,num=1440))
    latm = np.radians(np.linspace(-90.,90.,num=720))

    limb_mask = np.zeros((720,1440))   # rows by columns
    cmlr, dec = np.radians(fitsheader(hdu_list, 'CML', 'DECE'))             # convert CML to radians
    #?dec  = np.radians(dece)            # convert declination angle to radians
    for i in range(0,720):
        limb_mask[i,:] = np.sin(latm[i])*np.sin(dec) + np.cos(latm[i])*np.cos(dec)*np.cos(lonm-cmlr)

    limb_mask    = np.flip(limb_mask,axis=1)          # flip the mask horizontally, not sure why this is needed
    cliplim      = np.cos(np.radians(88.))            # set a minimum required vector normal surface-sun angle
    clipind      = np.squeeze([limb_mask >= cliplim]) # False = out of bounds (over the limb)

    image_data[clipind  == False] = np.nan  # set image array values outside clip mask to nans # noqa: E712
    #image_centred = np.roll(image_data,int(cml-180.)*4,axis=1) # centre the image on the cml rather than SIII lon = 180 ##at noon using cml value
    image_centred = image_data.copy() # don't shift by cml for J bc regions defined in longitude not LT

    im_clean = image_centred.copy()   # explicitly make independent image array copy 
    im_clean = np.flip(im_clean,0)    # a centred, clipped version of the full image
    im_4broject = im_clean.copy()
    #im_4broject[im_4broject < -100] = np.nan

    lons = np.arange(0,1440,1)
    lats = np.arange(0,720,1)   # 0-40 degrees colat in image pixel res.

    def _plot_raw_limbtrimmed():#?
        # Quick plot check of the centred, limb-trimmed image:
        plt.figure(figsize=(8,6))
        plt.pcolormesh(lons,lats,np.flip(np.flip(im_clean, axis=1)),cmap='cubehelix',
                    vmin=0.,vmax=1000.)
        plt.title('Image centred, limb-trimmed.')
        plt.xlabel('longitude pixels')
        plt.ylabel('co-latitude pixels')
        plt.show()

    # flip image vertically if required (ease of indexing) and extract auroral region
    # (not strictly required but makes polar projections less confusing):
    if fitsheader(hdu_list,'south'):
        image_extract = image_centred.copy()[0:160,:]
    else:
        im_flip = np.flip(image_centred.copy(),0)
        image_extract = im_flip[0:160,:] # extract image in colat range 0-40 deg (4*40 = 160 pixels in image lat space):


    def _plot_extracted_wcoords():#?
        plt.figure()
        plt.yticks(ticks=[0*4,10*4,20*4,30*4,40*4], labels=['0','10','20','30','40'])
        plt.xticks(ticks=[0*4,90*4,180*4,270*4,360*4], labels=['360','270','180','90','0'])
        #plt.set_yticklabels(['0','10','20','30','40'])   
        plt.imshow(image_extract,cmap='cubehelix',vmin=0., vmax=1000.)
        plt.title('Auroral region extracted')
        plt.xlabel('SIII longitude [deg]')
        plt.ylabel('co-latitude [deg]')
        cbar = plt.colorbar(pad=0.05)
        cbar.ax.set_ylabel('Intensity [kR]',fontsize=12)
        plt.show()

    # ==============================================================================
    # make polar projection plot
    # ==============================================================================
    # set up polar coords
    rho   = np.linspace(0,40,     num=160 ) # colat vector with image pixel resolution steps
    theta = np.linspace(0,2*np.pi,num=1440) # longitude vector in radian space and image pixel resolution steps
    dr = [[i[0] for i in dark_region], [i[1] for i in dark_region]]
    def _plot_polar_wregions():#?
        plt.figure(figsize=(8,6))
        ax = plt.subplot(projection='polar')           # initialize polar projection
        ax.set_title('Polar projection.')
        plt.fill_between(theta, 0, 40, alpha=0.2,hatch="/",color='gray')
        ax.set_theta_zero_location("N")                # set angle 0.0 to top of plot
        ax.set_xticklabels(['0','315','270','225','180','135','90','45'], # reverse these! ###################
                        fontweight='bold',fontsize=fs)
        ax.tick_params(axis='x',pad=-1.)               # shift position of LT labels
        ax.set_yticklabels(['','','','',''])           # turn off auto lat labels
        ax.set_yticks([0,10,20,30,40])                    # but set grid spacing
        ax.set_ylim([0,40])                            # max colat range
        # plot image data in linear colour-scale:
        plt.pcolormesh(theta,rho,image_extract,cmap='cubehelix',
                    vmin=0.,vmax=1000.)
        plt.plot([np.radians(360-r) for r in dr[1]], dr[0], color='red',linewidth=3.)
        # -> OVERPLOT THE ROI IN POLAR PROJECTION HERE. <-
        # Add colourbar: ---------------------------------------------------------------
        cbar = plt.colorbar(ticks=[0.,100.,500.,900.],pad=0.05)
        cbar.ax.set_yticklabels(['0','100','500','900'])
        cbar.ax.set_ylabel('Intensity [kR]',fontsize=12)
        plt.show()

    # # === Plot full image array using pcolormesh so we can understand the masking
    # # process i.e. isolating pixels that fall inside the ROI =======================
    def _plot_sqr_wregions():#?
        plt.figure(figsize=(8,6))
        #plt.yticks(ticks=[0*4,10*4,20*4,30*4,40*4], labels=['0','10','20','30','40'])
        plt.xticks(ticks=[0,90,180,270,360], labels=['360','270','180','90','0'])
        plt.pcolormesh(testlons,testcolats,image_extract,cmap='cubehelix',
                        vmin=0.,vmax=1000.)
        plt.title('ROI')
        plt.xlabel('SIII longitude [deg]')
        plt.ylabel('co-latitude [deg]')
        plt.plot([360-r for r in dr[1]], dr[0],color='red',linewidth=3.)
        plt.show()
    def _plot_roi_mask():#?
        # Quick plot check of the ROI mask:
        plt.figure()
        plt.imshow(dark_mask_2d, origin='lower')
        plt.title('ROI mask in image space')
        plt.xlabel('longitude pixels')
        plt.ylabel('co-latitude pixels')
        plt.show()

    # Now mask off the image by setting image regions where mask=False, to NaNs:
    image_extract[dark_mask_2d==False] = np.nan # noqa: E712
    # And insert auroral region image back into the full projected image space:
    roi_im_full          = np.zeros((720,1440))        # new array full of zeros
    roi_im_full[0:160,:] = image_extract
    # Do the same thing for a full image space ROI mask:
    roi_mask_full          = np.zeros((720,1440))
    roi_mask_full[0:160,:] = dark_mask_2d

    # ==============================================================================
    # Now at the point we can try back-projecting this projected image mask
    # (or masked-out image) using the back-project function:
    # ==============================================================================
    # Nicked from Jonny's pypline.py file:
    def _cylbroject(pimage, cml, dece, dmeq, xcen, ycen, psize, nppa, req, obt, ndiv=2, correct=True):

        if nppa == 999:
            nppa = 0

        ny, nx = pimage.shape
        xsize, ysize = 1400, 1400
        rimage = np.zeros((ysize, xsize))
        cimage = np.zeros((ysize, xsize))

        rad2deg = 180. / np.pi
        deg2rad = np.pi / 180.

    # # # third, calculate the pixel size. 1 au = 1.49598e8 km */
        plen = (psize / 3600.) * deg2rad * dmeq * 1.49598e8 
        #print(plen) # in km

        # # # first initialize global variables */
        sn,cn = np.sin(nppa * deg2rad) , np.cos(nppa * deg2rad)
        sd,cd = np.sin(dece * deg2rad) , np.cos(dece * deg2rad)
        cn = np.cos(nppa * deg2rad)
        sd = np.sin(dece * deg2rad)
        cd = np.cos(dece * deg2rad)
        td = np.tan((dece + 90.0) * deg2rad)
        a = req / plen  # /* equatorial planet radius in  pixel scale */
        o2 = 2.0 * obt
        oo = obt * obt
        a1o = a * (1.0 - obt)
        px = 0
        py = 0
        ii = 0
        jj = 0

        lo = np.zeros(nx * ndiv)
        sb = np.zeros(nx * ndiv)
        cb = np.zeros(nx * ndiv)
        ll = np.zeros(nx * ndiv)
        la = np.zeros(ny * ndiv)
        sa = np.zeros(ny * ndiv)
        ca = np.zeros(ny * ndiv)
        caca = np.zeros(ny * ndiv)
        r = np.zeros(ny * ndiv)

        d = 360.0 / nx / ndiv
        d2 = 360.0 / nx / ndiv / 2.0
        temp = (1.0 - obt) * (1.0 - obt) * td
        for i in range(nx * ndiv):  # / * longitude * /
            lo[i] = i * d + d2 + cml
            sb[i] = np.sin(lo[i] * deg2rad)
            cb[i] = np.cos(lo[i] * deg2rad)
            ll[i] = np.arctan(temp * cb[i]) * rad2deg

        d = 180.0 / ny / ndiv
        d2 = 180.0 / ny / ndiv / 2.0
        for i in range(ny * ndiv):  # / * latitude * /
            la[i] = i * d + d2 - 90.0
            sa[i] = np.sin(la[i] * deg2rad)
            ca[i] = np.cos(la[i] * deg2rad)
            caca[i] = ca[i] * ca[i]
            r[i] = a1o / (1.0 - (o2 - oo) * caca[i])**0.5

    #  / * here's the big loop. start from the longitude corresponding to the left edge of the brojected image.
    #    that is, start from 360-cml+90 and follow down (leftward) on the pimage while the apparent longitude on the
    #    brojected image increases to the right. * /
        for i in range(nx * ndiv):
            if dece < 0.0:
                start = 0
                end = int(((ll[i] + 90.0) / 180.0 * ndiv * ny) + 1)
            else:
                start = int(((ll[i] + 90.0) / 180.0 * ndiv * ny))
                end = ndiv * ny
            for j in range(start, end):
                x = r[j] * ca[j] * sb[i]
                y = r[j] * sa[j]
                z = r[j] * ca[j] * cb[i]
                px = x
                py = y * cd - z * sd
                temp = px
                px = int(px * cn - py * sn + xcen)
                py = int(temp * sn + py * cn + ycen)
                if (px >= 0) & (px < xsize) & (py >= 0) & (py < ysize):
                    ii = int(i / ndiv)
                    jj = int(j / ndiv)
                    value = pimage[jj, ii] / ndiv / ndiv
                    # print(i, j, ii, jj, py, px, value)
                    # return
                    rimage[py, px] += value


    #  /  *  Here, correct for area effect. Added by Juwhan Kim, 03 / 01 / 2005.
        if correct is True:
            value = 1.0 / ndiv / ndiv
            for i in range(nx * ndiv):
                if dece < 0.0:
                    start = 0
                    end = int(((ll[i] + 90.0) / 180.0 * ndiv * ny) + 1)
                else:
                    start = int(((ll[i] + 90.0) / 180.0 * ndiv * ny))
                    end = ndiv * ny
                for j in range(start, end):
                    x = r[j] * ca[j] * sb[i]
                    y = r[j] * sa[j]
                    z = r[j] * ca[j] * cb[i]
                    px = x
                    py = y * cd - z * sd
                    temp = px
                    px = int(px * cn - py * sn + xcen)
                    py = int(temp * sn + py * cn + ycen)
                    if (px >= 0) & (px < xsize) & (py >= 0) & (py < ysize):
                        cimage[py, px] += value

            for i in range(xsize):
                for j in range(ysize):
                    cimval = cimage[j, i]
                    if cimval != 0:
                        rimage[j, i] = rimage[j, i] / cimval

        return rimage

    # this cylbroject function definition is feeding inputs into _cylbroject - JK
    def cylbroject(image, ndiv=2):
        # self._check_image_loaded(proj=True)      # commented out JK
        print('Brojecting with ndiv = ', ndiv)
        bimage = _cylbroject(image,
                                    cml, dece, dist,
                                    pcx, pcy, pxsec,
                                    nppa, rpkm + delrpkm,
                                    oblt, ndiv, True)
        return bimage

    # Backprojecting! Image input needs to be full [1440,720] centred projection
    bimage_roi = cylbroject(np.flip(np.flip(roi_im_full, axis=1)),ndiv=2)   # flips required to get bimage looking right?
    full_image = cylbroject(np.flip(np.flip(im_4broject, axis=1)), ndiv=2)
    # bimage = cylbroject(image_centred,ndiv=2)

    def _plot_brojected_full():#?
        plt.figure()
        plt.title('Brojected full image.')
        plt.imshow(full_image, cmap='cubehelix',origin='lower',vmin=0,vmax=1000)
        plt.xlabel('pixels')
        plt.ylabel('pixels')
        cbar = plt.colorbar(pad=0.05)
        cbar.ax.set_ylabel('Intensity [kR]',fontsize=12)
        fignamei = 'broject_full.pdf'
        plt.savefig(fignamei, dpi=350) #, bbox_inches='tight')  
        plt.show()

    def _plot_brojected_roi():#?
        plt.figure()
        plt.title('Brojected ROI image.')
        plt.imshow(bimage_roi, cmap='cubehelix',origin='lower',vmin=0,vmax=1000)
        plt.xlabel('pixels')
        plt.ylabel('pixels')
        # plt.colorbar()
        cbar = plt.colorbar(pad=0.05)
        cbar.ax.set_ylabel('Intensity [kR]',fontsize=12)
        fignamei = 'broject_roi.pdf'
        plt.savefig(fignamei, dpi=350) #, bbox_inches='tight')  
        plt.show()

    # ==============================================================================
    # Once the back-projected image looks OK, we can proceed with the emission power
    # calculation here.
    # ==============================================================================
    #        ISOLATE THE ROI INTENSITIES IN A FULL 1440*720 PROJECTED IMAGE
    #                      (all other pixels set to nans/zeros)
    distance_squared = (dist * au_to_km)**2          # AU in km
    # calculate emitted power from ROI in GW (exposure time not required here as kR intensities are per second):

    def area_calc(vertices):
        '''Calculates the area in units of degrees^2 of the ROI'''
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        area = abs(area) / 2.0
        return area
    def area_to_km2(area_deg,rad_km=rpkm+240):
        '''Converts the area from degrees^2 to km^2'''
        return area_deg * (np.pi/180.)**2 * rad_km**2
    total_power_emitted_from_roi = np.nansum(bimage_roi) * cts2kr * distance_squared * gustin_conv_factor / 1e9
    area = area_to_km2(area_calc(dark_region))
    power_per_area = total_power_emitted_from_roi / area
    print('Total power emitted from ROI in GW:')
    print(total_power_emitted_from_roi)
    print('Power per unit area in GW/km^2:')
    print(power_per_area)

    visit= hdu_list[0].header['VISIT']
    filepath = 'powers.txt'
    with open(filepath, 'a') as f:
        f.write(visit + ' ' + str(start_time) + ' ' + str(total_power_emitted_from_roi) + ' ' + str(power_per_area) + '\n')
