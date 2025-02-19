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
from jarvis import fpath
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
spice.furnsh('python/jupiter.mk') # SPICE kernels
import time_conversions as tc
import collections
import collections.abc as collections # If using Python version 3.10 and above
#from scipy import signal
import scipy.constants as c
from matplotlib import path
fs = 12   # font size for plots
ctrs  = pathtest()# dictionary of contour paths 
print(ctrs)
#dusk_active_region = [[20,192.25],[30,200],[20,220],[15,230],[15,220]] # high cml, dusk
dusk_active_region = [[20,167.75],[30,160],[20,140],[15,130],[15,140]] # SIII longitudes
swirl_region = [[2.75,111],[17,155],[18,185],[4,190]] # high cml, swirl 360-SIII longitudes here
noon_active_region = [[18,154],[24,154],[28,192],[22,192]] # high cml, noon 360-SIII longitudes here
dark_region = ([8,247.5], [12,247.5], [18,225], [22,217.5], [22,202.5], [20,202.5], [14,202.5], [8,217.5])
#dark_boundary = path.Path([(8,247.5), (12,247.5), (18,225), (22,217.5), (22,202.5), (20,202.5), (14,202.5), (8,217.5)]) 
dark_region = [[30,172.5],[32,172.5],[32,180], [29.75,191.25],[26.5,202.5],[20,225],[16,225],[27,190],[28,180]]
dark_boundary =path.Path([(28,180),(27,190),(16,225),(20,225),(26.5,202.5),(29.75,191.25),(32,180),(32,172.5),(30,172.5)])
dark_region = ctrs
dark_boundary = path.Path(ctrs) 

#dar_boundary_2 = path.Path([(),(),(),()])
#dar_boundary = path.Path([(dusk_active_region[0][0],360-dusk_active_region[0][1]), 
                   #       (dusk_active_region[1][0],360-dusk_active_region[1][1]),
                    #      (dusk_active_region[2][0],360-dusk_active_region[2][1]),
                     #     (dusk_active_region[3][0],360-dusk_active_region[3][1]),
                      #    (dusk_active_region[4][0],360-dusk_active_region[4][1]),
                       #   (dusk_active_region[0][0],360-dusk_active_region[0][1]),]) 
testlons = np.arange(0,360,0.25)
testcolats = np.arange(0,40,0.25)
llons, llats = np.meshgrid(testlons, testcolats) # checked correct
coords = np.vstack([llats.flatten(), llons.flatten()]).T
dark_mask = dark_boundary.contains_points(coords) # ([[llats], [llons]])
dark_mask_2D = dark_mask.reshape(160,1440)
# mask needs to be applied to un-rolled image

# ------------------------------------------------------------------------------
# Nichols constants:
au_to_km = 1.495978707e8

# Nichols spice stuff:
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn',
           'Uranus', 'Neptune', 'Pluto', 'Vulcan']
inx = planets.index('Jupiter')
naifobj = 99 + (inx + 1) * 100
frame = 'IAU_' + 'Jupiter'
radii = spice.bodvcd(naifobj, 'RADII', 3)
print(radii)
rpeqkm = radii[1][0]
rpplkm = radii[1][2]
oblt = 1. - rpplkm / rpeqkm
obt = oblt

# Hardwire epoch
epoch = 'J2000'
corr = 'LT'


# Load fits file and image/header ----------------------------------------------
# path to a projected image:
# proj_filename = '/Users/joe/data/HST/HST2016_revised/long_exposure/sat_16-232-20-43-06_stis_f25srf2_sm_proj_nobg.fits'
# proj_filename = '/Users/joe/data/HST/HST2017_revised/long_exposure/sat_17-045-04-18-34_stis_f25srf2_sm_proj_nobg.fits'
# proj_filename = '/Users/joe/data/HST/HST2017_revised/long_exposure/sat_17-066-16-52-02_stis_f25srf2_sm_proj_nobg.fits'
#proj_filename = '/Users/Sarah/OneDrive - Lancaster University/Prog/Python/TestData/137_v01/jup_16-137-23-43-30_0100_v01_stis_f25srf2_proj.fits'

proj_filename = fpath(r'datasets\HST\v04\jup_16-140-20-07-19_0100_v04_stis_f25srf2_proj.fits')


hdu_list = fits.open(proj_filename)  # opens the FITS files, accessing data plus header info
hdu_list.info()                    # print file information

# accessing specific header info entries:
exp_time  = hdu_list[0].header['EXPT']
#ltime     = hdu_list[0].header['LIGHTIME']
aperture  = hdu_list[0].header['FILTER']   # filter type - important as it determines the Gustin conversion factors for intensity/counts/powers etc.
cml       = hdu_list[0].header['CML']
cml_start = hdu_list[0].header['CML1']     # cml at start of exposure
cml_end   = hdu_list[0].header['CML2']     # cml at end of exposure
dece      = hdu_list[0].header['DECE']
hem       = hdu_list[0].header['HEMISPH']
obt     = hdu_list[0].header['OBLT']       # can't see oblateness in the fits header?
dist_org  = hdu_list[0].header['DIST_ORG']   # Earth-planet distance in AU before reduction
pcx       = hdu_list[0].header['PCX']        # Planet centre pixel
pcy       = hdu_list[0].header['PCY']        # Planet centre pixel
nppa_org  = hdu_list[0].header['NPPA_ORG']   # North pole position angle before reduction
nppa      = hdu_list[0].header['NPPA']       # North pole position angle
pixsize   = hdu_list[0].header['PXSEC']    # Pixel size in arc seconds 'PIXSIZE' for IDL pprocessed Saturn files
pxsec     = pixsize                          # just matching a variable name here that's used in the broject function
dist      = hdu_list[0].header['DIST']       # Standard (scaled) Earth-planet distance in AU
dmeq_orig = hdu_list[0].header['DMEQ_ORG']   # Original diameter of planet equator in arcsecond
# ------------------------------------------------------------------------------
cts2kr    = hdu_list[0].header['CTS2KR']     # conversion factor in counts/sec/kR
print(1./cts2kr)                           # reciprocal of cts2kr to match in Gustin+2012 Table 1.

ltime = dist_org*c.au/c.c
lighttime = dt.timedelta(seconds=ltime)

# If 1 / conversion factor is ~3994, this implies a colour ratio of 1.10
# for Saturn with a STIS SrF2 image (see Gustin+ 2012 Table 1):
colour_ratio = 2.5 # 1.10

# And this in turn means that the counts-per-second to total emitted power (Watts)
# conversion factor is 9.04e-10 (Gustin+2012 Table 2), for STIS SrF2:
gustin_conv_factor =  9.04e-10 # 1.02e-9
# "Conversion factor to be multiplied by the squared HST-planet distance (km)
# to determine the total emitted power (Watts) from observed counts per second."

# ------------------------------------------------------------------------------
# In some fits files (Jupiter), these 'delta' values for auroral emission
#altitude are listed as DELRPKM in the header:
delrpkm = hdu_list[0].header['DELRPKM']    # auroral emission altitudes at homopause in km, a.k.a 'deltas'

# If not (Saturn), it's hard-wired in here depending on the target planet (probably Saturn!):
deltas = {'Mars': 0., 'Jupiter': 240., 'Saturn': 1100., 'Uranus': 0.}
delrpkm = deltas['Jupiter']
rpkm = rpeqkm            # matching a variable name used in the broject function

# ------------------------------------------------------------------------------
# convert HST timestamp to time at Saturn using light travel time:
start_time = parse(hdu_list[0].header['UDATE'])     # create datetime object
#lighttime = dt.timedelta(seconds=hdu_list[0].header['LIGHTIME'])
exposure = dt.timedelta(seconds=exp_time)
start_time_saturn = start_time - lighttime          # correct for light travel time
end_time_saturn = start_time_saturn + exposure      # end of exposure time
mid_ex_saturn = start_time_saturn + (exposure/2.)   # mid-exposure time at Saturn
mid_ex = start_time + (exposure/2.)                 # mid-exposure time at HST

# ------------------------------------------------------------------------------
# This function turns python datetime into spice/ET time - From time_conversions.py
# Input: datetime 1-D array/list or single value
def datetime2et(pytimes):
    isscalar = False
    if not isinstance(pytimes, collections.Iterable):
        isscalar = True
        pytimes = [pytimes]
    utctimes = np.array([dt.strftime(iii, '%Y-%m-%d, %H:%M:%S.%f') for iii in pytimes])
    ettimes = np.array([spice.utc2et(iii) for iii in utctimes])
    if isscalar:
        return np.ndarray.item(ettimes)
    else:
        return ettimes

et = tc.datetime2et(mid_ex)   # mid-exposure ephemeris time at HST (variable for bproject function)

# ------------------------------------------------------------------------------
image_data = hdu_list[1].data   # extract image data from first fits extension (ext=0)

# make a quick-look plot to check image array content:
plt.figure()
plt.title('Image array in fits file.')
plt.imshow(image_data, cmap='cubehelix',origin='lower', vmin=1, vmax=1000)
plt.xlabel('longitude pixels')
plt.ylabel('co-latitude pixels')

# plt.colorbar()
cbar = plt.colorbar(pad=0.05)
cbar.ax.set_ylabel('Intensity [kR]',fontsize=12)
plt.show()
hdu_list.close()                # close file once you're done with it

# ------------------------------------------------------------------------------
# perform limb trimming based on angle of surface vector normal to the sun
lonm = np.radians(np.linspace(0.,360.,num=1440))
latm = np.radians(np.linspace(-90.,90.,num=720))

limb_mask = np.zeros((720,1440))   # rows by columns
cmlr = np.radians(cml)             # convert CML to radians
dec  = np.radians(dece)            # convert declination angle to radians
for i in range(0,720):
    limb_mask[i,:] = np.sin(latm[i])*np.sin(dec) + np.cos(latm[i])*np.cos(dec)*np.cos(lonm-cmlr)

limb_mask    = np.flip(limb_mask,axis=1)          # flip the mask horizontally, not sure why this is needed
cliplim      = np.cos(np.radians(88.))            # set a minimum required vector normal surface-sun angle
clipind      = np.squeeze([limb_mask >= cliplim]) # False = out of bounds (over the limb)

image_data[clipind  == False] = np.nan # set image array values outside clip mask to nans
#image_centred = np.roll(image_data,int(cml-180.)*4,axis=1) # centre the image on the cml rather than SIII lon = 180 ##at noon using cml value
image_centred = image_data.copy() # don't shift by cml for J bc regions defined in longitude not LT

im_clean = image_centred.copy()   # explicitly make independent image array copy 
im_clean = np.flip(im_clean,0)    # a centred, clipped version of the full image
im_4broject = im_clean.copy()
#im_4broject[im_4broject < -100] = np.nan

lons = np.arange(0,1440,1)
lats = np.arange(0,720,1)   # 0-40 degrees colat in image pixel res.

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
if  hem == 'north':
    im_flip = np.flip(image_centred.copy(),0)
    image_extract = im_flip[0:160,:] # extract image in colat range 0-40 deg (4*40 = 160 pixels in image lat space):
elif hem == 'south':
    image_extract = image_centred.copy()[0:160,:]

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

# plot image data in log-colour scale:
# plt.pcolormesh(theta,rho,image_extract,cmap='cubehelix',
#                norm=LogNorm(vmin=.1,vmax=100.))

# plot image data in linear colour-scale:
plt.pcolormesh(theta,rho,image_extract,cmap='cubehelix',
               vmin=0.,vmax=1000.)

dr = [[i[0] for i in dark_region], [i[1] for i in dark_region]]
plt.plot([np.radians(360-r) for r in dr[1]], dr[0], color='red',linewidth=3.)
    
    # [np.radians(360-dark_region[0][1]),np.radians(360-dark_region[1][1]),
    #       np.radians(360-dark_region[2][1]),np.radians(360-dark_region[3][1]),
    #       np.radians(360-dark_region[4][1]),np.radians(360-dark_region[5][1]),
    #       np.radians(360-dark_region[6][1]),np.radians(360-dark_region[7][1]),
    #       np.radians(360-dark_region[0][1])],  # corner A repeated to
    #       [dark_region[0][0],dark_region[1][0],dark_region[2][0],
    #        dark_region[3][0], dark_region[4][0], dark_region[5][0],
    #        dark_region[6][0], dark_region[7][0], 
    #        dark_region[0][0]],  # close the box.
    #       color='red',linewidth=3.)
#
# -> OVERPLOT THE ROI IN POLAR PROJECTION HERE. <-

# Add colourbar: ---------------------------------------------------------------
cbar = plt.colorbar(ticks=[0.,100.,500.,900.],pad=0.05)
cbar.ax.set_yticklabels(['0','100','500','900'])
cbar.ax.set_ylabel('Intensity [kR]',fontsize=12)

plt.show()

# ==============================================================================
# Adding a shape/contour to the plot within which to calculate emission power
# e.g. an auroral oval, or a partial oval. But a simple box to start with.
# ==============================================================================

# Simple box shape - define lat/lon corners of a region of interest (roi) first.
# e.g. a box spanning 10-20 degrees colat, 2 hrs wide centred on noon:
# [lon, lat] in image res. space:

###### SVB cut out when making polygon #####
#colat_min = 10 ; colat_max = 20
#lt_min = 11    ; lt_max = 13
#lon_min = 160    ; lon_max = 180

# translate these colat/LT ranges into image space values used below:
# colat_min_imspace = colat_min*4  # 720 pixels / 180 deg lat =  4 scaling factor
# colat_max_imspace = colat_max*4
# lt_min_imspace    = 1440-(lon_min*4)    # 1440 pixels / 24 hours   = 60 scaling factor
# lt_max_imspace    = 1440-(lon_max*4)
# #                                              equatorward
# roi_a = [lt_min_imspace,colat_min_imspace]  # B ---------- C
# roi_b = [lt_min_imspace,colat_max_imspace]  # |            |
# roi_c = [lt_max_imspace,colat_max_imspace]  # |    ROI     |
# roi_d = [lt_max_imspace,colat_min_imspace]  # |            |
#                                             # A ---------- D
# #                                                poleward
# # n.b. 10 degrees colat is row index 40 in image space
# # and column index 720 is image centre at noon.

# # === Plot full image array using pcolormesh so we can understand the masking
# # process i.e. isolating pixels that fall inside the ROI =======================

# # print(np.array(image_extract).shape)  # [160,1440]
# lons = np.arange(0,1440,1)
# lats = np.arange(0,160,1)   # 0-40 degrees colat in image pixel res.

plt.figure(figsize=(8,6))
#plt.yticks(ticks=[0*4,10*4,20*4,30*4,40*4], labels=['0','10','20','30','40'])
plt.xticks(ticks=[0,90,180,270,360], labels=['360','270','180','90','0'])
plt.pcolormesh(testlons,testcolats,image_extract,cmap='cubehelix',
                vmin=0.,vmax=1000.)
plt.title('ROI')
plt.xlabel('SIII longitude [deg]')
plt.ylabel('co-latitude [deg]')

# Overplot the region of interest, e.g. a lat-lon box here:
#    dusk_active_region = [[20,192.25],[30,200],[20,220],[15,230],[15,220]]
#    dar_boundary = path.Path([(20,192.25), (30,200), (20,220), (15,230), (15,220), (20,192.25)]) 


plt.plot([360-r for r in dr[1]], dr[0], 
    # [360-dark_region[0][1],360-dark_region[1][1],
    #       360-dark_region[2][1],360-dark_region[3][1],
    #       360-dark_region[4][1],360-dark_region[5][1],
    #       360-dark_region[6][1],360-dark_region[7][1],
    #       360-dark_region[0][1]],  # corner A repeated to
    #       [dark_region[0][0],dark_region[1][0],dark_region[2][0],
    #        dark_region[3][0],dark_region[4][0],dark_region[5][0],
    #        dark_region[6][0],dark_region[7][0],
    #        dark_region[0][0]],  # close the box. color='red',linewidth=3.)

# plt.plot([roi_a[0],roi_b[0],roi_c[0],roi_d[0],roi_a[0]],  # corner A repeated to
#           [roi_a[1],roi_b[1],roi_c[1],roi_d[1],roi_a[1]],  # close the box.
           color='red',linewidth=3.)
plt.show()

# # Now use the roi to mask the projected image, and plot:
# roi_mask = np.zeros(image_extract.shape,dtype=bool) # initialise a mask array, same size as the auroral image
# # roi = np.array([roi_a,roi_b,roi_c,roi_d])

# #roi_bottom_left = (roi_a[1],roi_a[0])    # slicing a box region requires two opposite corners
# #roi_top_right = (roi_c[1],roi_c[0])
# roi_bottom_left = (roi_d[1],roi_d[0])    # slicing a box region requires two opposite corners
# roi_top_right = (roi_b[1],roi_b[0])

# # use slicing to set the mask to True inside the box (+1 is required for full range of corners to be included):
# roi_mask[roi_bottom_left[0]:roi_top_right[0]+1, roi_bottom_left[1]:roi_top_right[1]+1] = True

# Quick plot check of the ROI mask:
plt.figure()
plt.imshow(dark_mask_2D, origin='lower')
plt.title('ROI mask in image space')
plt.xlabel('longitude pixels')
plt.ylabel('co-latitude pixels')

plt.show()

# Now mask off the image by setting image regions where mask=False, to NaNs:
image_extract[dark_mask_2D==False] = np.nan
# Quick plot check of the masked off image:
# plt.figure(figsize=(8,6))
# plt.pcolormesh(lons,lats,image_extract,cmap='cubehelix',
#                vmin=0.,vmax=30.)
# plt.title('Image masked off by the ROI')
# plt.show()

# And insert auroral region image back into the full projected image space:
roi_im_full          = np.zeros((720,1440))        # new array full of zeros
roi_im_full[0:160,:] = image_extract
# Quick plot check to see auroral region inserted properly:
# plt.figure()
# plt.imshow(roi_im_full, origin='lower')
# plt.title('ROI intensities in full image space')
# plt.show()

# Do the same thing for a full image space ROI mask:
roi_mask_full          = np.zeros((720,1440))
roi_mask_full[0:160,:] = dark_mask_2D

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
    sn = np.sin(nppa * deg2rad)
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

# set CML to 180 for broject function input, as we've already centred the HST image by CML.
#cml = 180

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

plt.figure()
plt.title('Brojected full image.')
plt.imshow(full_image, cmap='cubehelix',origin='lower',vmin=0,vmax=1000)
plt.xlabel('pixels')
plt.ylabel('pixels')
cbar = plt.colorbar(pad=0.05)
cbar.ax.set_ylabel('Intensity [kR]',fontsize=12)

# plt.colorbar()
# cbar = plt.colorbar(pad=0.05)
# cbar.ax.set_ylabel('Intensity [kR]',fontsize=12)
fignamei = 'broject_full.pdf'
plt.savefig(fignamei, dpi=350) #, bbox_inches='tight')  

plt.show()

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
#
#        ISOLATE THE ROI INTENSITIES IN A FULL 1440*720 PROJECTED IMAGE
#                      (all other pixels set to nans/zeros)
#
distance_squared = (dist * au_to_km)**2          # AU in km
#
# calculate emitted power from ROI in GW (exposure time not required here as kR intensities are per second):
total_power_emitted_from_roi = np.nansum(bimage_roi) * cts2kr * distance_squared * gustin_conv_factor / 1e9

print('Total power emitted from ROI in GW:')
print(total_power_emitted_from_roi)


# ==============================================================================
# This next bit is to make a logic mask for an auroral oval polygon, via
# image wrapping and interpolation at full-image resolution to get it working
# properly.
# ==============================================================================

# # Read in statistical auroral boundaries from Bader+ 2019: ---------------------
# filename03 = 'Bader2019_saturn_boundaries.txt'
# # make header names for columns (since they're not in the file)
# # az=azimuth,ut=UT,n_p=north poleward boundary,n_uc=north centre uncertainy etc.
# names = ['az','ut','n_p','n_up','n_c','n_uc','n_e','n_ue','s_p','s_up','s_c',
#          's_uc','s_e','s_ue']
# col_index = [0,1,3,5,7,9,11,13,15,17,19,21,23,25]    # column indices to skip &s
# data_auroras = pd.read_csv(filename03,sep='[\s+]',usecols=col_index,names=names,
#                             engine='python')
# dfa = pd.DataFrame(data_auroras)     # make pandas dataframe
# dfa.s_ue = dfa.s_ue.str.strip('\\')  # remove \\ from the end of the last column
# dfa.ut   = dfa.ut.str.strip('()')    # remove brackets from UT column

# # First let's wrap the image in longitude a few times so we can deal with the
# # midnight crossing point of the auroral region boundary:
# im_wrap = np.append(im_clean,im_clean,axis=1)
# im_wrap = np.append(im_wrap,im_clean,axis=1)
# # print(np.array(im_wrap).shape)  # [720,4320]

# # Now we have three copies of the image array alongside eachother [720,4320]
# # We'll replicate the auroral boundaries across this extended longitude space,
# # interpolate to get a value at each pixel step, and then crop everything back
# # again using the 'middle' image of the wrapped array:

# # plt.figure()
# # plt.title('Image array wrapped in longitude.')
# # plt.imshow(im_wrap, cmap='cubehelix',origin='lower')
# # plt.show()

# azis     = np.array(dfa.az)
# colats_e = np.array(dfa.n_e)
# colats_p = np.array(dfa.n_p)

# azis_wrap = np.append(azis,azis+360)                  # wrapping azimuths
# azis_wrap = np.append(azis_wrap,azis+720)
# azis_wrap_scaled = np.multiply(azis_wrap,4)           # scale the azimuth values to image space

# colats_e_wrap = np.append(colats_e,colats_e)          # wrapping equatorial colats
# colats_e_wrap = np.append(colats_e_wrap,colats_e)
# colats_e_wrap_scaled = np.multiply(colats_e_wrap,4)   # scale the colat values to image space

# colats_p_wrap = np.append(colats_p,colats_p)          # wrapping poleward colats
# colats_p_wrap = np.append(colats_p_wrap,colats_p)
# colats_p_wrap_scaled = np.multiply(colats_p_wrap,4)   # scale the colat values to image space

# lons = np.arange(0,4320,1)    # lat-lons covering wrapped image space
# lats = np.arange(0,720,1)

# azis_im_res = np.arange(20,4300,1) # make azimuth steps covering the original scaled azimuth range
# # (i.e. 5 degrees around to 355 degree only)

# # Quick plot check of the wrapped image:
# plt.figure(figsize=(8,6))
# plt.pcolormesh(lons,lats,im_wrap,cmap='cubehelix',vmin=0.,vmax=30.)

# # upscale/interpolate the colatitudes to match the upscaled azimuths:
# colats_e_resamp = signal.resample(colats_e_wrap_scaled,len(azis_im_res))
# colats_p_resamp = signal.resample(colats_p_wrap_scaled,len(azis_im_res))

# # Add these to the wrapped image plot to check:
# plt.plot(azis_wrap_scaled,colats_e_wrap_scaled,'+',color='red')   # original wrapped colats
# plt.plot(azis_wrap_scaled,colats_p_wrap_scaled,'+',color='green')
# plt.plot(azis_im_res,colats_e_resamp,color='red')                 # upscaled colats
# plt.plot(azis_im_res,colats_p_resamp,color='green')
# plt.show()

# # Now extract 1440 values from the middle of the wrapped array - this will form the final mask
# azis_resamp_unwrapped = azis_im_res[1420:2860] - 1440   # -1440 sets the first azimuth back to zero

# colats_e_resamp_unwrapped = colats_e_resamp[1420:2860]  # and the same for the colatitudes
# colats_p_resamp_unwrapped = colats_p_resamp[1420:2860]

# lons = np.arange(0,1440,1)  # redefine original lon-lat space
# lats = np.arange(0,720,1)

# # Overplot new full-resolution UVIS auroral boundaries on the centred HST image
# # plt.figure(figsize=(8,6))
# # plt.pcolormesh(lons,lats,im_clean,cmap='cubehelix',
# #                vmin=0.,vmax=30.)
# # plt.plot(azis_resamp_unwrapped,colats_e_resamp_unwrapped,color='red')
# # plt.plot(azis_resamp_unwrapped,colats_p_resamp_unwrapped,color='green')
# # plt.show()

# # Need to repmat the lats and fill an array full of these values, to make the mask
# lats_replicated = np.tile(lats, (1440, 1)).T
# # print(lats_replicated.shape)
# # plt.figure()
# # plt.title('Colats replicated')
# # plt.imshow(lats_replicated, cmap='cubehelix',origin='lower')
# # plt.show()

# # Make a logic array with ones inside auroral oval boundaries, zero outside:
# auroral_mask = np.zeros((720, 1440), dtype=bool)   # initialise mask at image size

# for i in range(auroral_mask.shape[1]):  # step through each column
#     logic_col_e = lats_replicated[:, i] < np.ceil(colats_e_resamp_unwrapped[i])   # combine two masks here, one for equatorward values
#     logic_col_p = lats_replicated[:, i] > np.floor(colats_p_resamp_unwrapped[i])  # and one for poleward values
#     logic_col = logic_col_e * logic_col_p                                         # and simply multiply them together
#     auroral_mask[:, i] = logic_col      # assign logic column to the current mask column

# # Plot auroral oval logic mask to check
# plt.figure()
# plt.title('Auroral oval mask logic array')
# plt.imshow(auroral_mask, cmap='cubehelix',origin='lower')
# plt.show()

# # Mask off the image for emitted power calculation over the auroral oval region:
# image_masked_auroral_region = im_clean * auroral_mask

# plt.figure(figsize=(8,6))
# plt.pcolormesh(lons,lats,image_masked_auroral_region,cmap='cubehelix',
#                vmin=0.,vmax=30.)
# plt.title('Image masked off by the statistical auroral region')
# plt.show()

# # And back-project the masked image:
# bimage_oval = cylbroject(np.flip(np.flip(image_masked_auroral_region, axis=1)),ndiv=2)   # couple of flips required to get hemisphere and dawn-dusk orientation correct in back-projection
# # bimage = cylbroject(np.flip(np.flip(im_clean, axis=1)), ndiv=2)   # couple of flips required to get hemisphere and dawn-dusk orientation correct in back-projection

# plt.figure()
# plt.title('Intensities within the statistical auroral oval')
# plt.imshow(bimage_oval, cmap='cubehelix',origin='lower')
# # plt.colorbar()
# # cbar = plt.colorbar(pad=0.05)
# # cbar.ax.set_ylabel('Intensity [kR]',fontsize=12)
# plt.show()

# # calculate emitted power from ROI in GW (exposure time not required here as kR intensities are per second):
# distance_squared = (dist * au_to_km)**2          # AU in km
# total_power_emitted_from_oval = np.nansum(bimage_oval) * cts2kr * distance_squared * gustin_conv_factor / 1e9

# print('Total power emitted from auroral oval in GW:')
# print(total_power_emitted_from_oval)


# # ==============================================================================
# # Polar projection of ROI/auroral oval with emitted power:
# # ==============================================================================

# rho   = np.linspace(0,180,     num=720 ) # colat vector with image pixel resolution steps
# theta = np.linspace(0,2*np.pi,num=1440) # longitude vector in radian space and image pixel resolution steps

# plt.figure(figsize=(8,6))
# ax = plt.subplot(projection='polar')           # initialize polar projection
# ax.set_title('Polar projection.')
# plt.fill_between(theta, 0, 40, alpha=0.2,hatch="/",color='gray')
# ax.set_theta_zero_location("N")                # set angle 0.0 to top of plot
# ax.set_xticklabels(['00','03','06','09','12','15','18','21'],
#                    fontweight='bold',fontsize=fs)
# ax.tick_params(axis='x',pad=-1.)               # shift position of LT labels
# ax.set_yticklabels(['','','','',''])           # turn off auto lat labels
# ax.set_yticks([0,10,20,30,40])                 # but set grid spacing
# ax.set_ylim([0,40])                            # max colat range

# # plot image data in linear colour-scale:
# pcol1 = plt.pcolormesh(theta,rho,im_clean,cmap='cubehelix',
#                        vmin=0.,vmax=30.,linewidth=0, rasterized=True)
# pcol1.set_edgecolor('face')

# auroral_mask[auroral_mask == False] == np.nan
# auroral_mask = auroral_mask.astype(int)

# plt.contour(theta,rho,auroral_mask,  levels=[0.5],   colors='yellow',linewidths=2)
# plt.contourf(theta,rho,auroral_mask, levels=[0.5,1], colors='yellow', alpha=0.3)

# annotation_oval = f"Oval power: {total_power_emitted_from_oval:.2f} GW"
# ax.annotate(annotation_oval, xy=(.4, 0.15), xycoords='axes fraction',
#             fontsize=fs, color='yellow')

# roi_mask_full[roi_mask_full == False] == np.nan
# roi_mask_full = roi_mask_full.astype(int)
# plt.contour(theta,rho,roi_mask_full,  levels=[0.5],   colors='red',linewidths=2)
# plt.contourf(theta,rho,roi_mask_full, levels=[0.5,1], colors='red', alpha=0.3)

# annotation_roi = f"ROI power: {total_power_emitted_from_roi:.2f} GW"
# ax.annotate(annotation_roi, xy=(.4, 0.20), xycoords='axes fraction',
#             fontsize=fs, color='red')

# # plt.plot(np.radians(azis_resamp_unwrapped/4), colats_e_resamp_unwrapped/4, color='red')
# # plt.plot(np.radians(azis_resamp_unwrapped/4), colats_p_resamp_unwrapped/4, color='green')

# # plt.savefig('polar_test.pdf', format='pdf', bbox_inches='tight',dpi=600)

# plt.show()
