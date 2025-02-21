#> === HST_emission_power.py ===================================================
#> Script to read in projected/unprojected HST STIS .fits images, define a
#> spatial region, and calculate the UV emission power.
#>
#> Translation between unprojected/projected images is partly handled by IDL
#> pipeline legacy code from Boston University, and partly from python code
#> provided by Jonny Nichols at Leicester. (broject function)

#> Emission power computed as per Gustin+ 2012.

#> Should allow e.g., calculation of total auroral oval UV emission power using
#> statistical auroral boundaries, planetary auroral comparisons, or
#> application to Voronoi image segmentations, etc.
#> === Joe Kinrade - 8 January 2025 ============================================
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from matplotlib import path
from .const import KERNELDIR, FITSINDEX
from .utils import fitsheader, get_datetime
# Nichols constants:
au_to_km = 1.495978707e8
gustin_conv_factor =  9.04e-10 # 1.02e-9 #> "Conversion factor to be multiplied by the squared HST-planet distance (km) to determine the total emitted power (Watts) from observed counts per second."

#> Load SPICE kernels, and define planet radii and oblateness:
spice.furnsh(KERNELDIR+'jupiter.mk') # SPICE kernels
# Nichols spice stuff:
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn',
            'Uranus', 'Neptune', 'Pluto', 'Vulcan']
inx = planets.index('Jupiter')
naifobj = 99 + (inx + 1) * 100
radii = spice.bodvcd(naifobj, 'RADII', 3)
#// print(radii)
rpeqkm = radii[1][0]
rpplkm = radii[1][2]
oblt = 1. - rpplkm / rpeqkm


    # Nicked from Jonny's pypline.py file: 
    # Numpy vectorization & loop optimizations for python 3.8+ implemented by W. Roscoe, 2025
def _cylbroject(pimage, cml, dece, dmeq, xcen, ycen, psize, nppa, req, obt, 
                ndiv=2, correct=True):
    """
    cml: central meridian longitude (degrees)
    dece: declination of the equator (degrees)
    dmeq: mean equatorial diameter (degrees)
    xcen,ycen: x,y-coordinate of the planet centre (pixels)
    psize: pixel size (arcsec)
    nppa: north pole position angle (degrees)
    req: equatorial radius (km)
    obt: oblateness
    ndiv: number of divisions per pixel
    correct: correct for area effect
    """
    if nppa == 999:
        nppa = 0 # 999 is a flag for no NPPA in IDL code?
    ny, nx = pimage.shape
    xsize, ysize = 1400, 1400 
    rimage = np.zeros((ysize, xsize))
    cimage = np.zeros((ysize, xsize))
    px_size = (psize / 3600.) * np.radians(dmeq) * 1.49598e8 #pixel size. 1 au = 1.49598e8 km
    sin_nppa, cos_nppa = np.sin(np.radians(nppa)), np.cos(np.radians(nppa))
    sin_dec, cos_dec = np.sin(np.radians(dece)), np.cos(np.radians(dece))
    adj_req = (req / px_size) * (1.0 - obt)
    longs = np.linspace(0, 360, nx * ndiv) + cml
    sin_lon = np.sin(np.radians(longs))
    cos_lon = np.cos(np.radians(longs))
    ll = np.degrees(np.arctan(
        (1.0 - obt)*(1.0 - obt)*np.tan(np.radians(dece + 90.0))*cos_lon))
    lats = np.linspace(-90, 90, ny * ndiv)
    sin_lat = np.sin(np.radians(lats))
    cos_lat = np.cos(np.radians(lats))
    r = adj_req / np.sqrt(1.0 - (2.0*obt - obt*obt) * (cos_lat*cos_lat))
    def pxpy(start, end, i):
        x=r[start:end] * cos_lat[start:end] * sin_lon[i]
        y=r[start:end] * sin_lat[start:end]
        z=r[start:end] * cos_lat[start:end] * cos_lon[i]
        px = (x*cos_nppa - (y*cos_dec - z*sin_dec)*sin_nppa + xcen).astype(int)
        py = (x*sin_nppa + (y*cos_dec - z*sin_dec)*cos_nppa + ycen).astype(int)
        return px, py
    
    if dece < 0.0:
        def start_end(i):
            return 0, int(((ll[i] + 90.0) / 180.0*ndiv*ny) + 1)
    else:
        def start_end(i):
            return int(((ll[i] + 90.0) / 180.0*ndiv*ny)), ndiv*ny
    
    for i in range(nx*ndiv):
            start, end = start_end(i)
            px, py = pxpy(start, end, i)
            valid = (px >= 0) & (px < xsize) & (py >= 0) & (py < ysize)
            values = pimage[np.arange(start, end)//ndiv, i // ndiv] / (ndiv*ndiv)
            rimage[py[valid], px[valid]] += values[valid]
    #  /  *  Here, correct for area effect. Added by Juwhan Kim, 03 / 01 / 2005.
    if correct:
        value = 1.0 / (ndiv*ndiv)
        for i in range(nx*ndiv):
            start, end = start_end(i)
            px, py = pxpy(start, end, i)
            valid = (px >= 0) & (px < xsize) & (py >= 0) & (py < ysize)
            cimage[py[valid], px[valid]] += value
        non_zero = cimage != 0
        rimage[non_zero] /= cimage[non_zero]

    return rimage

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

def area_to_km2(area_deg,rad_km):
    '''Converts the area from degrees^2 to km^2'''
    return area_deg * (np.pi/180.)**2 * rad_km**2



def powercalc(fits_obj:fits.HDUList="", dpr_coords:np.ndarray=""):
    #-------------------- FUNC INPUT CHECKS & TRANSFORMS ------------------------#
    if not isinstance(dpr_coords, np.ndarray):
        if isinstance(dpr_coords, int):
            dpr_coords = fits_obj[dpr_coords].data
        else:
            #> check in the hdulist with the name 'BOUNDARY' and take the newest one 
            b_inds = [i for i in range(len(fits_obj)) 
                      if fits_obj[i].name == 'BOUNDARY']
            newestb = fits_obj[b_inds[0]]
            for bo in b_inds:
                if fits_obj[bo].header['EXTVER'] > newestb.header['EXTVER']:
                    newestb = fits_obj[bo]
            dpr_coords = newestb.data
    #> if the columns are the wrong way round ie [ [colat1,colat2...], [lon1,lon2...],]
    if dpr_coords.shape[0] == 2:
        #> then make the columns the right way round, [[colat1,lon1],[colat2,lon2],...]
        dpr_coords = dpr_coords.T
    dpr_path = path.Path(dpr_coords) 
    #-------------------------- MASKING THE IMAGE ARRAY  ------------------------#
    llons, llats = np.meshgrid(np.arange(0,360,0.25), np.arange(0,40,0.25)) # checked correct
    dark_mask_2d = dpr_path.contains_points(np.vstack(
        [llats.flatten(), llons.flatten()]).T).reshape(160,1440)#> mask needs to be applied to un-rolled image
    # ----------------- FITS FILE HEADER INFO AND VARIABLE DEFS -----------------#
    #fits_obj.info()                    #> print file information
    #> accessing specific header info entries:
    cml       = fits_obj[FITSINDEX].header['CML']
    dece      = fits_obj[FITSINDEX].header['DECE']
    pcx       = fits_obj[FITSINDEX].header['PCX']        #> Planet centre pixel
    pcy       = fits_obj[FITSINDEX].header['PCY']        #> Planet centre pixel
    nppa      = fits_obj[FITSINDEX].header['NPPA']       #> North pole position angle
    pxsec   = fits_obj[FITSINDEX].header['PXSEC']        #> Pixel size in arc seconds 'PIXSIZE' for IDL pprocessed Saturn files
    dist      = fits_obj[FITSINDEX].header['DIST']       #> Standard (scaled) Earth-planet distance in AU
    cts2kr    = fits_obj[FITSINDEX].header['CTS2KR']     #> conversion factor in counts/sec/kR
    #print(1./cts2kr)                                    #> reciprocal of cts2kr to match in Gustin+2012 Table 1.
    #> If 1 / conversion factor is ~3994, this implies a colour ratio of 1.10. for Saturn with a STIS SrF2 image (see Gustin+ 2012 Table 1):
    #> And this in turn means that the counts-per-second to total emitted power (Watts). conversion factor is 9.04e-10 (Gustin+2012 Table 2), for STIS SrF2:
    #> In some fits files (Jupiter), these 'delta' values for auroral emission altitude are listed as DELRPKM in the header:
    delrpkm = fits_obj[0].header['DELRPKM']    # auroral emission altitudes at homopause in km, a.k.a 'deltas'
    #> If not (Saturn), it's hard-wired in here depending on the target planet (probably Saturn!):
    deltas = {'Mars': 0., 'Jupiter': 240., 'Saturn': 1100., 'Uranus': 0.}
    delrpkm = deltas['Jupiter']
    #> convert HST timestamp to time at "object" using light travel time:
    start_time = get_datetime(fits_obj)     # create datetime object
    # ------------------------ IMAGE ARRAY EXTRACTION ---------------------------#
    image_data = fits_obj[FITSINDEX].data   
    #! _plot_raw()
    #//fits_obj.close()                # close file once you're done with it
    # --------------------------- LIMB TRIMMING ---------------------------------#
    #> perform limb trimming based on angle of surface vector normal to the sun
    lonm = np.radians(np.linspace(0.,360.,num=1440))
    latm = np.radians(np.linspace(-90.,90.,num=720))
    limb_mask = np.zeros((720,1440))   # rows by columns
    cmlr, decr = np.radians([cml, dece])    
    for i in range(0,720):
        limb_mask[i,:] = (np.sin(latm[i])*np.sin(decr) + 
        np.cos(latm[i])*np.cos(decr)*np.cos(lonm-cmlr))
    limb_mask    = np.flip(limb_mask,axis=1)          # flip the mask horizontally, not sure why this is needed
    cliplim      = np.cos(np.radians(88.))            # >set a minimum required vector normal surface-sun angle
    clipind      = np.squeeze([limb_mask >= cliplim]) #> False = out of bounds (over the limb)
    image_data[clipind  == False] = np.nan  #> set image array values outside clip mask to nans # noqa: E712
    image_centred = image_data.copy() # don't shift by cml for J bc regions defined in longitude not LT
    im_clean = np.flip(image_centred.copy(),0)    #> a centred, clipped version of the full image
    im_4broject = im_clean.copy()       #// im_4broject[im_4broject < -100] = np.nan  #> 0-40 degrees colat in image pixel res.
    #! _plot_raw_limbtrimmed()
    # -------------------------- AURORAL REGION EXTRACTION ----------------------#
    #> flip image vertically if required (ease of indexing) and extract auroral region
    #> (not strictly required but makes polar projections less confusing):
    if fitsheader(fits_obj,'south'):
        image_extract = image_centred.copy()[0:160,:]
    else:
        im_flip = np.flip(image_centred.copy(),0)
        image_extract = im_flip[0:160,:] # extract image in colat range 0-40 deg (4*40 = 160 pixels in image lat space):
    #! _plot_extracted_wcoords()
    #! _plot_polar_wregions()
    #! _plot_sqr_regions()
    #! _plot_roi_mask()
    #> Now mask off the image by setting image regions where mask=False, to NaNs:
    image_extract[dark_mask_2d==False] = np.nan # noqa: E712
    #> And insert auroral region image back into the full projected image space:
    roi_im_full          = np.zeros((720,1440))        #> new array full of zeros
    roi_im_full[0:160,:] = image_extract
    #> Do the same thing for a full image space ROI mask:
    roi_mask_full          = np.zeros((720,1440))
    roi_mask_full[0:160,:] = dark_mask_2d
    #------------------------------ BACK-PROJECTION -----------------------------#
    #> Now at the point we can try back-projecting this projected image mask
    #> (or masked-out image) using the back-project function:
    #> this cylbroject function definition is feeding inputs into _cylbroject - JK

    def cylbroject(image, ndiv=2):
        #// self._check_image_loaded(proj=True)      # commented out JK
        print('Brojecting with ndiv = ', ndiv)
        bimage = _cylbroject(image,cml,dece,dist,pcx,pcy,pxsec,nppa,
                             rpeqkm+delrpkm,oblt,ndiv,True)
        return bimage
    #> Backprojecting! Image input needs to be full [1440,720] centred projection
    bimage_roi = cylbroject(np.flip(np.flip(roi_im_full, axis=1)),ndiv=2)   #? flips required to get bimage looking right?
    full_image = cylbroject(np.flip(np.flip(im_4broject, axis=1)), ndiv=2)
    #// bimage = cylbroject(image_centred,ndiv=2)
    #! _plot_brojected_full()
    #! _plot_brojected_roi()

    #-------------------------- EMISSION POWER CALC ----------------------------#
    #> Once the back-projected image looks OK, we can proceed with the emission power calculation here.
    #> ISOLATE THE ROI INTENSITIES IN A FULL 1440*720 PROJECTED IMAGE (all other pixels set to nans/zeros)
    distance_squared = (dist * au_to_km)**2          #> AU in km
    #> calculate emitted power from ROI in GW (exposure time not required here as kR intensities are per second):
    total_power_emitted_from_roi = (np.nansum(bimage_roi) * cts2kr * 
                                    distance_squared * gustin_conv_factor / 1e9)
    area = area_to_km2(area_calc(dpr_coords), rpeqkm+240)
    power_per_area = total_power_emitted_from_roi / area
    print('Total power emitted from ROI in GW:')
    print(total_power_emitted_from_roi)
    print('Power per unit area in GW/km^2:')
    print(power_per_area)
    visit= fits_obj[0].header['VISIT']
    filepath = 'powers.txt'
    with open(filepath, 'a') as f:
        f.write(visit + ' ' + str(start_time) + ' ' + str(total_power_emitted_from_roi) 
                + ' ' + str(power_per_area) + '\n')
    return total_power_emitted_from_roi, power_per_area, full_image, bimage_roi, roi_mask_full















def _plot_raw(ax, image_data, cmap='cubehelix', origin='lower', vlim=(1., 1000.)):#?
    # make a quick-look plot to check image array content:
    ax.set(title='Image array in fits file.', xlabel='longitude pixels', ylabel='co-latitude pixels')
    im = ax.imshow(image_data, cmap=cmap,origin=origin, vmin=vlim[0], vmax=vlim[1])
    cbar = plt.colorbar(pad=0.05, ax=ax, mappable=im)
    cbar.ax.set_ylabel('Intensity [kR]',fontsize=12)
    return ax


def _plot_raw_limbtrimmed(ax, im_clean,lons=np.arange(0,1440,1),lats=np.arange(0,720,1), cmap='cubehelix',vlim=(0.,1000.)):#?
    # Quick plot check of the centred, limb-trimmed image:
    ax.set(title='Image centred, limb-trimmed.', xlabel='longitude pixels', ylabel='co-latitude pixels')
    ax.pcolormesh(lons,lats,np.flip(np.flip(im_clean, axis=1)),cmap=cmap,
                vmin=vlim[0],vmax=vlim[1])
    return ax

def _plot_extracted_wcoords(ax:plt.Axes, image_extract,cmap='cubehelix',vlim=(0., 1000.)):#?
    ax.set(title='Auroral region extracted.', xlabel='SIII longitude [deg]', ylabel='co-latitude [deg]')
    im = ax.imshow(image_extract,cmap=cmap,vmin=vlim[0],vmax=vlim[1])
    ax.set_yticks(ticks=[0*4,10*4,20*4,30*4,40*4], labels=['0','10','20','30','40'])
    ax.set_xticks(ticks=[0*4,90*4,180*4,270*4,360*4], labels=['360','270','180','90','0'])
    #plt.set_yticklabels(['0','10','20','30','40'])   
    cbar = plt.colorbar(pad=0.05, mappable=im, ax=ax)
    cbar.ax.set_ylabel('Intensity [kR]',fontsize=12)
    return ax

def _plot_polar_wregions(ax: plt.Axes, image_extract,cpath, rho=np.linspace(0,40,num=160), theta=np.linspace(0,2*np.pi,num=1440), fs=12, cmap='cubehelix', vlim=(1.,1000.)):#?
    # ==============================================================================
    # make polar projection plot
    # ==============================================================================
    # set up polar coords
    #rho    # colat vector with image pixel resolution steps
    #theta  # longitude vector in radian space and image pixel resolution steps
    dr = [[i[0] for i in cpath], [i[1] for i in cpath]]
    ax.set(title='Polar projection.', xlabel='SIII longitude [deg]', ylabel='co-latitude [deg]', theta_zero_location='N',
           ylim=[0,40])
    ax.set_yticks(ticks=[0,10,20,30,40], labels=['','','','',''])
    ax.set_xticklabels(['0','315','270','225','180','135','90','45'], # reverse these! ###################
                    fontweight='bold',fontsize=fs)
    ax.tick_params(axis='x',pad=-1.)  
    ax.fill_between(theta, 0, 40, alpha=0.2,hatch="/",color='gray')
    cm=ax.pcolormesh(theta,rho,image_extract,cmap=cmap,vmin=vlim[0],vmax=vlim[1])
    ax.plot([np.radians(360-r) for r in dr[1]], dr[0], color='red',linewidth=3.)
    # -> OVERPLOT THE ROI IN POLAR PROJECTION HERE. <-
    # Add colourbar: ---------------------------------------------------------------
    cbar = plt.colorbar(ticks=[0.,100.,500.,900.],pad=0.05, mappable=cm, ax=ax)
    cbar.ax.set_yticklabels(['0','100','500','900'])
    cbar.ax.set_ylabel('Intensity [kR]',fontsize=12)
    return ax

def _plot_sqr_wregions(ax,image_extract,cpath,testlons=np.arange(0,360,0.25),testcolats=np.arange(0,40,0.25),cmap='cubehelix', vlim=(1.,1000.) ):#?
     # # === Plot full image array using pcolormesh so we can understand the masking
    # # process i.e. isolating pixels that fall inside the ROI =======================
    ax.set(title='ROI', xlabel='SIII longitude [deg]', ylabel='co-latitude [deg]')
    ax.set_xticks(ticks=[0,90,180,270,360], labels=['360','270','180','90','0'])
    dr = [[i[0] for i in cpath], [i[1] for i in cpath]]
    #plt.yticks(ticks=[0*4,10*4,20*4,30*4,40*4], labels=['0','10','20','30','40'])
    ax.pcolormesh(testlons,testcolats,image_extract,cmap=cmap,vmin=vlim[0],vmax=vlim[1])
    ax.plot([360-r for r in dr[1]], dr[0],color='red',linewidth=3.)
    return ax
def _plot_roi_mask(ax,dark_mask_2d):#?
    # Quick plot check of the ROI mask:
    ax.imshow(dark_mask_2d, origin='lower')
    ax.set(title='ROI mask in image space', xlabel='longitude pixels', ylabel='co-latitude pixels')
    return ax

def _plot_brojected_full(ax,full_image,cmap='cubehelix', origin='lower', vlim=(1., 1000.), saveto=None):#?
    ax.set(title='Brojected full image.', xlabel='pixels', ylabel='pixels')
    im=ax.imshow(full_image, cmap=cmap,origin=origin, vmin=vlim[0], vmax=vlim[1])
    cbar = plt.colorbar(pad=0.05, mappable=im, ax=ax)
    cbar.ax.set_ylabel('Intensity [kR]',fontsize=12)
    if saveto is not None:
        plt.savefig(saveto, dpi=350)
    return ax
    

def _plot_brojected_roi(ax, bimage_roi,cmap='cubehelix', origin='lower', vlim=(1., 1000.), saveto=None):#?
    ax.set(title='Brojected ROI image.', xlabel='pixels', ylabel='pixels')
    im = ax.imshow(bimage_roi, cmap=cmap,origin=origin, vmin=vlim[0], vmax=vlim[1])
    cbar = plt.colorbar(pad=0.05, mappable=im, ax=ax)
    cbar.ax.set_ylabel('Intensity [kR]',fontsize=12)
    if saveto is not None:
        plt.savefig(saveto, dpi=350)
    return ax

