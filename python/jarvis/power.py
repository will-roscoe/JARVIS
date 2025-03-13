#!/usr/bin/env python3
# --------------- HST_emission_power.py->power.py->jarvis.power ----------------#
"""functions for calculating the total power emitted from an auroral region in GW and the power per unit area in GW/km².

Translation between unprojected/projected images is partly handled by IDL pipeline legacy code from Boston University, and partly from python code provided by Jonny Nichols at Leicester. (broject function)
Emission power computed as per Gustin+ 2012.
Should allow e.g., calculation of total auroral oval UV emission power using statistical auroral boundaries, planetary auroral comparisons, or application to Voronoi image segmentations, etc.

Contributors:
J Nichols;
Juwhan Kim, 03 / 01 / 2005;
Joe Kinrade - 08 / 01 /2025;
JAR:VIS team - 21 / 02 / 2025;
"""

import datetime
from typing import Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from astropy.io import fits
from matplotlib import path

from .const import Power, FITSINDEX, Dirs, CONST
from .utils import fitsheader, fpath, get_datetime

# --------------------------------- CONSTANTS ----------------------------------#
# Nichols constants:

# > Load SPICE kernels, and define planet radii and oblateness:
spice.furnsh(Dirs.KERNEL + "jupiter.mk")  # SPICE kernels
# Nichols spice stuff:
planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto", "Vulcan"]
inx = planets.index("Jupiter")
naifobj = 99 + (inx + 1) * 100
radii = spice.bodvcd(naifobj, "RADII", 3)
# // print(radii)   
rpeqkm = radii[1][0]
rpplkm = radii[1][2]
oblt = 1.0 - rpplkm / rpeqkm

deltas = {"Mars": 0.0, "Jupiter": 240.0, "Saturn": 1100.0, "Uranus": 0.0}

# > In some fits files (Jupiter), these 'delta' values for auroral emission altitude are listed as DELRPKM in the header:
# //delrpkm = fits_obj[0].header['DELRPKM']    # auroral emission altitudes at homopause in km, a.k.a 'deltas'
# > If not (Saturn), it's hard-wired in here depending on the target planet (probably Saturn!):
# -------------------------------- MODULE CONFIG -------------------------------#



# ------------------------------------------------------------------------------#
def __null(*args, **kwargs):
    pass


__write_to_file, __print, __plot = __null, __null, __null
if Power.WRITETO:
    def __write_to_file(args,writeto=Power.WRITETO, **kwargs):
        with open(writeto, "a") as f:
            f.write(" ".join([str(x) for x in [*args, "\n"]]))


if Power.DISPLAY_PLOTS:
    from .extensions import QuickPlot as Qp

    funcdict = {
        "raw_data": Qp._plot_,
        "raw_limbtrim": Qp._plot_raw_limbtrimmed,
        "extracted": Qp._plot_extracted_wcoords,
        "polar_region": Qp._plot_polar_wregions,
        "sq_region": Qp._plot_sqr_wregions,
        "mask": Qp._plot_mask,
        "brj": Qp._plot_brojected,
    }

    def __plot(which, *args, **kwargs):
        fkws = {"figsize": kwargs.pop("figsize", (8, 8))}
        if "polar" in which:
            fkws["subplot_kw"] = {"projection": "polar"}
        fig, ax = plt.subplots(**fkws)
        funcdict[which](ax, *args, **kwargs)
        plt.show()


if Power.DISPLAY_MSGS:

    def __print(*args):
        print(args)  # tqdm.write(*args)  # noqa: T201
# ---------------------------------- MAIN BODY ---------------------------------#


# Nicked from Jonny's pypline.py file:
# Numpy vectorization & loop optimizations for python 3.8+ implemented by W. Roscoe, 2025
def _cylbroject(pimage, cml, dece, dmeq, xcen, ycen, psize, nppa, req, obt, ndiv=2, correct=True):
    """Brojection (back projection) core function.

    Args:
        pimage (np.ndarray): image array to back project.
        cml (float): central meridian longitude (degrees).
        dece (float): declination of the equator (degrees).
        dmeq (float): mean equatorial diameter (degrees).
        xcen (int): x-coordinate of the planet centre (pixels).
        ycen (int): y-coordinate of the planet centre (pixels).
        psize (float): pixel size (arcsec).
        nppa (float): north pole position angle (degrees).
        req (float): equatorial radius (km).
        obt (float): oblateness.
        ndiv (int): number of divisions per pixel.
        correct (bool): correct for area effect.

    Returns:
        np.ndarray

    """
    if nppa == 999:
        nppa = 0  # 999 is a flag for no NPPA in IDL code?
    ny, nx = pimage.shape
    xsize, ysize = 1400, 1400
    rimage = np.zeros((ysize, xsize))
    cimage = np.zeros((ysize, xsize))
    px_size = (psize / 3600.0) * np.radians(dmeq) * 1.49598e8  # pixel size. 1 au = 1.49598e8 km
    sin_nppa, cos_nppa = np.sin(np.radians(nppa)), np.cos(np.radians(nppa))
    sin_dec, cos_dec = np.sin(np.radians(dece)), np.cos(np.radians(dece))
    adj_req = (req / px_size) * (1.0 - obt)
    longs = np.linspace(0, 360, nx * ndiv) + cml
    sin_lon = np.sin(np.radians(longs))
    cos_lon = np.cos(np.radians(longs))
    ll = np.degrees(np.arctan((1.0 - obt) * (1.0 - obt) * np.tan(np.radians(dece + 90.0)) * cos_lon))
    lats = np.linspace(-90, 90, ny * ndiv)
    sin_lat = np.sin(np.radians(lats))
    cos_lat = np.cos(np.radians(lats))
    r = adj_req / np.sqrt(1.0 - (2.0 * obt - obt * obt) * (cos_lat * cos_lat))

    def pxpy(start, end, i):
        x = r[start:end] * cos_lat[start:end] * sin_lon[i]
        y = r[start:end] * sin_lat[start:end]
        z = r[start:end] * cos_lat[start:end] * cos_lon[i]
        px = (x * cos_nppa - (y * cos_dec - z * sin_dec) * sin_nppa + xcen).astype(int)
        py = (x * sin_nppa + (y * cos_dec - z * sin_dec) * cos_nppa + ycen).astype(int)
        return px, py

    if dece < 0.0:

        def start_end(i):
            return 0, int(((ll[i] + 90.0) / 180.0 * ndiv * ny) + 1)
    else:

        def start_end(i):
            return int((ll[i] + 90.0) / 180.0 * ndiv * ny), ndiv * ny

    for i in range(nx * ndiv):
        start, end = start_end(i)
        px, py = pxpy(start, end, i)
        valid = (px >= 0) & (px < xsize) & (py >= 0) & (py < ysize)
        values = pimage[np.arange(start, end) // ndiv, i // ndiv] / (ndiv * ndiv)
        rimage[py[valid], px[valid]] += values[valid]
    #  /  *  Here, correct for area effect. Added by Juwhan Kim, 03 / 01 / 2005.
    if correct:
        value = 1.0 / (ndiv * ndiv)
        for i in range(nx * ndiv):
            start, end = start_end(i)
            px, py = pxpy(start, end, i)
            valid = (px >= 0) & (px < xsize) & (py >= 0) & (py < ysize)
            cimage[py[valid], px[valid]] += value
        non_zero = cimage != 0
        rimage[non_zero] /= cimage[non_zero]

    return rimage


def area_calc(vertices):
    """Calculate the area in units of degrees^2 of the ROI."""
    n = len(vertices)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0


def area_to_km2(area_deg, rad_km):
    """Convert the area from degrees^2 to km^2."""
    return area_deg * (np.pi / 180.0) ** 2 * rad_km**2


def cylbroject(img, fits_obj, ndiv=2):
    """Back-projects the image array."""
    # // self._check_image_loaded(proj=True)      # commented out JK
    __print(f"Brojecting with ndiv = {ndiv}")
    # > PCX,PCY:  Planet centre pixel
    # > NPPA: North pole position angle
    # > PXSEC: Pixel size in arc seconds 'PIXSIZE' for IDL pprocessed Saturn files
    image = np.flip(np.flip(img, axis=1))  # ? flips required to get bimage looking right?
    return _cylbroject(
        image,
        *fitsheader(fits_obj, "CML", "DECE", "DIST", "PCX", "PCY", "PXSEC", "NPPA"),
        rpeqkm + CONST.delrp_jup,
        oblt,
        ndiv,
        True,
    )


def powercalc(
    fits_obj: fits.HDUList, dpr_coords: np.ndarray = None, **kwargs,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Calculate the total power emitted from the ROI in GW and the power per unit area in GW/km².

    Args:
        fits_obj (fits.HDUList): The fits object.
        dpr_coords (np.ndarray): The DPR coordinates.
        **kwargs:
            - writeto(str): alternative file to write to.
            - extname (str): The extension name.

    Returns:
        Tuple[float,float,np.ndarray,np.ndarray]
            - The total power emitted from the ROI in GW,
            - the power per unit area in GW/km²,
            - the area,
            - and the full image.


    """
    # -------------------- FUNC INPUT CHECKS & PREPROCESSING -------------------#
    extname = kwargs.get("extname", "BOUNDARY")
    if not isinstance(dpr_coords, np.ndarray):
        if isinstance(dpr_coords, int):
            dpr_coords = fits_obj[dpr_coords].data
        else:  # > check in the hdulist with the name 'BOUNDARY' and take the newest one
            b_inds = [i for i in range(len(fits_obj)) if fits_obj[i].name == extname]
            ind_ = b_inds[0]
            newestb = fits_obj[b_inds[0]]
            for bo in b_inds:
                if fits_obj[bo].ver > newestb.ver:
                    newestb = fits_obj[bo]
                    ind_ = bo
            data = newestb.data
            dpr_coords = np.array([data["colat"], data["lon"]], dtype=np.float32)
    else:
        ind_ = -1
    if isinstance(dpr_coords, np.recarray):
        dpr_coords = np.array([dpr_coords["colat"], dpr_coords["lon"]], dtype=np.float32)
    if dpr_coords.shape[0] == 2:
        dpr_coords = dpr_coords.T
    dpr_path = path.Path(dpr_coords)
    # ------------------------ MASKING THE IMAGE ARRAY  ------------------------#
    llons, llats = np.meshgrid(np.arange(0, 360, 0.25), np.arange(0, 40, 0.25))  # checked correct
    dark_mask_2d = dpr_path.contains_points(np.vstack([llats.flatten(), llons.flatten()]).T).reshape(
        160, 1440,
    )  # > mask needs to be applied to un-rolled image
    dark_mask_2d = np.flip(dark_mask_2d, axis=1)  # > flip mask vertically to match image
    # --------------- FITS FILE HEADER INFO AND VARIABLE DEFS -----------------#
    # __print(fits_obj.info(output=False))                  #> print file information
    # > accessing specific header info entries:
    cml, dece, dist, cts2kr = fitsheader(fits_obj, "CML", "DECE", "DIST", "CTS2KR")
    # > CML:       Central meridian longitude
    # > DECE:      Declination of the equator in degrees
    # > DIST:      Standard (scaled) Earth-planet distance in AU
    # > CTS2KR:    conversion factor in counts/sec/kR
    __print(1.0 / cts2kr)  # > reciprocal of cts2kr to match in Gustin+2012 Table 1.

    # ------------------------ IMAGE ARRAY EXTRACTION -------------------------#
    image_data = fits_obj[FITSINDEX].data
    __plot("raw_data", image_data)
    # //fits_obj.close()                # close file once you're done with it
    # --------------------------- LIMB TRIMMING -------------------------------#
    # > perform limb trimming based on angle of surface vector normal to the sun
    lonm = np.radians(np.linspace(0.0, 360.0, num=1440))
    latm = np.radians(np.linspace(-90.0, 90.0, num=720))
    limb_mask = np.zeros((720, 1440))  # rows by columns
    cmlr, decr = np.radians([cml, dece])
    for i in range(0, 720):
        limb_mask[i, :] = np.sin(latm[i]) * np.sin(decr) + np.cos(latm[i]) * np.cos(decr) * np.cos(lonm - cmlr)
    limb_mask = np.flip(limb_mask, axis=1)  # flip the mask horizontally, not sure why this is needed
    cliplim = np.cos(np.radians(88.0))  # >set a minimum required vector normal surface-sun angle
    clipind = np.squeeze([limb_mask >= cliplim])  # > False = out of bounds (over the limb)
    image_data[clipind == False] = np.nan  # > set image array values outside clip mask to nans # noqa: E712
    image_centred = image_data.copy()  # don't shift by cml for J bc regions defined in longitude not LT
    im_clean = np.flip(image_centred.copy(), 0)  # > a centred, clipped version of the full image
    im_4broject = (
        im_clean.copy()
    )  # // im_4broject[im_4broject < -100] = np.nan  #> 0-40 degrees colat in image pixel res.
    __plot("raw_limbtrim", im_clean)
    # ------------------------ AURORAL REGION EXTRACTION ----------------------#
    # > flip image vertically if required (ease of indexing) and extract auroral region
    # > (not strictly required but makes polar projections less confusing):
    if fitsheader(fits_obj, "south"):
        image_extract = image_centred.copy()[0:160, :]
    else:
        im_flip = np.flip(image_centred.copy(), 0)
        image_extract = im_flip[
            0:160, :,
        ]  # extract image in colat range 0-40 deg (4*40 = 160 pixels in image lat space):
    __plot("extracted", image_extract)
    __plot("polar_region", image_extract, dpr_coords)
    __plot("sq_region", image_extract, dpr_coords)
    __plot("mask", dark_mask_2d, loc="ROI")
    # > Now mask off the image by setting image regions where mask=False, to NaNs:
    image_extract[dark_mask_2d == False] = np.nan  # noqa: E712
    # > And insert auroral region image back into the full projected image space:
    roi_im_full = np.zeros((720, 1440))  # > new array full of zeros
    roi_im_full[:160, :] = image_extract
    # > Do the same thing for a full image space ROI mask:
    roi_mask_full = np.zeros((720, 1440))
    roi_mask_full[:160, :] = dark_mask_2d
    # ---------------------------- BACK-PROJECTION -----------------------------#
    # > Now at the point we can try back-projecting this projected image mask
    # > (or masked-out image) using the back-project function:
    # > this cylbroject function definition is feeding inputs into _cylbroject - JK

    # > Backprojecting! Image input needs to be full [1440,720] centred projection
    bimage_roi = cylbroject(roi_im_full, fits_obj, ndiv=2)
    full_image = cylbroject(im_4broject, fits_obj, ndiv=2)
    # // bimage = cylbroject(image_centred,ndiv=2)
    __plot("brj", full_image, loc="full")
    __plot("brj", bimage_roi, loc="ROI")
    # -------------------------- EMISSION POWER CALC ---------------------------#
    # > Once the back-projected image looks OK, we can proceed with the emission power calculation here.
    # > ISOLATE THE ROI INTENSITIES IN A FULL 1440*720 PROJECTED IMAGE (all other pixels set to nans/zeros)
    distance_squared = (dist * CONST.au_to_km) ** 2  # > AU in km
    # > calculate emitted power from ROI in GW (exposure time not required here as kR intensities are per second):
    total_power_emitted_from_roi = np.nansum(bimage_roi) * cts2kr * distance_squared * CONST.gustin_factor / 1e9
    area = area_to_km2(area_calc(dpr_coords), rpeqkm + 240)
    power_per_area = total_power_emitted_from_roi / area
    __print(f"Total power emitted from ROI in GW:\n{total_power_emitted_from_roi}")
    __print(f"Power per unit area in GW/km²:\n{power_per_area}")
    # VISIT DATETIME POWER FLUX AREA EXTNAME LMIN LMAX NUMPTS CALCTIME



    args1= kwargs.get("exinfo")
    args1 = list(args1.values()) if args1 is not None else fitsheader(fits_obj, "LMIN", "LMAX", "NUMPTS", ind=ind_)
    __write_to_file(
        writeto=kwargs.get("writeto", Power.WRITETO),
        args=(fitsheader(fits_obj, "VISIT"),
        get_datetime(fits_obj),
        total_power_emitted_from_roi,
        power_per_area,
        area,
        *args1,
        datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        ))
    return {"visit":fitsheader(fits_obj, "VISIT"),"datetime":
        get_datetime(fits_obj),"power":total_power_emitted_from_roi, "flux":power_per_area, "area":area, "fullim":full_image, "roi":bimage_roi,"imex":image_extract,"coords":dpr_coords}


    
    

def avg_intensity(img: np.ndarray) -> float:
    """Return the average intensity of the image."""
    return np.nanmean(img), np.nanstd(img)
