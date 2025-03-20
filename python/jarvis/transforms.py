#!/usr/bin/env python3
"""Functions for transforming image arrays and FITS ImageHDUs into different coordinate systems or processed arrays."""
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.io.fits import HDUList, getdata
from scipy.signal import convolve2d

from .const import FITSINDEX, IMG
from .utils import adapted_hdul, fitsheader


##########################################################################################################
#                            COORDINATE SYSTEM TRANSFORMS
##########################################################################################################
def azimeq_to_polar(*points, img: np.ndarray, rlim: int = 40) -> np.ndarray:
    """Transform a x,y point or points to polar colatitude and longitude.

    Args:
    *points (list,np.ndarray): list of x,y coordinates of the points [[x1,y1],[x2,y2],...]
    img (np.ndarray): the image the coordinates are from
    rlim (float): the maximum colatitude value

    """
    if len(points) == 2:
        xys = np.array([points]) if isinstance(points[0], (int, float)) else np.array(points).T
    elif len(points)>2:
        # points is a list of points; [[x1,y1],[x2,y2],...]
        xys = np.array(points)
    elif len(points)==1:
        # points is already an array
        xys = points[0] if isinstance(points[0],np.ndarray) else np.array(points[0])
    else:
        raise ValueError("points must be a list of points or a list of x and y values.")
    cl = []
    r0 = img.shape[1] / 2
    for i, (x, y) in enumerate(xys):
        x_ = x - r0
        y_ = y - r0
        r = np.sqrt(x_**2 + y_**2)
        colat = r / r0 * rlim
        lon = np.degrees(np.arctan2(y_, x_)) + 90
        while lon < 0:
            lon += 360
        while lon > 360:
            lon -= 360
        cl.append((colat, lon))
    return np.array(cl)


def polar_to_azimeq(colat: float, lon: float, img: np.ndarray, rlim: float) -> np.ndarray:
    """Transform polar colatitude and longitude to full image coordinates.

    Args:
    colat: the colatitude value
    lon: the longitude value
    img: the image the coordinates are from
    rlim: the maximum colatitude value

    """
    r0 = img.shape[1] / 2
    x_ = r0 * np.cos(np.radians(lon - 90))
    y_ = r0 * np.sin(np.radians(lon - 90))
    rxy = colat / rlim * r0
    x = x_ + rxy
    y = y_ + rxy
    return [x, y]


def azimuthal_equidistant(data, rlim=40, meridian_pos="N", origin="upper",clip_low=0,clip_high=3000, **kwargs):
    """Convert colatitude,longitude array to a "polar projection" aka azimuthal equidistant projection.

    Args:
        data (`Any`): HDUList or np.array to get the original data from.
        img (`np.ndarray`): 2D array (colatitude, longitude) with data.
        rlim (`float`): Maximum colatitude to include in degrees (default: `40`).
        clip_low (`float`): value to clip on the low limit. default = 0, set to -np.inf for no lim
        clip_high (float): value to set as maximum. all above this will be set to this value. set to np.inf for no lim. default = 3000
        meridian_pos (`str`): which direction to align the meridian (0° longitude) to. (default: `False`)
        origin (`str`): where the origin of the array is (`upper`= imshow default, yaxis faces down; `lower`=pcolormesh, axis in standard arrangement). (Default: `upper`)
        **kwargs:
            - south (bool): explicitly identify whether the projection should be north or south.
            - drawcml (float): if provided, will fill in points along the cml with it.
            - clip_angle (float): define the longitude boundary in angle away from CML in degrees symmetrically.
            - clip_angle (tuple[float]): two angles defining the longitude boundary, relative to the CML.
            - proj_bg (): background value of the circle. default is MAX
            - arr_bg (): background value of the array. default is 0.

    Returns:
        np.ndarray: Projected image in azimuthal equidistant coordinates.

    """
    sgn = 1 if origin == "upper" else -1 # flip the sign if origin is upper
    # differentiate between data and fits object
    if isinstance(data,str):
        d = getdata(data, FITSINDEX)
        img = np.asarray(d.astype(np.float32))
        fits_obj=fits.open(data)
    else:
        img = np.asarray(data[FITSINDEX].data)
        fits_obj=data
    cml, is_south = fitsheader(fits_obj, "CML", "south")
    is_south = kwargs.get("south",is_south)
    cm = kwargs.get("clip_angles", *[(-1*c, c) for c in [kwargs.get("clip_angle",90)]])
    n_theta, n_phi = img.shape  # Grid size of input (θ, φ)
    output_size = n_theta  # Output image size (square)
    # Convert longitude limits to radians
    corrected_cml = (-cml) % 360
    lon_clip = [(corrected_cml+cm[0]) % 360, (corrected_cml+cm[1]) % 360]
    lon_clip_pixel = [int(i / 360 * n_phi) for i in lon_clip]
    lon_mask = np.zeros_like(img, dtype=bool) # masking out the longitudes we dont want
    lon_mask[:, lon_clip_pixel[0] : lon_clip_pixel[1]] = True
    img[lon_mask==False] = np.nan # set unused longitudes to the square bg. # noqa: E712
    if not is_south:
        img = np.flip(img, axis=0)  # flip to have north pole at 0 index (image data is south at 0)
    # Compute the pixel corresponding to the longitude
    cml_pixel = int(corrected_cml / 360 * n_phi)

    if kwargs.get("drawcml",False): # Annotate the line vertically at the correct longitude
        img[:, cml_pixel] = kwargs["drawcml"]

    img = np.nan_to_num(img, nan=kwargs.get("proj_bg",np.max(np.nan_to_num(img)))) # turn nan values to circle bg
    # if zero, also set to proj_bg.
    img = np.where(img<=1, kwargs.get("proj_bg",np.max(np.nan_to_num(img))), img)
    img = np.clip(img, clip_low, clip_high) # clipping out useless data.
    # Flip and shift so that the origin is at the top, but we still view the correct part when rotated.
    if origin == "lower":
        img = np.roll(np.flip(img, axis=1), shift=n_phi // 2, axis=1)
    ## Dealing with the cases when meridian pos needs to be changed (rotate the image by translating longitudes)
    sh = n_phi//4 * sgn
    p = meridian_pos.lower()
    if p.startswith("t") or p.startswith("n"):
        img = np.roll(img, shift=2 * sh, axis=1)  # roll to have 0° long at top (0 indexed)
    elif p.startswith("b") or p.startswith("s"):
        pass  # img = np.roll(img, shift=N_phi//2, axis=1)
    elif p.startswith("l") or p.startswith("w"):
        img = np.roll(img, shift=3 * sh, axis=1)
    elif p.startswith("r") or p.startswith("e"):
        img = np.roll(img, shift=sh, axis=1)
    theta_lim_rad = np.radians(rlim)
    # Create a coordinate grid for the output image
    y, x = np.meshgrid(np.linspace(-1, 1, output_size), np.linspace(-1, 1, output_size))
    r = np.sqrt(x**2 + y**2)  # Radial distance from center
    mask = r <= 1  # Valid points inside the projection circle
    # colatitude θ
    theta = np.arcsin(r * np.sin(theta_lim_rad)) * (n_theta - 1) / np.pi  # Scale to input grid
    # Compute longitude φ
    phi = np.arctan2(y, x)  # Angle in radians
    phi[phi < 0] += 2 * np.pi  # Convert to [0, 2π]
    phi = phi / (2 * np.pi) * (n_phi - 1)  # Scale to input grid indices
    # Create empty projected image
    projected_array = np.full((output_size, output_size),kwargs.get("arr_bg",0))
    # Nearest-neighbor interpolation
    theta_idx = np.clip(np.round(theta).astype(int), 0, n_theta - 1)
    phi_idx = np.clip(np.round(phi).astype(int), 0, n_phi - 1)
    # # Fill the projected array using the computed indices
    projected_array[mask] = img[theta_idx[mask], phi_idx[mask]]
    return projected_array#, copies

##########################################################################################################
#                             IMAGE ARRAY TRANSFORMS
##########################################################################################################
def contrast_adjust(im:np.ndarray, **kwargs):
    r"""Contrast enhancement function for images. Ensures the minimum and maximum values are unchanged.

    $$K(v)=v_{min}+(v_{max}-v_{min}) (\\frac{v-v_{min}}{v_{max}-v_{min}}}^{\\frac{1}{k}}$$
    """
    v_min,v_max = kwargs.get("lims",(np.min(im),np.max(im)))
    return v_min + (v_max - v_min) * ((im - v_min) / (v_max - v_min))**(1/kwargs.get("k",IMG.contrast.power))



def gaussian_blur(
    input_arr: np.ndarray, radius: int, amount: float, boundary: str = "wrap", mode: str = "same",
) -> np.ndarray:
    """Return the input array convolved with the kernel2d convolution kernel.

    Args:
    input_arr (np.ndarray): input array to apply the kernel to.
    radius (int): pixel radius of the kernel.
    amount (float): standard deviation of the gaussian kernel.
    boundary (str): see scipy.signals.convolve2d.
    mode (str): see scipy.signals.convolve2d.

    """
    kernel2d = np.exp(-(np.arange(-radius, radius + 1) ** 2) / (2 * amount**2))
    kernel2d = np.outer(kernel2d, kernel2d)
    kernel2d /= np.sum(kernel2d)
    return convolve2d(input_arr, kernel2d, mode=mode, boundary=boundary)


def gradmap(
    input_arr: np.ndarray,
    kernel2d: np.array = np.array([[-1 - 1j, -2j, 1 - 1j], [-2, 0, 2], [-1 + 1j, 2j, 1 + 1j]]),
    boundary: str = "wrap",
    mode: str = "same",
) -> np.ndarray:
    """Return the gradient of the input array using the kernel2d convolution kernel.

    convolution kernels used are from Swithenbank-Harris, B.G., Nichols, J.D., Bunce, E.J. 2019
    ```
    Gx = [[-1  0  1],[-2  0  2],[-1  0  1]]
    Gy = [[-1 -2 -1],[ 0  0  0],[ 1  2  1]]
               ((==> G = Gx + j*Gy ==>))
    G= [[-1-1j 0-2j 1-1j],[-2    0     2  ],[-1+1j 0+2j  1+1j]]
    ```
    """
    complexret = convolve2d(input_arr, kernel2d, mode=mode, boundary=boundary)
    return np.abs(complexret)


def dropthreshold(input_arr: np.ndarray, threshold: float) -> np.ndarray:
    """Return a the input array, but with valuesbelow the threshold set to 0."""
    return np.where(input_arr < threshold, 0, input_arr)



def coadd(input_arrs: list[np.ndarray], weights: Optional[list[float]] = None) -> np.ndarray:
    """Coadd a list of arrays with optional weights.

    input arrs N arrays of x*y shape
    """
    if weights is None:
        weights = [1 for i in range(len(input_arrs))]
    combined = np.stack(input_arrs, axis=0)
    return np.average(combined, axis=0, weights=weights)


def normalize(input_arr: np.ndarray) -> np.ndarray:
    """Normalize the input array to the range [0, 1]."""
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)
    return (input_arr - min_val) / (max_val - min_val)


##########################################################################################################
#                         FITS DATA TRANSFORMS
##########################################################################################################
def adaptive_coadd(input_fits: list[HDUList], eff_ang: int = 90) -> HDUList:
    """Coadd a list of fits objects with different resolutions, by aligning them to a primary fits object."""
    # each array has better resolution closer to cml, and worse at the edges. we need to identify how
    cmls = [fitsheader(f, "CML") for f in input_fits]
    effective_lims = [[cml - eff_ang, cml + eff_ang] for cml in cmls]
    for i in range(len(effective_lims)):
        if effective_lims[i][0] < 0:
            effective_lims[i][0] += 360
        if effective_lims[i][1] > 360:
            effective_lims[i][1] -= 360
    # for each longitude, we need to identify which of the input fits objects are relevant


def align_cmls(input_fits: list[HDUList], primary_index: int) -> list[HDUList]:
    """Align a list of fits objects to a primary fits object by rolling the images to match the CML positions."""
    primary_fits = input_fits[primary_index]  # align all images to this one
    data0, header0 = primary_fits[FITSINDEX].data, primary_fits[FITSINDEX].header  # 'zero point' data and fits
    cml0 = header0["CML"]  # 'zero point' cml
    assert all(f[FITSINDEX].data.shape == data0.shape for f in input_fits), "All images must have the same shape."
    height, width = data0.shape  # get shape, so we can identify index to roll by
    diffs = [cml0 - fitsheader(f, "CML") for f in input_fits]  # angle differences
    dwidths = [int(d / 360 * width) for d in diffs]  # index/pixel differences
    aligned = [np.roll(f[FITSINDEX].data, d, axis=1) for f, d in zip(input_fits, dwidths)]  # roll each image
    return [adapted_hdul(f, new_data=arr) for f, arr in zip(input_fits, aligned)]  # return new fits objects.
