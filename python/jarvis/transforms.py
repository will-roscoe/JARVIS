#!/usr/bin/env python3
"""This module contains functions for transforming image arrays and FITS ImageHDUs into different coordinate systems or processed arrays."""

import numpy as np
from astropy.io.fits import HDUList
from scipy.signal import convolve2d

from .const import FITSINDEX
from .utils import adapted_hdul, fitsheader


##########################################################################################################
#                            COORDINATE SYSTEM TRANSFORMS
##########################################################################################################
def fullxy_to_polar(x: int, y: int, img: np.ndarray, rlim: int = 40) -> tuple:
    """transforms a point on an image (x,y=0,0 at top_left) to polar colatitude and longitude.
    x,y: x,y coordinates of the point
    img: the image the coordinates are from
    rlim: the maximum colatitude value
    """
    r0 = img.shape[1] / 2
    x_ = x - r0
    y_ = y - r0
    r = np.sqrt(x_**2 + y_**2)
    colat = r / r0 * rlim
    lon = np.degrees(np.arctan2(y_, x_)) + 90
    while lon < 0:
        lon += 360
    while lon > 360:
        lon -= 360
    return (colat, lon)


def fullxy_to_polar_arr(xys: list[tuple[int, int]], img: np.ndarray, rlim: int = 40) -> np.ndarray:
    """transforms a list of coordinates of points on an image (x,y=0,0 at top_left) to polar colatitude and longitude.
    xy: list of x,y coordinates of the points [[x1,y1],[x2,y2],...]
    img: the image the coordinates are from
    rlim: the maximum colatitude value
    """
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


def polar_to_fullxy(colat: float, lon: float, img: np.ndarray, rlim: float) -> np.ndarray:
    """transforms polar colatitude and longitude to full image coordinates.
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


def azimuthal_equidistant(fits_obj, rlim=40, clip_negatives=False, meridian_pos="N", origin="upper", **kwargs):
    """
    converts colatitude,longitude array to a "polar projection" aka azimuthal equidistant projection
    Parameters:
        img (`np.ndarray`): 2D array (colatitude, longitude) with data.
        rlim (`float`): Maximum colatitude to include in degrees (default: `40`).
        clip_negatives (`bool`): make all negative values in the returned array 0 (default: `False`)
        meridian_pos (`str`): which direction to align the meridian (0° longitude) to. (default: `False`)
        origin (`str`): where the origin of the array is (`upper`= imshow default, yaxis faces down; `lower`=pcolormesh, axis in standard arrangement). (Default: `upper`)
    Returns:
        np.ndarray: Projected image in azimuthal equidistant coordinates.
    """
    copies = {}
    img = fits_obj[FITSINDEX].data
    copies["1.raw"] = img.copy()
    cml, is_south = fitsheader(fits_obj, "CML", "south")
    if kwargs.get("south") is not None:
        is_south = kwargs.get("south")

    n_theta, n_phi = img.shape  # Grid size of input (θ, φ)
    output_size = n_theta  # Output image size (square)
    # cml=90 #if long0_at_0 else 0
    # Convert longitude limits to radians
    corrected_cml = (-cml) % 360
    lon_clip = [(cml - 90) % 360, (cml + 90) % 360]

    if not is_south:
        img = np.flip(img, axis=0)  # flip to have north pole at 0 index (image data is south at 0)
    copies["2.flipped"] = img.copy()

    # Compute the pixel corresponding to the longitude
    cml_pixel = int(corrected_cml / 360 * n_phi)
    lon_clip_pixel = [int(i / 360 * n_phi) for i in lon_clip]
    # Annotate the line vertically at the correct longitude
    img[:, cml_pixel] = 1e7  # You can set to NaN or a specific value for the line
    copies["3. cml annotated"] = img.copy()

    lon_mask = np.zeros_like(img, dtype=bool)
    lon_mask[:, lon_clip_pixel[0] : lon_clip_pixel[1]] = True
    copies["4. clipped +/- 90 cml"] = img.copy()
    if origin == "upper":
        sh = n_phi // 4
    else:
        sh = -n_phi // 4
        img = np.flip(img, axis=1)
        img = np.roll(img, shift=n_phi // 2, axis=1)
    # else:
    #

    p = meridian_pos.lower()
    if p.startswith("t") or p.startswith("n"):
        img = np.roll(img, shift=2 * sh, axis=1)  # roll to have 0° long at top (0 indexed)
    elif p.startswith("b") or p.startswith("s"):
        pass  # img = np.roll(img, shift=N_phi//2, axis=1)
    elif p.startswith("l") or p.startswith("w"):
        img = np.roll(img, shift=3 * sh, axis=1)
    elif p.startswith("r") or p.startswith("e"):
        img = np.roll(img, shift=sh, axis=1)

    copies["5. rolled"] = img.copy()

    theta_lim_rad = np.radians(rlim)
    # Create a coordinate grid for the output image
    y, x = np.meshgrid(np.linspace(-1, 1, output_size), np.linspace(-1, 1, output_size))
    r = np.sqrt(x**2 + y**2)  # Radial distance from center
    mask = r <= 1  # Valid points inside the projection circle

    theta = np.arcsin(r * np.sin(theta_lim_rad)) * (n_theta - 1) / np.pi  # Scale to input grid

    # theta = np.arcsin(r) * (N_theta - 1) / np.pi  # Scale to input grid
    # theta = theta * (N_theta - 1) / np.pi  # Scale to input grid

    # Compute longitude φ
    phi = np.arctan2(y, x)  # Angle in radians
    phi[phi < 0] += 2 * np.pi  # Convert to [0, 2π]
    phi = phi / (2 * np.pi) * (n_phi - 1)  # Scale to input grid indices
    # if not flip_for_pcolormesh:
    #       phi = np.flip(phi)
    # if lon_min_rad <= lon_max_rad:
    #     lon_mask = (phi >= lon_min_rad) & (phi <= lon_max_rad)
    # # elif lon_min_rad == lon_max_rad:
    # #       lon_mask = phi == phi
    # else:
    #     lon_mask = (phi >= lon_min_rad) | (phi <= lon_max_rad)  # Wrap-around case
    # mask &= lon_mask  # Combine with existing mask

    # Create empty projected image
    projected_array = np.full((output_size, output_size), np.nan)

    # Nearest-neighbor interpolation
    theta_idx = np.clip(np.round(theta).astype(int), 0, n_theta - 1)
    phi_idx = np.clip(np.round(phi).astype(int), 0, n_phi - 1)

    # Fill projected array
    # projected_array[:] = img[theta_idx, phi_idx]
    # # Fill the projected array using the computed indices
    projected_array[mask] = img[theta_idx[mask], phi_idx[mask]]
    # copies = {}
    copies["6. projected"] = projected_array.copy()
    if clip_negatives:
        projected_array = np.clip(projected_array, 0, None)
        copies["7. clipped <0"] = projected_array.copy()
    # Normalize longitude to [0, 360]

    # Plot result
    # ncols = int(np.ceil(np.sqrt(len(copies))))
    # nrows = int(np.ceil(len(copies) / ncols))
    # fig,axs=plt.subplots(nrows,ncols,figsize=(ncols*3, nrows*3))
    # axs = axs.flatten()
    # for i,(text,array) in enumerate(copies.items()):
    #     axs[i].pcolormesh(array,cmap='viridis', norm=LogNorm(vmin=1,vmax=3000),)
    #     #axs[i].axis("off")
    #     axs[i].set(xticks=[],yticks=[])
    #     axs[i].set_title(f'{text}',fontsize=10, pad=0)
    # plt.show()

    return projected_array, copies


##########################################################################################################
#                             IMAGE ARRAY TRANSFORMS
##########################################################################################################
def gaussian_blur(
    input_arr: np.ndarray, radius: int, amount: float, boundary: str = "wrap", mode: str = "same",
) -> np.ndarray:
    """Return the input array convolved with the kernel2d convolution kernel.
    radius = pixel radius of the kernel
    amount = standard deviation of the gaussian kernel
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
    Gx = [[-1  0  1]    Gy = [[-1 -2 -1]                      G= [[-1-1j 0-2j 1-1j]
          [-2  0  2]          [ 0  0  0]  ==> G = Gx + j*Gy =     [-2    0     2  ]
          [-1  0  1]]         [ 1  2  1]]                         [-1+1j 0+2j  1+1j]
    """
    complexret = convolve2d(input_arr, kernel2d, mode=mode, boundary=boundary)
    return np.abs(complexret)


def dropthreshold(input_arr: np.ndarray, threshold: float) -> np.ndarray:
    """Return a the input array, but with valuesbelow the threshold set to 0."""
    return np.where(input_arr < threshold, 0, input_arr)


def coadd(input_arrs: list[np.ndarray], weights: list[float] = None) -> np.ndarray:
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
