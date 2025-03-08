#!/usr/bin/env python3
"""
This module contains the headless functions to generate contour paths from fits files via openCV.
"""

import os
import random
from typing import List, Tuple, Union
from warnings import warn

import cmasher as cmr
import cv2
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

from .const import DPR_IMXY, FITSINDEX
from .polar import plot_polar, prep_polarfits
from .transforms import coadd, fullxy_to_polar_arr, gaussian_blur
from .utils import adapted_hdul, assign_params, filename_from_path, fitsheader, fpath, hst_fpath_list, mcolor_to_lum


def crop_to_axes(inpath: str, outpath: str = None, img_background: int = 0) -> Union[None, np.ndarray]:
    """Crop an image to the bounding box of the non 'image_background' pixels.
    inpath: path to the image to crop
    outpath: path to save the cropped image
    ax_background: the luminance value of the axis background
    MUST OUTPUT A SQUARE IMAGE, CENTRED ON THE ORIGIN OF THE CIRCLE IN THE IMAGE
    """
    warn(
        "This method is deprecated. polar array generation can be done using transforms.azimuthal_equidistant.",
        category=DeprecationWarning(),
    )

    img = cv2.imread(inpath, cv2.IMREAD_GRAYSCALE)
    # Convert to grayscale
    # Threshold the image to get the circle
    _, thresh = cv2.threshold(img, img_background + 1, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get the largest contour which should be the circle
    contour = max(contours, key=cv2.contourArea)
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    # Determine the size of the square
    size = max(w, h)
    # Calculate the center of the bounding box
    center_x, center_y = x + w // 2, y + h // 2
    # Calculate the top-left corner of the square
    start_x = max(center_x - size // 2, 0)
    start_y = max(center_y - size // 2, 0)
    # Crop the image to the square region
    img = img[start_y : start_y + size, start_x : start_x + size]
    # return cropped_image
    if outpath is not None:
        cv2.imwrite(outpath, img)
        return None
    return img


def mask_top_stripe(
    img: np.ndarray, threshold: float = 0.01, trimfrac: float = 0.3, bg_color: int = 0, facecolor: int = 255,
) -> np.ndarray:
    """Makes a mask to remove the top stripe of the image."""
    warn(
        "This method is deprecated. polar array generation can be done using transforms.azimuthal_equidistant.",
        category=DeprecationWarning(),
    )
    bg_color, facecolor = (i if isinstance(i, int) else mcolor_to_lum(i) for i in [bg_color, facecolor])
    imwidth, imheight = img.shape
    trimmedimg = img[:, int(trimfrac * imwidth) : int((1 - trimfrac) * imwidth)]
    # get the first row containing a non-white or black value#
    for i in range(imheight):
        # if more than 1% of the row is not white or black, we assume it is data
        if np.sum(np.logical_and(trimmedimg[i] != bg_color, trimmedimg[i] != facecolor)) > threshold * imwidth:
            datai = i

            break
    else:
        return img
    # make a mask where all values below datai row are allowed, and above are excluded
    mask = np.ones_like(img) * 255
    mask[:datai, :] = 0
    return cv2.bitwise_and(img, mask)


def imagexy(
    fits_obj: fits.HDUList,
    ax_background: str = "white",
    img_background: str = "black",
    crop: bool = True,
    cmap: cmr.Colormap = cmr.neutral,
    dpi: int = 300,
    **kwargs,
) -> np.ndarray:
    """Generate a stripped down, grey scale image of the fits file.
    Args:
    fits_obj: the fits object to generate the image
    ax_background: the background color of the axis
    img_background: the background color of the image
    crop: whether to crop the image to the bounding box of the circle
    cmap: the colormap to use
    dpi: the resolution of the image
    kwargs: additional arguments to pass to plot_polar
    Returns:
    np.ndarray: the image as a numpy array
    """
    warn("This method is deprecated, replaced by transforms.azimuthal_equidistant.", category=DeprecationWarning())
    # ~ replaced because using this method limits us to a resolution of 1/256 (16bit) over the normalized range
    full = fitsheader(fits_obj, "full")
    # do normal plotting with no decorations, just the circle
    fig = plt.figure(figsize=(6, 6 if full else 3), dpi=dpi, layout="none")
    ax = fig.subplots(1, 1, subplot_kw={"projection": "polar"})
    plot_polar(fits_obj, ax, cmap=cmap, nodec=kwargs.pop("nodec", True), **kwargs)

    ax.set(facecolor=ax_background)
    fig.patch.set_facecolor(img_background)
    ax.set_aspect(1)
    ax.set_position([0, -0.45, 1, 1.8] if not full else [0.05, 0.05, 0.9, 0.9])
    for spine in ax.spines.values():
        spine.set_linewidth(0)
    tempdir = fpath(r"temp/" + f"temp_stripped_{random.randint(0,99999999):0<8}.png")
    fig.savefig(tempdir, dpi=dpi)
    plt.close()
    img = (
        crop_to_axes(tempdir, ax_background=ax_background, img_background=ax_background)
        if crop
        else cv2.imread(tempdir)
    )
    os.remove(tempdir)
    cv2.imwrite("temp/temp_Crop.png", img)
    return img


def generate_coadded_fits(fits_objs, saveto=None, kernel_params=(3, 1), overwrite=True, indiv=True, coadded=True):
    """Generate a coadded fits file from a list of fits files.
    Args:
    fits_objs: a list of fits objects to coadd
    saveto: the path to save the coadded fits file
    gaussian: the parameters for the gaussian blur
    overwrite: whether to overwrite the file if it already exists
    indiv: whether to blur the individual fits files
    coadded: whether to blur the coadded fits file
    Returns:
    fits.HDUList: the coadded fits file
    """
    fdatas = [fits_objs[i][1].data for i in range(len(fits_objs))]
    if saveto == "auto":
        saveto = fpath(f"datasets/HST/custom/{filename_from_path(hst_fpath_list)}_coadded_gaussian{kernel_params}.fits")
    fdatas = [gaussian_blur(fd, *kernel_params) for fd in fdatas] if indiv else fdatas
    coaddg = coadd(fdatas)
    coaddg = gaussian_blur(coaddg, 3, 1) if coadded else coaddg
    cofitsd = adapted_hdul(fits_objs[0], new_data=coaddg)
    if saveto is not None:
        # ensure_dir(saveto)
        cofitsd.writeto(saveto, overwrite=overwrite)
    return cofitsd


def generate_rollings(fits_objs, window=3, kernel_params=(3, 1), indiv=True, coadded=True, preserve="array"):
    """Generate rolling coadded fits files from a list of fits files.
    given N inputs and a window size W,
    - mode: preserve='array' will preserve the number of inputs to match the number of outputs
    - mode: preserve='window' will preserve the window size, giving null outputs at the start and end of the list
    the window will include equal number of earlier and later inputs, with the current input in the middle.
    ie Nearlier = W//2, Nlater= W//2 - W%2
    if an even window is given, Nearlier = W//2, Nlater = W//2 - 1
    Args:
    fits_objs: a list of fits objects to coadd
    window: the number of fits files to coadd
    kernel_params: the parameters for the gaussian blur
    indiv: whether to blur the individual fits files
    coadded: whether to blur the coadded fits file
    preserve: whether to preserve the array length or the window size
    Returns:
    List[fits.HDUList]: the coadded fits files
    """
    fdatas = np.stack([fits_objs[i][FITSINDEX].data for i in range(len(fits_objs))])
    # print(f"{fdatas.shape=}")
    if indiv:
        fdatas = [gaussian_blur(fdatas[i], *kernel_params) for i in range(len(fdatas))]

    if preserve == "array":
        result = np.zeros_like(fdatas) * np.nan
        for i in range(len(fdatas)):
            result[i] = coadd(fdatas[max(0, i - window // 2) : min(len(fdatas), i + window // 2 + 1)])
    elif preserve == "window":
        result = np.zeros_like(fdatas) * np.nan
        for i in range(len(fdatas)):
            if i < window // 2 or i > len(fdatas) - window // 2:
                continue
            result[i] = coadd(fdatas[i - window // 2 : i + window // 2 + 1])
            # print(f"{result[i].shape=}")
            # sample_indices = [j for j in range(i-window//2, i+window//2+1)]
            # print(f"{sample_indices=}, {len(sample_indices)=}, {len(fdatas)=}")
    else:
        raise ValueError("Invalid value for preserve")
    if coadded:
        result = [gaussian_blur(result[i], *kernel_params) for i in range(len(result))]
    return [adapted_hdul(fits_objs[i], new_data=result[i]) for i in range(len(result))]
    # print(f"{len(result)=}, {len(result[0])=}")


def generate_contours(
    fits_obj: fits.HDUList,
    lrange: Tuple[float, float] = (0.2, 0.4),
    morphex: Tuple[int] = (cv2.MORPH_CLOSE, cv2.MORPH_OPEN),
    fcmode: int = cv2.RETR_EXTERNAL,
    fcmethod: int = cv2.CHAIN_APPROX_SIMPLE,
    cvh: bool = False,
) -> List[List[float]]:
    """Generate contours from a fits file.
    Args:
    fits_obj: the fits object to generate the contours from
    lrange: the luminance range to use for the mask
    morphex: the morphological operations to use
    fcmode: the mode to use for finding contours
    fcmethod: the method to use for finding contours
    cvh: whether to use convex hulls
    Returns:
    List[List[float]]: the contours in image coordinates"""
    # generate a stripped down, grey scale image of the fits file
    proc = prep_polarfits(assign_params(fits_obj, fixed="LON", full=True))
    img = imagexy(proc, cmap=cmr.neutral, ax_background="white", img_background="black")
    # if a pixel is not provided, ask the user to provide one, and show the image
    # normalize image
    normed = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    # create a mask of the image using the provided luminance range
    mask = cv2.inRange(normed, lrange[0] * 255, lrange[1] * 255)
    # smooth out the mask to remove noise
    kernel = np.ones((5, 5), np.uint8)  # Small kernel to smooth edges
    for morph in morphex:  # cv2.MORPH_CLOSE (Fill small holes) cv2.MORPH_OPEN (Remove small noise)
        mask = cv2.morphologyEx(mask, morph, kernel)
    # find the contours of the mask
    contours, hierarchy = cv2.findContours(image=mask, mode=fcmode, method=fcmethod)
    if cvh:
        contours = [cv2.convexHull(cnt) for cnt in contours]  # Convex hull of the contours
    # Find the contour that encloses the given point
    return contours, hierarchy, img


def identify_contour(
    contours: List[np.ndarray], hierarchy: dict, img: np.ndarray, id_pixel: List[int, int] = None,
) -> np.ndarray:  # noqa: ARG001
    """Identify the contour that contains a given pixel. will open a plot if no pixel is provided.
    Args:
    contours: the contours to search
    hierarchy: the hierarchy of the contours
    img: the image the contours were generated from
    id_pixel: the pixel to search for
    Returns:
    List[float]: the contour in polar coordinates"""

    if id_pixel is None or isinstance(id_pixel, str):

        def on_click(event):
            global click_coords
            if event.xdata is not None and event.ydata is not None:
                click_coords = (event.xdata, event.ydata)
                plt.close()

        fig, ax = plt.subplots()
        ax.imshow(img, cmap=cmr.neutral)
        ax.set_title("Click on a point")
        event_connection = fig.canvas.mpl_connect("button_press_event", on_click)  # noqa: F841
        plt.show()
        id_pixel = click_coords
    selected_contour = None
    for contour in contours:
        if cv2.pointPolygonTest(contour, id_pixel, False) > 0:  # >0 means inside
            selected_contour = contour
            break  # Stop searching once we find the correct contour
    # if a contour is found, convert the contour points to polar coordinates and return them
    if selected_contour is not None:
        paths = selected_contour.reshape(-1, 2)
        return fullxy_to_polar_arr(paths, img, 40)
    raise ValueError("No contour found for the selected pixel at the given luminance range.")


def plot_contourpoints(clist):
    """Plot the contour points in both polar and cartesian coordinates.
    Args:
    clist: the list of contours to plot (in polar coordinates)"""
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, polar=True)
    ax2 = fig.add_subplot(122)
    for c in clist:
        ax.scatter(np.radians(c[:, 1]), [c[:, 0]], s=1)
        ax2.scatter(c[:, 1], c[:, 0], s=1)

    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    ax2.invert_yaxis()
    ax2.invert_xaxis()
    plt.show()


def find_contour_basic(
    fits_obj: fits.HDUList, id_pixel: List[int, int] = None, lrange: List[float, float] = (0.2, 0.4),
) -> np.ndarray:
    """Basic convenience function to generate contours and identify a contour from a fits file.
    Args:
    fits_obj: the fits object to generate the contours
    id_pixel: the pixel to identify the contour for
    lrange: the luminance range to use for the mask
    Returns:
    List[float]: the contour in polar coordinates"""
    contours, hierarchy, img = generate_contours(fits_obj, lrange)
    return identify_contour(contours, hierarchy, img, id_pixel)


def pathtest(lrange=(0.25, 0.35)):
    """Headless generation of contour points for testing.
    Args:
    lrange: the luminance range to use for the mask
    Returns:
    List[float]: the contour in polar coordinates"""
    test = fits.open(fpath(r"datasets/HST/custom/v04_coadded_gaussian[3_1].fits"))
    clist = [find_contour_basic(test, DPR_IMXY["04"], lrange)]
    plot_contourpoints(clist)
    return clist[0]


def save_contour(fits_obj: fits.HDUList, cont: np.ndarray, index: Union[int, str] = None) -> fits.HDUList:
    """Save a contour to a fits file.
    Args:
    fits_obj: the fits object to save the contour to
    cont: the contour to save
    index: the index to save the contour at
    Returns:
    fits.HDUList: the fits object with the contour added"""
    # if no index given, use the next available index
    if index is None:
        index = len(fits_obj)
    table = Table(data=cont, names=["colat", "lon"])
    # find any existing boundary HDUs
    ver = 0
    for i, hdu in enumerate(fits_obj):
        if hdu.name == "BOUNDARY":
            ver = max(ver, hdu.ver)
    ver += 1
    newtablehdu = fits.BinTableHDU(table, name="BOUNDARY", ver=ver)
    curr_hdus = [fits_obj[i] for i in range(len(fits_obj))]
    curr_hdus.insert(index, newtablehdu)
    return fits.HDUList(curr_hdus)


def contourhdu(cont: np.ndarray, name: str = "BOUNDARY", header: fits.Header = None, **kwargs) -> fits.BinTableHDU:
    """Generate a fits BinTableHDU from a contour.
    Args:
    cont: the contour to generate the HDU from
    name: the name of the HDU
    header: the header to use
    Returns:
    fits.BinTableHDU: the HDU generated from the contour"""
    kwargs.setdefault("uint", True)
    table = Table(data=cont, names=["colat", "lon"], dtype=[np.float32, np.float32])
    binhdu = fits.BinTableHDU(table, name=name, header=header, **kwargs)
    binhdu.update_header()
    binhdu.add_checksum()
    binhdu.verify()
    return binhdu


def contourid(fits_obj: fits.HDUList, name: str = None) -> str:
    """Find the ID of a contour in a fits file.
    Args:
    fits_obj: the fits object to search
    name: the name of the contour to search for
    Returns:
    str: the name of the contour"""
    if name is None:
        name = "BOUNDARY"
        if len(fits_obj) > 2:
            return 2
    for hdu in fits_obj:
        if hdu.name == name:
            return name
    else:
        raise ValueError(f"No contour found with the name {name}, and no default contour found.")


def getcontour(fits_obj: fits.HDUList, name: Union[str, int] = None) -> np.ndarray:
    """Get the contour from a fits file.
    Args:
    fits_obj: the fits object to get the contour from
    name: the name of the contour to get
    Returns:
    np.ndarray
    """
    name = contourid(fits_obj, name)
    return np.array(fits_obj[name].data.tolist())
