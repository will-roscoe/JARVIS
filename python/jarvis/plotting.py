#!/usr/bin/env python3
"""Module containing functions to generate polar projection plots of Jupiter's image data from FITS files.

The main function, moind(), generates a polar projection plot of Jupiter's image data from a FITS file.
The make_gif() function creates a GIF from a directory of FITS files.

adapted from dmoral's original code by the JAR:VIS team.
"""

# python standard libraries
import contextlib
import datetime
import os
from datetime import timedelta, timezone
from glob import glob

import matplotlib as mpl
import numpy as np
import pandas as pd

# third party libraries
from astropy.io.fits import HDUList
from astropy.timeseries import TimeSeries
from dateutil.parser import parse
from fastgif import make_gif as makefastgif
from imageio import imread, mimsave
from matplotlib import pyplot as plt
from matplotlib import rc_file
from matplotlib.colors import LogNorm
from matplotlib.dates import AutoDateLocator, DayLocator, MinuteLocator, num2date
from matplotlib.patheffects import withStroke
from matplotlib.projections.polar import PolarAxes
from matplotlib.ticker import FuncFormatter
from pandas import DataFrame, Series
from tqdm import tqdm

# local modules
from .const import CONST, FITSINDEX, HISAKI, HST, plot
from .reading_mfp import moonfploc
from .utils import (
    adapted_hdul,
    approx_grid_dims,
    assign_params,
    clock_format,
    ensure_dir,
    filename_from_hdul,
    fits_from_glob,
    fitsheader,
    fpath,
    get_datetime,
    get_obs_interval,
    get_time_interval_from_multi,
    group_to_visit,
    hist2xy,
    hst_fpath_list,
    merge_dfs,
)


def prep_polarfits(fitsobj: HDUList) -> HDUList:
    """Process a FITS file to apply transformations for polar plotting of a hemisphere. Use header information to adjust the image data.

    Args:
        fitsobj (HDUList): The FITS file object to be processed.

    Returns:
        HDUList: The processed FITS file object with updated data.
    The function performs the following steps:
    1. Extracts header information such as CML, DECE, EXPT, and south.
    2. Parses the start time from the header.
    3. Calculates light travel time based on the distance from the origin.
    4. Computes the start, end, and mid-exposure times adjusted for Jupiter.
    5. Generates latitude and longitude bins for the image data.
    6. Creates a mask based on the latitude and longitude bins.
    7. Applies the mask to the image data to filter out invalid regions.
    8. Returns a new FITS file object with the processed image data.

    """
    cml, dece, exp_time, is_south = fitsheader(fitsobj, "CML", "DECE", "EXPT", "south")
    is_lon = fitsheader(fitsobj, "fixed_lon")
    start_time = parse(fitsheader(fitsobj, "UDATE"))
    try:
        dist_org = fitsheader(fitsobj, "DIST_ORG")
        ltime = dist_org * 1.495979e11 / 299792458
        lighttime = timedelta(seconds=ltime)
    except ValueError:
        lighttime = timedelta(seconds=2524.42)
    exposure = timedelta(seconds=exp_time)
    start_time_jup = start_time - lighttime
    end_time_jup = start_time_jup + exposure  # noqa: F841
    mid_ex_jup = start_time_jup + (exposure / 2.0)  # noqa: F841
    image_data = fitsobj[FITSINDEX].data
    latbins = np.radians(np.linspace(-90, 90, num=image_data.shape[0]))
    lonbins = np.radians(np.linspace(0, 360, num=image_data.shape[1]))
    mask = np.zeros((int(image_data.shape[0]), image_data.shape[1]))
    cmlr = np.radians(cml)
    dec = np.radians(dece)
    for i in range(0, mask.shape[0]):
        mask[i, :] = np.sin(latbins[i]) * np.sin(dec) + np.cos(latbins[i]) * np.cos(dec) * np.cos(lonbins - cmlr)
    mask = np.flip(mask, axis=1)
    cliplim = np.cos(np.radians(89))
    clipind = np.squeeze([mask >= cliplim])
    image_data[clipind == False] = np.nan  # noqa: E712
    return adapted_hdul(fitsobj, new_data=image_data, FIXED="LT" if not is_lon else "LON")


def plot_polar(fitsobj: HDUList, ax: PolarAxes, **kwargs) -> PolarAxes:
    """Plot a polar projection of the given FITS object data.

    Args:
    fitsobj (HDUList): The FITS file object containing the data to be plotted.
    ax (matplotlib.axes._subplots.PolarAxesSubplot): The matplotlib axis object to plot on.
    **kwargs: Additional keyword arguments for customization.

    Keyword Arguments:
    title (str): Title for the plot.
    suptitle (str): Supertitle for the plot.
    ticks (list): List of tick values for the colorbar.
    cmap (str): Colormap to be used for the plot.
    norm (matplotlib.colors.Normalize): Normalization for the colormap.
    shrink (float): Shrink factor for the colorbar.
    pad (float): Padding for the colorbar.
    draw_cbar, draw_grid, draw_ticks, ax_params, ml, hemis (bool): Flags to draw the colorbar, grid, ticks, axis parameters, meridian lines, and hemisphere text. all default to True.

    Returns:
    matplotlib.axes._subplots.PolarAxesSubplot: The axis object with the plot.

    """
    image_data = fitsobj[FITSINDEX].data
    cml, is_south, fixed_lon, crop, full, rlim = fitsheader(
        fitsobj, "CML", "south", "fixed_lon", "CROP", "FULL", "RLIM",
    )
    rlim = 40
    crop = 1

    if kwargs.pop("nodec", False):
        kwargs.update(
            {
                "draw_cbar": False,
                "draw_grid": False,
                "draw_ticks": False,
                "ax_params": False,
                "ml": False,
                "hemis": False,
                "title": False,
                "cax": True,
            },
        )
    draw_ticks,  = (
        kwargs.pop("draw_ticks", True),
    )
    ax.set(
        **{
            "theta_zero_location": "N",
            "facecolor": "k",
            "rlabel_position": 0,
            "thetalim": [np.pi / 2, 3 * np.pi / 2] if not full else None,
            "rlim": [0, rlim],
            "rgrids": np.arange(0, rlim, 10, dtype="int"),
        },
    )  # set the polar plot
    ax.yaxis.set(
        **{
            "major_locator": plt.MultipleLocator(base=10),
            "major_formatter": plt.FuncFormatter(lambda x, _: f"{x:.0f}°"),
        }
        if draw_ticks
        else {"visible": False},
    )  # set radial ticks
    ax.yaxis.set_tick_params(**{"labelcolor": "white"} if draw_ticks else {})  # set radial ticks
    ax.xaxis.set(
        **{
            "major_locator": plt.MultipleLocator(base=np.pi / 2 if full else np.pi / 4),
            "major_formatter": plt.FuncFormatter(
                lambda x, _: f"{np.degrees((lambda x: x if is_south else 2*np.pi-x)(x))%360:.0f}°"
                if fixed_lon
                else clock_format(x),
            ),
            "minor_locator": plt.MultipleLocator(base=np.pi / 18 if fixed_lon else np.pi / 12),
        }
        if draw_ticks
        else {"visible": False},
    )
    ax.tick_params(axis="both", pad=2.0)  # shift position of LT labels
    # Titles
    tkw, stkw = kwargs.pop("title", True), kwargs.pop("suptitle", True)
    if tkw:
        t_ = {
            "suptitle": f'Visit {fitsobj[1].header["VISIT"]} (DOY: {fitsobj[1].header["DOY"]}/{fitsobj[1].header["YEAR"]}, {get_datetime(fitsobj)})',
            "title": f'{"Fixed LT. " if not fixed_lon else ""}Integration time={fitsobj[1].header["EXPT"]} s. CML: {np.round(cml, decimals=1)}°',
        }
        t_["title"] = tkw if isinstance(tkw, str) else t_["title"]
        t_["suptitle"] = stkw if isinstance(stkw, str) else t_["suptitle"]
        parentfig = ax.get_figure()
        parentfig.suptitle(t_["suptitle"], y=0.99, fontsize=14)  # one of the two titles for every plot
        ax.set_title(t_["title"], y=1.05 if full else 1.03 if not fixed_lon else 1.02, fontsize=12)
    if kwargs.pop("ml", True):
        if fixed_lon:  # plotting cml, only for lon
            rot = 180 if is_south else 360
            ax.plot(np.roll([np.radians(rot - cml), np.radians(rot - cml)], 180 * 4), [0, 180], "r--", lw=1.2)  # cml
            ax.text(
                np.radians(rot - cml),
                3 + rlim,
                "CML",
                fontsize=11,
                color="r",
                ha="center",
                va="center",
                fontweight="bold",
            )
        if not fixed_lon and full:  # meridian line (0°)
            ax.text(
                np.radians(cml) + np.pi,
                4 + rlim,
                "0°",
                color="coral",
                fontsize=12,
                ha="center",
                va="bottom",
                fontweight="bold",
            )
            ax.plot(
                [np.radians(cml) + np.pi, np.radians(cml) + np.pi],
                [0, 180],
                color="coral",
                path_effects=[withStroke(linewidth=1, foreground="black")],
                linestyle="-.",
                lw=1,
            )  # prime meridian (longitude 0)
        # Actual plot and colorbar (change the vmin and vmax to play with the limits
    # of the colorbars, recommended to enhance/saturate certain features)
    ticks = kwargs.pop(
        "ticks",
        [10.0, 40.0, 100.0, 200.0, 400.0, 800.0, 1500.0]
        if int(fitsobj[1].header["EXPT"]) < 30
        else [10.0, 40.0, 100.0, 200.0, 400.0, 1000.0, 3000.0],
    )
    kwd = {"cmap": "viridis", "norm": LogNorm(vmin=ticks[0], vmax=ticks[-1]), "shrink": 1 if full else 0.75, "pad": 0.06}
    cmap, norm, shrink, pad = (kwargs.pop(k, v) for k, v in kwd.items())
    rho = np.linspace(0, 180, num=int(image_data.shape[0]))
    theta = np.linspace(0, 2 * np.pi, num=image_data.shape[1])
    image_centred = image_data if fixed_lon else np.roll(image_data, int(cml - 180.0) * 4, axis=1)  # shifting the image to have CML pointing southwards in the image
    corte = np.flip(image_centred, 0)[: (int((image_data.shape[0]) / crop)), :]

    if is_south:
        rho = rho[::-1]
        corte = np.roll(corte, 180 * 4, axis=1)
    cmesh = ax.pcolormesh(
        theta, rho[: (int((image_data.shape[0]) / crop))], corte, norm=norm, cmap=cmap,
    )  # ~ <- Color of the plot

    if kwargs.pop("draw_cbar", True):
        cbar = plt.colorbar(ticks=ticks, shrink=shrink, pad=pad, ax=ax, mappable=cmesh)
        cbar.ax.set_yticklabels([str(int(i)) for i in ticks])
        cbar.ax.set_ylabel("Intensity [kR]", rotation=270.0)
    #-Grids: [major & minor]
    if not kwargs.pop("draw_grid", True):
        ax.grid(False, which="both")
    else:
        ax.grid(True, which="major", color="w", alpha=0.7, linestyle="-")
        ax.minorticks_on()
        ax.grid(True, which="minor", color="w", alpha=0.5, linestyle="-")
        # stronger meridional lines for the 0, 90, 180, 270 degrees:
        for i in range(0, 4):
            ax.plot([np.radians(i * 90), np.radians(i * 90)], [0, 180], "w", lw=0.9)
    # deprecated variable (maybe useful for the South/LT fixed definition?)
    # print which hemisphere are we in:
    if kwargs.pop("hemis", True):
        ax.text(
            45 if full else 135 if any([not is_south, not fixed_lon]) else -135,
            1.3 * rlim,
            str(fitsheader(fitsobj, "HEMISPH")).capitalize(),
            fontsize=21,
            color="k",
            ha="center",
            va="center",
            fontweight="bold",
        )
    if kwargs.pop("cax", False):  # get tightest layout possible
        plt.tight_layout()
    return ax


def plot_moonfp(fitsobj: HDUList, ax: PolarAxes) -> PolarAxes:
    """Plot the footprints of moons on a given axis based on the provided FITS object and the moonfploc function.

    Args:
    fitsobj (HDUList): The FITS object containing the data.
    ax (mpl.projections.polar.PolarAxes): The matplotlib axis on which to plot the moon footprints.

    Returns:
    PolarAxes

    The function extracts relevant data from the FITS header, calculates the positions of the moons,
    and plots their footprints on the provided axis. It handles both northern and southern hemispheres
    and adjusts the plotting based on whether the longitude is fixed or not.

    """
    cml, is_south, fixed_lon = fitsheader(fitsobj, "CML", "south", "fixed_lon")
    lons = [fitsheader(fitsobj, f"IOLON{k}", f"EULON{k}", f"GALON{k}") for k in ("", 1, 2)]
    fp = np.array([moonfploc(*lon) for lon in lons])
    moon_list = [[[*fp[:, 4 * i], fp[0, 4 * i + 1]], [*fp[:, 4 * i + 2], fp[0, 4 * i + 3]]] for i in range(3)]
    # LT:
    moonrange = []  # empty list, see if the coordinates of the moons are in range
    for x in moon_list:
        if not is_south:
            moonrange.append(
                x[0],
            )  # appends north hemisphere data to moon range, only use north hemis: list of northern hemisphere data of moons to the list
        else:
            moonrange.append(x[1])  # only using south hemisphere
    # moonrange has all north or all south now
    if not fixed_lon:
        for i in range(3):  # for IO first index, EUR second index, GAN third index
            x = np.radians(180 + cml - moonrange[i][1])  # calculate the coordinates values, the second file
            y = np.radians(180 + cml - moonrange[i][2])  # third file
            w = np.radians(180 + cml - moonrange[i][0])  # first file
            v = moonrange[i][3]  # last file
            if (
                abs(cml - moonrange[i][1]) < 120 or abs(cml - moonrange[i][1]) > 240
            ):  # if the coordinates are in range of HST viewing, of each moon
                ax.plot([x, y], [v, v], "k-", lw=4)
                color, key = (("gold", "IO"), ("aquamarine", "EUR"), ("w", "GAN"))[i]
                ax.plot([x, y], [v, v], color=color, linestyle="-", lw=2.5)
                ax.text(
                    w,
                    3.5 + v,
                    key,
                    color=color,
                    fontsize=10,
                    alpha=0.5,
                    path_effects=[withStroke(linewidth=1, foreground="black")],
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontweight="bold",
                )
    if fixed_lon:
        if not is_south:  # process coordinates for north hemis of lon
            for i in range(3):  # for IO first index, EUR second index, GAN third index
                x = 2 * np.pi - (np.radians(moonrange[i][1]))  # calculate the coordinates values, the second file
                y = 2 * np.pi - (np.radians(moonrange[i][2]))  # third file
                w = 2 * np.pi - (np.radians(moonrange[i][0]))  # first file
                v = moonrange[i][3]  # last file
                if (
                    abs(cml - moonrange[i][1]) < 120 or abs(cml - moonrange[i][1]) > 240
                ):  # if the coordinates are in range of HST viewing, of each moon
                    ax.plot([x, y], [v, v], "k-", lw=4)
                    color, key = (("gold", "IO"), ("aquamarine", "EUR"), ("w", "GAN"))[i]
                    ax.plot([x, y], [v, v], color=color, linestyle="-", lw=2.5)
                    ax.text(
                        w,
                        3.5 + v,
                        key,
                        color=color,
                        fontsize=10,
                        alpha=0.5,
                        path_effects=[withStroke(linewidth=1, foreground="black")],
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontweight="bold",
                    )
        else:  # process coordinates for south hemis of lon
            for i in range(3):  # for IO first index, EUR second index, GAN third index
                x = np.radians(180 - moonrange[i][1])  # calculate the coordinates values, the second file
                y = np.radians(180 - moonrange[i][2])  # third file
                w = np.radians(180 - moonrange[i][0])  # first file
                v = moonrange[i][3]  # last file
                if (
                    abs(cml - moonrange[i][1]) < 120 or abs(cml - moonrange[i][1]) > 240
                ):  # if the coordinates are in range of HST viewing, of each moon
                    ax.plot([x, y], [v, v], "k-", lw=4)
                    color, key = (("gold", "IO"), ("aquamarine", "EUR"), ("w", "GAN"))[i]
                    ax.plot([x, y], [v, v], color=color, linestyle="-", lw=2.5)
                    ax.text(
                        w,
                        3.5 + v,
                        key,
                        color=color,
                        fontsize=10,
                        alpha=0.5,
                        path_effects=[withStroke(linewidth=1, foreground="black")],
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontweight="bold",
                    )

        # ORGINALLY LON N, range is 250 idk why :abs(cml-nlonga1) < 120 or abs(cml-nlonga1) > 250:
        # LT, S/N: (np.radians(180+cml-X1)),(np.radians(180+cml-X2)), TEXT np.radians(180+cml-X0)
        # LT N EUR: (2*np.pi-(np.radians(180+cml-X1)) (weird)
        # LON N: 2*np.pi-(np.radians(X1)),2*np.pi-(np.radians(X2) TEXT 2*np.pi-(np.radians(X0))
        # LON S: (np.radians(180-X1)),(np.radians(180-X2)), TEXT np.radians(180-X0)
    return ax


def plot_regions(fitsobj: HDUList, ax: PolarAxes) -> PolarAxes:
    """Plot various regions on a polar plot using the provided FITS object and matplotlib axis.

    Args:
    fitsobj (HDUList): The FITS object containing the data and header information.
    ax (mpl.projections.polar.PolarAxes): The matplotlib axis on which to plot the regions.

    The function plots the following regions:
    - Dusk boundary in red.
    - Dawn boundary in black or blue depending on the 'fixed_lon' header value.
    - Noon boundary in yellow.
    - Polar boundary in white dashed lines.

    """
    lon_fixed = fitsheader(fitsobj, "fixed_lon")
    updusk = np.linspace(np.radians(205), np.radians(170), 200)
    dawn = {
        k: np.linspace(v[0], v[1], 200)
        for k, v in (("lon", (np.radians(180), np.radians(130))), ("uplat", (33, 15)), ("downlat", (39, 23)))
    }
    noon_a = {
        k: np.linspace(v[0], v[1], 100) for k, v in (("lon", (np.radians(205), np.radians(190))), ("downlat", (28, 32)))
    }
    noon_b = {
        k: np.linspace(v[0], v[1], 100) for k, v in (("lon", (np.radians(190), np.radians(170))), ("downlat", (32, 27)))
    }
    regions = [
        [
            [[np.radians(205), np.radians(170)], [20, 10]],
            [[np.radians(170), np.radians(205)], [10, 20]],
            [updusk, 200 * [10]],
            [updusk, 200 * [20]],
        ],  # dusk boundary
        [
            [[np.radians(130), np.radians(130)], [23, 15]],
            [[np.radians(180), np.radians(180)], [33, 39]],
            [dawn["lon"], dawn["uplat"]],
            [dawn["lon"], dawn["downlat"]],
        ],  # dawn boundary
        [
            [[np.radians(205), np.radians(205)], [22, 28]],
            [[np.radians(170), np.radians(170)], [27, 22]],
            [noon_a["lon"], noon_a["downlat"]],
            [noon_b["lon"], noon_b["downlat"]],
        ],  # noon boundary
        [
            [[np.radians(205), np.radians(205)], [10, 28]],
            [[np.radians(170), np.radians(170)], [27, 10]],
            [noon_a["lon"], noon_a["downlat"]],
            [updusk, 200 * [10]],
        ],
    ]  # polar boundary
    for i, region in enumerate(regions):
        c = ["r-", "k-" if lon_fixed else "b-", "y-", "w--"][i]
        lw = [1.5, 1, 1.5, 1][i]
        for line in region:
            ax.plot(*line, c, lw=lw)
    return ax


def plot_boundaries(fits_obj: HDUList, ax: plt.Axes, fill=True):
    """Plot boundaries on a given axis based on the provided FITS object.

    Args:
        fits_obj (HDUList): The FITS object containing the boundary data.
        ax (mpl.axes._subplots.Axes): The matplotlib axis on which to plot the boundaries.
        fill (bool): Whether to fill the boundaries with color. Default is True.

    """
    for i in range(2, len(fits_obj) + 1):
        colat, lon = fits_obj["BOUNDARY"].data["colat"], fits_obj["BOUNDARY"].data["lon"]
        bound = ax.plot(lon, colat, linewidth=100, zorder=100, color="red", clip=False)
        if fill:
            ax.fill(colat, np.radians(360 - r for r in lon), alpha=0.1, color=bound[0].get_color(), zorder=98)
    return ax


def moind(
    fitsobj: HDUList,
    crop: float = 1,
    rlim: float = 40,
    fixed: str = "lon",
    full: bool = True,
    regions: bool = False,
    moonfp: bool = False,
    region: bool = True,
    **kwargs,
) -> tuple:
    """Process and plot a FITS file in polar coordinates.

    Args:
    fitsobj (HDUList): The FITS file object to be processed.
    crop (float, optional): The crop factor for the FITS file. Default is 1.
    rlim (float, optional): The radial limit for the plot. Default is 40.
    fixed (str, optional): The fixed parameter for the FITS file. Default is 'lon'.
    hemis (str, optional): The hemisphere to be plotted ('North' or 'South'). Default is 'North'.
    full (bool, optional): Whether to plot the full FITS file. Default is True.
    regions (bool, optional): Whether to plot regions on the FITS file. Default is False.
    moonfp (bool, optional): Whether to plot the moon footprint. Default is False.
    region (bool, optional): whether to plot any boundaries found in the HDUList.
    **kwargs: Additional keyword arguments for plotting.

    Returns:
    Union[None, mpl.figure.Figure]: The matplotlib figure and axis objects if successful, otherwise None.

    """
    fits_obj = prep_polarfits(
        assign_params(fitsobj, crop=crop, rlim=rlim, fixed=fixed, full=full, regions=regions, moonfp=moonfp),
    )
    fig = plt.figure(figsize=(7, 6))
    ax = plt.subplot(projection="polar")
    ax = plot_polar(fits_obj, ax, **kwargs)
    if regions and not fitsheader(fits_obj, "south"):
        ax = plot_regions(fits_obj, ax)
    if moonfp:
        ax = plot_moonfp(fits_obj, ax)
    if region:
        ax = plot_boundaries(fits_obj, ax)
    return fig, ax, fits_obj


def make_gif(fits_dir, fps=5, remove_temp=False, savelocation="auto", filename="auto", **kwargs):
    """Create a GIF from a directory of FITS files.

    Args:
    fits_dir (str): The directory containing the FITS files.
    fps (int, optional): The frames per second for the GIF. Default is 5.
    remove_temp (bool, optional): Whether to remove temporary files. Default is False.
    savelocation (str, optional): The directory to save the GIF. Default is 'auto'.
    filename (str, optional): The name of the GIF file. Default is 'auto'.
    **kwargs: Additional keyword arguments for the moind function.

    Returns:
    None

    The function reads all FITS files in the specified directory, generates polar plots using the moind function,
    and creates a GIF from the images. The GIF is saved in the specified location with the given filename.

    """
    fitslist, fnames = fits_from_glob(fits_dir, names=True)
    imagesgif = []
    with tqdm(total=len(fitslist)) as pb:
        for i, file in enumerate(fitslist):
            pb.set_postfix(file=fnames[i])
            fig, ax, f = moind(file, **kwargs)
            fig.savefig(fpath("temp/") + f"gifpart_{i}.jpg", dpi=300)
            plt.close(fig)
            pb.update(1)

            imagesgif.append(imread(fpath("temp/") + f"gifpart_{i}.jpg"))
            # saving the GIF
    if savelocation == "auto":
        savelocation = fpath("pictures/gifs/")
    ensure_dir(savelocation)
    if filename == "auto":
        filename = filename_from_hdul(f) + f"{len(fitslist)}fr_{fps}fps" + ".gif"
    mimsave(savelocation + filename, imagesgif, fps=fps)
    if remove_temp:
        for file in glob(fpath("temp/") + "*"):
            os.remove(file)
        os.rmdir(fpath("temp/"))


def makefast_gif(fitsobjs, initfunc=None, fps=5, showprogress=True, **kwargs):
    """Create a GIF from a list of FITS files using the fastgif module.

    Args:
    fitsobjs (list): A list of FITS file objects.
    initfunc (function, optional): The initialization function for the GIF. this function should take an index as an argument and return a figure object. Default is None, which uses the moind function.
    fps (int, optional): The frames per second for the GIF. Default is 5.
    showprogress (bool, optional): Whether to show the progress bar. Default is True.
    **kwargs: Additional keyword arguments for the moind function. If 'saveto' is provided, the GIF is saved to the specified location.

    Returns:
    None

    The function generates a GIF from the list of FITS files using the fastgif module. It uses the initialization function to create the figure objects for each frame.
    The GIF is saved to the specified location if 'saveto' is provided in the keyword arguments.

    """
    if initfunc is None:

        def initfunc(idx):
            fits_obj = prep_polarfits(assign_params(fitsobjs[idx], **kwargs.pop("fits", {})))
            fig = plt.figure(figsize=(7, 6))
            ax = plt.subplot(projection="polar")
            plot_polar(fits_obj, ax, **kwargs)
            if all([not fitsheader(fits_obj, "south"), fitsheader(fits_obj, "REGIONS")]):
                plot_regions(fits_obj, ax)
            if fitsheader(fits_obj, "MOONFP"):
                plot_moonfp(fits_obj, ax)
            return fig

    if "saveto" in kwargs:
        savelocation = kwargs.pop("saveto")
    else:
        savelocation = fpath("figures/gifs/") + filename_from_hdul(assign_params(fitsobjs[0], **kwargs)) + ".gif"
    makefastgif(
        initfunc,
        num_calls=len(fitsobjs),
        filename=savelocation,
        show_progress=showprogress,
        writer_kwargs={"duration": 1 / fps},
    )

def plot_discontinuous_angle(ax,ts, colname, threshold=180,  **kwargs):
    """Plot a time series with discontinuous angles on a given axis.

    Args:
    ax (plt.Axes): The matplotlib axis on which to plot the time series.
    ts (TimeSeries): The time series object containing the data.
    colname (str): The name of the column to plot.
    threshold (float, optional): The threshold for identifying discontinuities. Default is 180.
    **kwargs: Additional keyword arguments for the plot.
        - color (str): The color of the plot. Default is 'b'.
        - linestyle (str): The line style of the plot. Default is '-'.
        - linewidth (float): The line width of the plot. Default is 1.
        - label (str): The label for the plot. Default is None.
        - and various other keyword arguments for the plot function.

    """
    time = ts.time
    angles = ts[colname]    # Compute differences to identify discontinuities
    diffs = np.abs(np.diff(angles))
    discontinuities = np.where(diffs > threshold)[0]    # Split indices at discontinuities
    segments = np.split(np.arange(len(angles)), discontinuities + 1)
    for segment in segments:
        t_seg = time[segment]
        ang_seg = angles[segment]# Extend segment at both ends if possible
        if len(segment) > 1:
            t_extended = time[[segment[0], *segment, segment[-1]]]  # Keep as Time object
            ang_extended = np.array([ang_seg[0], *ang_seg, ang_seg[-1]])
        else:
            t_extended, ang_extended = t_seg, ang_seg
        ax.plot(t_extended.datetime64, ang_extended, **kwargs)
    ax.set_ylim([0, 360])
    ax.yaxis.set_major_locator(plt.MultipleLocator(90))


def datetime_to_doy(x, *_):
    """Convert a mpl datetime to day of year."""
    return num2date(x).strftime("%j")

def datetime_to_min(x, *_):
    """Convert a mpl datetime to minutes."""
    return num2date(x).strftime("%H:%M")
def apply_plot_defaults(ax:plt.Axes=None, fig: plt.Figure =  None, day_of_year=True,visits=True, **kwargs):
    """Apply default settings to a plot. should be called just before saving the plot.

    Args:
         ax (plt.Axes, optional): The axes to apply the settings to. Defaults to None.
         fig (plt.Figure, optional): The figure to apply the settings to. Defaults to None.
         day_of_year (bool, optional): Whether to format the x-axis as day of year. Defaults to True.
         time_interval (tuple, optional): The time interval to set the x-axis to. Defaults to None.
         visits (bool, optional): Whether to show visit numbers on the plot. Defaults to True.
         **kwargs:
            - time_interval (tuple, optional): The time interval to set the x-axis to. Defaults to None.
            - df_time_interval (DataFrame or TimeSeries, optional): data object to set the time interval to. Defaults to None.
            - day_interval (int, optional): The interval between day labels, if not specified, defaults to use AutoDateLocator
            - interval_length (float, optional): the length of the interval as a fraction of the time_interval. Defaults to 0.1.
            - y_fixedexp (int, optional): fixed exponent for y-axis. Defaults to None. if specified, will remove that exponent from the y-axis labels, and add to axis label unit if already set.

    """
    # Apply the style settings from the specified .mplstyle file
    rc_file(plot.defaults.rcfile)

    if fig is not None and ax is None:
        # define ax= list of child axes
        ax = []
        for a in fig.get_children():
            if isinstance(a, plt.Axes):
                ax.append(a)
    if not isinstance(ax, (list, tuple)):
        ax = [ax]

    #----text scaling ----#
    # scale the text size of each y axis label based on the bbox of the label and height of the axis. each label should fit within the axis height (at least or less)
    if fig is None:
        fig = ax[0].get_figure()
    fig.align_ylabels(ax)
    for a in ax:
        axpos = a.get_window_extent()
        axlabel = a.yaxis.label
        # ensure the bbox height is less than the axis height
        labelpos = axlabel.get_window_extent()
        height_ratio = labelpos.height/axpos.height
        if height_ratio > 1:
            a.set_ylabel(axlabel.get_text(), fontsize=a.yaxis.label.get_fontsize()/height_ratio)


    #---- X-axis ticks ----#
    adjust_annotations = False
    # day_of_year --> format x-axis as day of year [subargs: day_interval]
    if day_of_year:
        for a in ax:
            # figure out if x axis are shared, if so, only set the label on the last axis
            if a.get_shared_x_axes() is not None:
                a.set_xlabel("")
                a.xaxis.set_tick_params(bottom=True, top=True, direction="in", labeltop=False, labelbottom=False)
            else:
                a.set_xlabel(plot.defaults.DOY_label)
                a.xaxis.set_major_formatter(FuncFormatter(datetime_to_doy))
                a.xaxis.set_major_locator(DayLocator(interval=kwargs.get("day_interval")) if kwargs.get("day_interval") is not None else AutoDateLocator())
        # Ensure the lowest axis always has the label
        lowest_ax = min(ax, key=lambda a: a.get_position().y0)
        if not visits:
            lowest_ax.set_xlabel(plot.defaults.DOY_label)
            lowest_ax.xaxis.set_major_formatter(FuncFormatter(datetime_to_doy))
            lowest_ax.xaxis.set_major_locator(DayLocator(interval=kwargs.get("day_interval")) if kwargs.get("day_interval") is not None else AutoDateLocator())
        else:
            # dates at topmost axis, with label, with ticks at top
            topmost_ax = max(ax, key=lambda a: a.get_position().y1)
            topmost_ax.tick_params(axis="x",direction="in", labeltop=True, top=True, labelbottom=False)
            topmost_ax.set_xlabel(plot.defaults.DOY_label)
            topmost_ax.xaxis.set_label_position("top")

            topmost_ax.xaxis.set_major_formatter(FuncFormatter(datetime_to_doy))
            topmost_ax.xaxis.set_major_locator(DayLocator(interval=kwargs.get("day_interval")) if kwargs.get("day_interval") is not None else AutoDateLocator())
            # bottom axis has visit number at ticks corresponding to them.
            visit_list = visits if isinstance(visits, (list, tuple)) else None

            dlim_a = ax[0].get_xlim()
            dlim_ax = [mpl.dates.num2date(a) for a in dlim_a]
            spanlist = []
            for f in hst_fpath_list():
                mind, maxd = get_obs_interval(f)
                mind = pd.to_datetime(mind).replace(tzinfo=timezone.utc)
                maxd = pd.to_datetime(maxd).replace(tzinfo=timezone.utc)
                if mind >= dlim_ax[0] and maxd <= dlim_ax[1]:
                    v = group_to_visit(int(f.split("_")[-1][:2]))
                    spanlist.append([mind, maxd,v, (maxd-mind)/2 + mind])
            if len(spanlist) > 0:
                bottom_ax = min(ax, key=lambda a: a.get_position().y0) if len(ax) > 1 else ax[0]
                # set xaxis label to be at the bottom
                twinax = bottom_ax.twiny()
                twinax.set_xlabel("Visit", labelpad=14)
                twinax.xaxis.set_label_position("bottom")
                twinax.tick_params(axis="both",labeltop=False, labelbottom=False)
                for span in spanlist:
                    if visit_list is None or span[2] in visit_list:
                        for a in ax:
                            axy = a.get_ylim()
                            a.axvspan(span[0], span[1], color="#aaa", alpha=0.2)
                            a.set_ylim(axy)
                    else:
                        spanlist.remove(span)
                    # annotate the visitnumber just below the axes, for each span, like tick labels
                if kwargs.get("annotate", True):
                    anns = [bottom_ax.annotate(f"{int(tick[2])}", xy=(tick[3], 0),**plot.visit_annotate_kws,
                                            ) for i,tick in enumerate(spanlist)]
                    anns = sorted(anns, key=lambda a: a.xy[0])
                    adjust_annotations = True # need to adjust the annotations to avoid overlap, once the axes have been adjusted
                else:
                    twinax.set_xlabel("Day of Year")
                    bottom_ax.set_xlabel("")


    #---- X-axis bounds ----#
    # 1. time_interval --> set x-axis limits to specified interval
    if kwargs.get("time_interval") is not None:
        for a in ax:
            a.set_xlim(kwargs["time_interval"])
    # 2. df_time_interval --> set x-axis limits to the bounds of the input data. [subargs: interval_length]
    elif kwargs.get("df_time_interval") is not None:
        tdf = kwargs["df_time_interval"]
        if isinstance(tdf, (DataFrame, Series)):
            lims = [tdf.time.min(), tdf.time.max()]
        elif isinstance(tdf,TimeSeries):
            tdf = tdf.to_pandas()
            lims = [tdf.index.min(), tdf.index.max()]

        # set the axis limits based on the interval length: min = start - interval_length*(end-start), max = end + interval_length*(end-start)
        intlen,delta = kwargs.get("interval_length", 0.05), lims[1]-lims[0]
        lims = [lims[0] - intlen*delta, lims[1] + intlen*delta]
        for a in ax:
            a.set_xlim(lims)
    #---- Post-plot adjustments ----#
    if kwargs.get("compact") is not None:
        for a in ax:
            if kwargs.get("compact",{}).get("time","default") == "default":
                ff = FuncFormatter(datetime_to_min)
            else:
                init_time = ax[0].get_xlim()[0]
                def datetime_to_min_delta(x,_):
                    return datetime_to_min(x-init_time)
                ff = FuncFormatter(datetime_to_min_delta)
            a.set_xlabel("")
            a.xaxis.set_major_locator(MinuteLocator(interval=10))
            a.xaxis.set_major_formatter(ff)
            a.xaxis.set_tick_params(labelsize="small")
            # get leftmost axis (multiple if several axes are the same x position, ie in a grid)
        leftmost_ax = min(ax, key=lambda a: a.get_position().x0)
        all_leftmost = [a for a in ax if a.get_position().x0 == leftmost_ax.get_position().x0]
        for a in ax:
            if a not in all_leftmost:
                a.yaxis.set_tick_params(labelleft=False)
                a.yaxis.set_label_text("")
    if adjust_annotations:
        noolps = [False]*len(anns)
        count =0
        while not all(noolps) and count < 100:
            count+=1
            # if overap is found, shift annotaion acrros to align its edge with the edge of the previous annotation
            for i in range(len(anns)):
                # check if the annotation overlaps with the previous annotation
                if i>0:
                    iwin = anns[i].get_window_extent()
                    pwin = anns[i-1].get_window_extent()
                    if iwin.overlaps(pwin):
                        noolps[i] = False
                        anns[i].xyann = (anns[i].xyann[0]+0.1, anns[i].xyann[1])
                    else:
                        noolps[i] = True
                if i<len(anns)-1:
                    iwin = anns[i].get_window_extent()
                    nwin = anns[i+1].get_window_extent()
                    if iwin.overlaps(nwin):
                        noolps[i] = False
                        anns[i].xyann = (anns[i].xyann[0]-0.1, anns[i].xyann[1])
                    else:
                        noolps[i] = True
        else:
            if count > 100:
                tqdm.write(f"Visit number overlap resolution reached maximum iterations, continuing without full resolution. approx overlaps: {sum(noolps)}")

def get_axis_exp(lims,*args, e3_only=True):
    """Return a suggested exponent for the axis labels based on the range of the data.

    Args:
        lims (tuple): The axis limits.
        *args: if two args are passed, lower == lims, upper == args[0]
        e3_only (bool, optional): Whether to only allow exponents that are multiples of 3. Defaults to True.

    Returns:
        int: The suggested exponent.

    """
    if len(args) == 1:
        lims = (lims,args[0])
    lims = np.array(lims, dtype=float)
    # Compute order of magnitude of range
    range_mag = int(np.floor(np.log10(abs(lims[1]-lims[0]))))
    # Compute order of magnitude of limits (avoid log(0) issues)
    min_mag = int(np.floor(np.log10(abs(lims[0]))) if lims[0] != 0 else range_mag)
    max_mag = int(np.floor(np.log10(abs(lims[1]))) if lims[1] != 0 else range_mag)
    # Decide exponent based on range unless min/max are similar in magnitude
    mag = (max_mag-1) if lims[1]/10**max_mag < 3 else min_mag if lims[0]/10**min_mag < 3 else range_mag
    if e3_only: # only allow exponents that are multiples of 3, prefer lower if not multiple of 3
        mag = 3*round(mag/3)
    return mag


def set_yaxis_exfmt(axe,label,unit=None,change_prefix=False, **kwargs):
    """Set the axis label (with unit if specified), and format the ticks if an exponent can be removed.

    Args:
        axe (plt.Axis): The axis to set the format for.
        label (str): The axis label.
        unit (str, optional): The axis unit. Defaults to None.
        change_prefix (bool, optional): Whether to attempt to change the prefix of the unit. Defaults to False.
        **kwargs:
            - exponent (int, optional): fixed exponent for y-axis. Defaults to None. if specified, will remove that exponent from the y-axis labels, and add to axis label unit if already set.
            - clip_neg (bool, optional): Whether to clip negative values. Defaults to True.

    """
    ylim = list(axe.get_ylim())

    if kwargs.get("clip_neg", True):
        ylim = [max(ylim[0],0), max(ylim[1],0)]
    ylim[1]+=kwargs.get("y_extend",0.1)*abs(np.diff(ylim))
    axe.set_ylim(ylim)
    axis = axe.yaxis
    unit = unit if unit is not None else " "
    tex = unit[0] == "$"
    use_exp = True
    if tex:
        unit = unit[1:-1]
    exponent = kwargs.get("exponent", get_axis_exp(axe.get_ylim()))
    if change_prefix: # attempt to change the prefix of the unit.
        #will likely fail if unit is not si or something like km^2 (where k needs to be sqared as well)
        # or if the unit is more complex (GW/km)
        if len(unit) > 1:
            prefix_exp = CONST.SI_exponents.get(unit[0], None)
            if (prefix_exp is not None) and (unit != "kg"):
                unit = unit[1:]
                exponent += prefix_exp
        if exponent in CONST.SI_exponents.values():
            # find  key for the exponent
            for k,v in CONST.SI_exponents.items():
                if v == exponent:
                    unit = k+"~" + unit
                    use_exp = False
                    break
    exp_lbl = "\\times 10^{" + str(exponent) + "}~" if exponent != 0 and use_exp else ""
    unitpart = "$(" +  exp_lbl +unit +")$"
    label = label + " " + unitpart if len(unitpart) > 2 else label
    axe.set_ylabel(label)

    axis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/10**exponent:g}"))



def figsize(**kwargs):
    """Create a figure with a specified size parameters, to allow for consistent figure sizes--> text sizes, etc.

    Args:
        **kwargs:
            - width (float, optional): The fraction of the page width to use. Defaults to 1.
            - height (float, optional): The fraction of the page height to use. Defaults to 1.
            - aspect (float, optional): The aspect ratio of the figure (width/height). Defaults to 3/2. Width takes precedence. Ignored if both width and height are specified.
            - margin (float, optional): The margin around the figure. Defaults to plot.margin.

    Returns:
        tuple: figure width and height in inches

    """
    size = plot.size_a4
    margin = kwargs.get("margin", plot.margin)
    size = {k:size[k] - 2*margin for k in size} # maximum allowed size
    if kwargs.get("max") is not None:
        return size["width"], size["height"]
    width = kwargs.get("width", 1)*size["width"]
    if kwargs.get("height") is not None:
        height = kwargs["height"]*size["height"]
    else:
        height = width/kwargs.get("aspect", 3/2)
    return width, height





def plot_visit_intervals(axs,visits,fitsdirs=hst_fpath_list()[:-1],color="#aaa", **kwargs):
    """Plot the visit intervals as shaded x-axis spans on the provided axes.

    Args:
        axs (plt.Axes or list of plt.Axes): The axes to plot the visit intervals on.
        visits (list): The list of visits to plot.
        fitsdirs (list, optional): The list of FITS directories to extract the visit intervals from. Defaults to hst_fpath_list()[:-1].
        color (str, optional): The color of the shaded intervals. Defaults to "#aaa".
        annotate_ax (bool, optional): Whether to annotate the axes with the visit numbers. Defaults to False.
        **kwargs: Additional keyword arguments for the axvspan function.

    """
    for f in fitsdirs:
        mind, maxd = get_obs_interval(f)
        mind = pd.to_datetime(mind)
        maxd = pd.to_datetime(maxd)
        # span over the interval
        fvisit = group_to_visit(int(f.split("_")[-1][:2]))
        for ax in axs if isinstance(axs,(list,tuple)) else axs.flatten() if isinstance(axs,np.ndarray) else [axs]:
            ax.axvspan(mind, maxd, alpha=0.5, color="#aaa5" if fvisit not in visits else color, **kwargs)


def fill_between_qty(ax,xdf,df,quantity,visits, edgecorrect=False, fc=(0.7,0.7,0.7,0.3), ec=(0.7,0.7,0.7,0.3), hatch="\\\\",zord=5):
    r"""Plot the quantity ranges for the specified visits as shaded regions on the provided axes.

    Args:
        ax (plt.Axes): The axes to plot the quantity ranges on.
        xdf (DataFrame): The Dataframe containing all the data.
        df (DataFrame): The DataFrame containing the quantity data. (ie a single series, not merged.)
        quantity (str): The quantity to plot.
        visits (list): The list of visits to plot.
        edgecorrect (bool, optional): Whether to correct the edges of the shaded regions, to remove any borders. Defaults to False.
        fc (tuple, optional): The face color of the shaded regions. Defaults to (0.7,0.7,0.7,0.3).
        ec (tuple, optional): The edge color of the shaded regions. Defaults to (0.7,0.7,0.7,0.3).
        hatch (str, optional): The hatch pattern for the shaded regions. Defaults to `'\\\\'`.
        zord (int, optional): The z-order of the shaded regions. Defaults to 5.

    """
    # define minimums and maximums per visit, get minquant, maxquant, tmin, tmax.
    ranges = []
    for j,u in enumerate(visits): # for each visit

        if u in xdf["Visit"].unique(): # if the visit is in the total dataframe
            idfint = df.where(df["Visit"] == u) # get the interval over the visit
            dmax, dmin = [idfint["time"].max(), idfint["time"].min()]
            qmin, qmax = [idfint[quantity].min(), idfint[quantity].max()]
            ranges.append([dmin, qmin, qmax])
            ranges.append([dmax, qmin, qmax])
    ranges=pd.DataFrame(ranges, columns=["time","min","max"])
    ranges = ranges.dropna()
    ranges = ranges.sort_values(by=["time"])
    ax.fill_between(ranges['time'],  ranges["max"], ranges["min"],facecolor=fc, edgecolor=ec,hatch=hatch,linewidth=0.5,  interpolate=True, zorder=zord)
    if edgecorrect:
        ax.fill_between(ranges['time'],  ranges["max"], ranges["min"],facecolor=(0,0,0,0), edgecolor=(1,1,1), interpolate=True, zorder=zord+1)

def mgs_grid(fig, visits, main_height=4, grid_height=5, grid_kwargs={}, main_gs_kwargs={"hspace":0.1, "wspace":0}):
    """Create a main grid and subgrid of axes for the provided figure and visits.

    Args:
        fig (plt.Figure): The figure to create the grids on.
        visits (list): The list of visits to create the grids for.
        main_height (int, optional): The height of the main grid. Defaults to 4.
        grid_height (int, optional): The height of the subgrid. Defaults to 5.
        grid_kwargs (dict, optional): Additional keyword arguments for the subgrid. Defaults to {"wspace":0}.
        main_gs_kwargs (dict, optional): Additional keyword arguments for the main grid. Defaults to {"hspace":0.5, "wspace":0}.

    Returns:
        tuple: The main axis and subaxes for the figure.

    """
    icols, jrows = approx_grid_dims(visits)
    gs = fig.add_gridspec(2, 1, height_ratios=[main_height, grid_height], **main_gs_kwargs)  # hspace=0,
    mgs = gs[1].subgridspec(jrows, icols, height_ratios=[1 for _ in range(jrows)], **grid_kwargs)  # hspace=0,wspace=0,
    main_ax = fig.add_subplot(gs[0])
    main_ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval=2))
    main_ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%d/%m/%y"))
    main_ax.xaxis.set_minor_locator(mpl.dates.DayLocator(interval=1))
    axs = [[fig.add_subplot(mgs[j, i], label=f"v{visits[i+icols*j]}") for i in range(icols)] for j in range(jrows)]
    return main_ax, axs




def stacked_plot(hst_datasets, hisaki_dataset, savepath,hisaki_cols=[],hst_cols=[], fill_between=True):
        """Plot a stacked x-axis plot of all columns in the datasets, over the timeperiod of hst_datasets.

        Args:
            hst_datasets (list): The list of HST datasets to plot.
            hisaki_dataset (DataFrame): The HISAKI dataset to plot.
            savepath (str): The path to save the plot to.
            hisaki_cols (list, optional): The columns to plot from the HISAKI dataset. Defaults to [].
            hst_cols (list, optional): The columns to plot from the HST datasets. Defaults to [].
            fill_between (bool, optional): Whether to fill between the quantities. Defaults to True.

        """
        # plot a stacked x-axis plot of all columns in the datasets, over the timeperiod of hst_datasets
        xlim = get_time_interval_from_multi(hst_datasets)
        extend = np.timedelta64(1, "D")
        xlim = (xlim[0]-extend, xlim[1]+extend)
        figure,axs = plt.subplots(len(hst_cols)+len(hisaki_cols),1,figsize=figsize(max=True),sharex=True, gridspec_kw={"hspace":0.05}, subplot_kw={"xmargin":0.02, "ymargin":0.02}, constrained_layout=True)
        hst_m = merge_dfs(hst_datasets)
        for i, col in enumerate(hst_cols):
            for hst in hst_datasets:
                axs[i].scatter(hst["time"],hst[col],label=col, color=hst["color"][0], marker=hst["marker"][0], zorder=hst["zorder"][0], s=2)
                mv = HST.mapval(col, ["label","unit"])
            #if col not in ["Total_Power", "Avg_Flux"]:
            axs[i].set_ylim(0,hst_m[col].max())
            set_yaxis_exfmt(axs[i], label=mv[0], unit=mv[1])
            if fill_between:
                fill_between_qty(axs[i],hst_m,hst_m,quantity=col,visits=hst_m["Visit"].unique(), edgecorrect=True, zord=hst_m["zorder"][0]-10)
        for i, col in enumerate(hisaki_cols):
            axs[i+len(hst_cols)].plot(hisaki_dataset.time.datetime64,hisaki_dataset[col],label=col)
            mv = HISAKI.mapval(col, ["label","unit"])
            set_yaxis_exfmt(axs[i+len(hst_cols)], label=mv[0], unit=mv[1])
        figure.suptitle("Comparison of Energy Flux, Area and Total Power against HISAKI data")
        apply_plot_defaults(fig=figure, time_interval=xlim, day_interval=1, visits=True)
        file = savepath+f"stacked_{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.png"
        figure.savefig(file)
        plt.close()
        return file

#TODO @will-roscoe remove or use visit_intervals
def overlaid_plot(cols, hst_datasets, hisaki_dataset, savepath, visit_intervals=False, fill_between=True):#noqa: ARG001
    """Plot a stacked x-axis plot of the provided columns from the datasets, with visit intervals and quantity ranges. Similar columns are grouped together.

    Args:
        cols (list): structured list of columns to plot, items in level 1 are plotted on same axes, with twin axis, and level 2 items are plotted on the same axis.
        hst_datasets (list): The list of HST datasets to plot.
        hisaki_dataset (DataFrame): The HISAKI dataset to plot.
        savepath (str): The path to save the plot to.
        visit_intervals (bool, optional): Whether to plot visit intervals. Defaults to False.
        fill_between (bool, optional): Whether to fill between the quantities. Defaults to True.

    Returns:
        str: The path to the saved plot

    """
    xlim = get_time_interval_from_multi(hst_datasets)
    extend = np.timedelta64(1, "D")
    xlim = (xlim[0]-extend, xlim[1]+extend)
    # identify complementary columns in the datasets
    # make a map of datasets to cols: True if hisaki, False if hst
    df_map = []
    for col in cols:
        if isinstance(col,str):
            df_map.append(col in list(hisaki_dataset.columns))
        else:
            part = []
            for c in col:
                if isinstance(c,str):
                    part.append(c in list(hisaki_dataset.columns))
                else:
                    pa = []
                    for cc in c:
                        pa.append(cc in list(hisaki_dataset.columns))
                    part.append(pa)

            df_map.append(part)
    # axes is a list of axes and 2-tuples of axes (for twin axes)
    figure,axs = plt.subplots(len(cols),1,figsize=figsize(max=True),sharex=True, gridspec_kw={"hspace":0.05}, subplot_kw={"xmargin":0.02, "ymargin":0.02}, constrained_layout=True)
    figure.suptitle("Comparison of Energy Flux, Area and Total Power against HISAKI data")
    axes = []
    for i, col in enumerate(cols):
        if isinstance(col,str):
            axes.append(axs[i])
        else:
            axes.append([axs[i], axs[i].twinx()])
    def plot_hisaki(ax, col):
        ax.plot(hisaki_dataset.time.datetime64,hisaki_dataset[col],label=col)
        mv = HISAKI.mapval(col, ["label","unit"])
        set_yaxis_exfmt(ax, label=mv[0], unit=mv[1])
    hst_m = merge_dfs(hst_datasets)
    def plot_hst(ax, col):
        for hst in hst_datasets:
            ax.scatter(hst["time"],hst[col],label=col, color=hst["color"][0], marker=hst["marker"][0], zorder=hst["zorder"][0], s=2)
            mv = HST.mapval(col, ["label","unit"])
        ax.set_ylim(0,hst_m[col].max())
        set_yaxis_exfmt(ax, label=mv[0], unit=mv[1])
        if fill_between:
                fill_between_qty(ax,hst_m,hst_m,quantity=col,visits=hst_m["Visit"].unique(), edgecorrect=True, zord=hst_m["zorder"][0]-10)
    def plot_one(ax,col,dfv):
        if dfv:
            plot_hisaki(ax,col)
        else:
            plot_hst(ax,col)

    # go through the maps and plot the data on the designated axes, using the designated datasets
    for idx, col in enumerate(cols):
        if isinstance(col,str):
            plot_one(axes[idx],col,df_map[idx])
        else:
            for idy, c in enumerate(col):
                if isinstance(c,str):
                    plot_one(axes[idx][idy],c,df_map[idx])
                else:
                    for id_, cc in enumerate(c):
                        if isinstance(cc,str):
                            plot_one(axes[idx][idy],cc,df_map[idx][idy])
                        else:
                            for idz, ccc in enumerate(cc):
                                plot_one(axes[idx][idy],ccc,df_map[idx][idy][idz])
    apply_plot_defaults(fig=figure, time_interval=xlim, day_interval=1, visits=True)
    file = savepath+f"overlaid_{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.png"
    figure.savefig(file)
    plt.close()
    return file




def megafigure_plot(hisaki_cols,hst_cols, hst_datasets, hisaki_dataset, savepath, fill_between=True):
    """Plot a megafigure of all the provided columns from the datasets, with visit intervals and quantity ranges.

    Args:
        hisaki_cols (list): The columns to plot from the HISAKI dataset.
        hst_cols (list): The columns to plot from the HST datasets.
        hst_datasets (list): The list of HST datasets to plot.
        hisaki_dataset (DataFrame): The HISAKI dataset to plot.
        savepath (str): The path to save the plot to.
        fill_between (bool, optional): Whether to fill between the quantities. Defaults to True.

    Returns:
        str: The path to the saved plot

    """
    xlim = get_time_interval_from_multi(hst_datasets)
    extend = np.timedelta64(1, "D")
    xlim = (xlim[0]-extend, xlim[1]+extend)
    visits_ = [list(hs["Visit"].unique()) for hs in hst_datasets]
    visits = set()
    for v in visits_:
        visits = visits.union(v)
    visits = list(visits)
    figure = plt.figure(figsize=figsize(width =1.4, aspect=3/2), constrained_layout=True)

    figure.suptitle(f"Overview of {", ".join([h.replace("_"," ") for h in hst_cols])} over all visits")
    main_ax,axs = mgs_grid(figure, visits) #  I,J grid of subplots, per visit
    icols, jrows = approx_grid_dims(visits)
    merged = merge_dfs(hst_datasets)
    for i in range(icols):
        for j in range(jrows):
            if i + icols * j < len(visits):
                ax = axs[j][i]
                if i == 0:
                    ax.set_ylabel(f"Visit {visits[i+j*icols]}")
                for col in hst_cols:
                    for h in hst_datasets:
                        ax.scatter(h["time"],h[col],label=col, color=h["color"][0], marker=h["marker"][0], zorder=h["zorder"][0], s=0.5)
                        ax.plot(h["time"],h[col], color=h["color"][0], lw=0.2, zorder=h["zorder"][0])
                    mv = HST.mapval(col, ["label","unit"])
                    ax.set_ylim(0,merged[col].max())
                    set_yaxis_exfmt(ax, label=mv[0], unit=mv[1])
                    # get min and max for this visit.
                xdf = merged.where(merged["Visit"] == visits[i+j*icols])
                dmax, dmin = [xdf["time"].max(), xdf["time"].min()]


                ax.set_xlim(dmin,dmax)
                for col in hisaki_cols:
                    tax=ax.twinx()
                    # make ticks, labels, spine invisible on the y-axis
                    with contextlib.suppress(Exception):
                        tax.spines["left"].set_visible(False)
                        tax.set_yticklabels([])
                        tax.set_yticks([])
                    #tax.plot(hisaki_dataset.time.datetime64,hisaki_dataset[col],label=col, lw=0.2, color = "black")
                apply_plot_defaults(ax=ax, day_of_year=False,visits=False, compact={})
                ax.set_ylabel("")
                ax.tick_params(axis="x", labelsize="small")
                ax.annotate(f"v{visits[i+j*icols]}", **plot.inset_annotate_kws)
                ax.annotate(dmin.strftime("%d/%m/%y"), **plot.meta_annotate_kws)
                if i!=0:
                    ax.tick_params(axis="y", labelleft=False, labelright=False)
    for i, col in enumerate(hst_cols):
        ax = main_ax
        for h in hst_datasets:
            ax.scatter(h["time"],h[col],label=col, color=h["color"][0], marker=h["marker"][0], zorder=h["zorder"][0], s=0.5)
        if fill_between:
            fill_between_qty(ax,merged,merged,quantity=col,visits=merged["Visit"].unique(), edgecorrect=True, zord=merged["zorder"][0]-10)

        mv = HST.mapval(col, ["label","unit"])
    main_ax.set_ylim(0,merged[col].max())
    set_yaxis_exfmt(main_ax, label=mv[0], unit=mv[1])
    apply_plot_defaults(ax=[main_ax], time_interval=xlim, day_interval=1, visits=True, annotate=False)
    file = savepath+f"megafigure_[{".".join(hst_cols)}]{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.png"
    figure.savefig(file)
    plt.close()
    return file


def dpr_histogram_plot(array_path, method="indiv", bins=1000,rolling_avg=0.05,stats=False,log="y", **kwargs):
    """Plot a histogram of the provided array path, with the specified method and parameters.

    Args:
        array_path (str or list): The path to the array to plot.
        method (str, optional): The method to use for the plot. Defaults to "indiv".
        bins (int, optional): The number of bins to use. Defaults to 1000.
        rolling_avg (float, optional): The rolling average to use. Defaults to 0.05.
        stats (bool, optional): Whether to display statistics. Defaults to False.
        log (str, optional): The axis to log. Defaults to "y".
        **kwargs: Additional keyword arguments for the plot.

    Returns:
        tuple: The figure, axis, and data

    """
    array_path = array_path if isinstance(array_path,(list,tuple)) else [array_path]
    data = np.array([np.load(path) for path in array_path]).squeeze()
    ndim = data.ndim
    # replace all nans and infs with a large negative number, then filter out zeros and negative values
    data = np.nan_to_num(data, nan=-1e10, posinf=-1e10, neginf=-1e10)
    data[data <= 0] = np.nan
    data = np.nan_to_num(data, nan=0)
    fig,ax = plt.subplots(figsize=figsize())
    fig.suptitle("Distribution of Observed...")
    if rolling_avg:
        kernel = np.ones(int(bins*rolling_avg))/int(bins*rolling_avg)
    if ndim == 2:
        x,y = hist2xy(np.histogram(data, bins=1000))
        ax.bar(x,y, width=kwargs.pop("width",np.mean(np.diff(x))),**kwargs)
        if rolling_avg:
            # rolling average
            xs = np.convolve(x, kernel, mode="valid")
            ys = np.convolve(y, kernel, mode="valid")
            ax.plot(xs,ys, color="red", zorder=10,lw=0.5,)
    elif ndim == 3:
        if method == "indiv":
            color = kwargs.pop("color",None)
            width=kwargs.pop("width",None)
            d = []
            for i in range(data.shape[0]):
                x,y = hist2xy(np.histogram(data[i], bins=bins))

                c = color[i] if color is not None else ("#49f",1/len(data)) if rolling_avg else mpl.cm.jet(i/len(data),alpha=1/len(data))
                w = width if width is not None else np.mean(np.diff(x))
                ax.bar(x,y, width=w,color=c,**kwargs)
                if rolling_avg:
                    # rolling average
                    xs = np.convolve(x, kernel, mode="valid")
                    ys = np.convolve(y, kernel, mode="valid")
                    ax.plot(xs,ys, color=("#000",1/(0.5*len(data))), zorder=10, lw=0.5)
                d.append([x,y])


        elif method == "avg":
            x,y = hist2xy(np.histogram(data.mean(axis=0), bins=bins))
            ax.bar(x,y, width=kwargs.pop("width",np.mean(np.diff(x))),**kwargs)
            if rolling_avg:
                # rolling average
                xs = np.convolve(x, kernel, mode="valid")
                ys = np.convolve(y, kernel, mode="valid")
                ax.plot(xs,ys, color="red", zorder=10,lw=0.5)
    # remove indices where x close to zero and y is very high (in comparison to the rest of the data)
    x = np.multiply(x, CONST.calib_cts2kr) # convert to kilorayleighs (might not work)
    ax.set_xlabel("Intensity (kR)")

    if stats:
        x,y = np.array(x), np.array(y)
        st = {"mean":np.mean(data), "med":np.median(data), "mode":x[np.argmax(y[x>0.01])]}
        for k,v in st.items():
            # draw vertical lines to the height of the histogram value nearest to the value
            ys = np.interp(v,x,y)
            ax.vlines(v,0,ys, color="red", linestyle="--")
            # annotate above the line
            ax.annotate(f"{k}: {v:.2f}", (v, ys), textcoords="offset points", xytext=(0,5), ha="center")
    apply_plot_defaults(fig=fig, day_of_year=False)

    fig.suptitle("Distribution of Observed Flux within ROI (visit##, at DATETIME)")
    if stats or "y" in log:
        ax.set_yscale("log")
        ax.set_ylabel("Frequency")
    else:
        set_yaxis_exfmt(ax, label="Frequency", unit="f")
    if "x" in log:
        ax.set_xscale("log")
    return fig,ax,d if ndim == 3 and method == "indiv" else np.array([x,y]) if ndim == 2 else None























