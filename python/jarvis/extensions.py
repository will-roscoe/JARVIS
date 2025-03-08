#!/usr/bin/env python3
"""jarvis.extensions
- This module contains functions and classes that extend the functionality of the jarvis package.
- The functions and classes in this module are designed to be used in conjunction with the core functionality of the jarvis package.

Functions:
- pathfinder: A GUI tool to select contours from a fits file, for use in defining auroral boundaries.

Classes:
- QuickPlot: Predefined plotting functions for quick visualization of data in power.py [internal]
"""

from datetime import datetime
from os.path import exists
from typing import Iterable, List, Union

import cmasher as cmr
import cv2
import matplotlib.pyplot as plt
import numpy as np
from astropy.io.fits import HDUList, Header
from astropy.io.fits import open as fopen
from matplotlib import rc_context, use
from matplotlib.font_manager import fontManager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle as Rectangle
from matplotlib.path import Path as mPath
from matplotlib.text import Text
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox
from tqdm import tqdm

from .cvis import contourhdu, imagexy
from .polar import prep_polarfits
from .transforms import azimuthal_equidistant, fullxy_to_polar_arr
from .utils import assign_params, filename_from_path, fitsheader, fpath, get_datetime, hdulinfo, rpath, split_path

try:  # this is to ensure that this file can be imported without needing PyQt6 or PyQt5, eg for QuickPlot
    from PyQt6 import QtGui  # type: ignore #
except ImportError:
    try:
        from PyQt5 import QtGui  # type: ignore #
    except ImportError:
        tqdm.write("Warning: PyQt6 or PyQt5 not found, pathfinder app may not function correctly")


MAXPOINTSPERCONTOUR = 2000
MAXCONTOURDRAWS = 125
MAXLEGENDITEMS = 50
TWOCOLS = True
# ------------------- Tooltips and Info about Config Flags ---------------------#
#  naming convention:
#    - flag (RETR, CHAIN, MORPH, KSIZE, ...) -> axes label name, configuration parameter
#    - label (EXTERNAL, LIST, SIMPLE, ...) -> value name, configuration option
#    - value -> the value of the configuration option, at [0] in _cvtrans[flag][label]
#    - index -> the index of the configuration option, usually the same as value, otherwise the index of the value in the translation list, trans
#
# required for each configuration flag dictionary:
#    - info: a description of the flag
#    - 'FLAG' : for each flag option, a list of the form [value, keybinding,description, extension], where:
#      if the flag is a boolean, or otherwise has no labels, use the flag name as the label
#    - kbtemplate: a string template for the keybinding tooltip, opitionally using the variables
#                   - flag,
#                   - label,
#                   - extension (extension being an optional index 3 value in each label list),
#                   - tooltip (the entire tooltip string),
#                   - info (the flag info).
# not required:
#    - trans: a list of the values in the order they should be displayed in the GUI, if not present, it is assumed to be values [0,1,2,3,....].

_cvtrans = {
    "RETR": {
        "info": "modes to find the contours",
        "kbtemplate": "$flag mode: $label",
        "EXTERNAL": [cv2.RETR_EXTERNAL, "a", "retrieves only the extreme outer contours"],  # 0
        "LIST": [
            cv2.RETR_LIST,
            "s",
            "retrieves all of the contours without establishing any hierarchical relationships",
        ],  # 1
        "CCOMP": [
            cv2.RETR_CCOMP,
            "d",
            "retrieves all of the contours and organizes them into a two-level hierarchy, where external contours are on the top level and internal contours are on the second level",
        ],  # 2
        "TREE": [
            cv2.RETR_TREE,
            "f",
            "retrieves all of the contours and reconstructs a full hierarchy of nested contours",
        ],  # 3
    },
    "CHAIN": {
        "info": "methods to approximate the contours",
        "kbtemplate": "$flag mode: $label",
        "trans": (1, 2, 3, 4),  # 0 is floodfill, but not used here
        "NONE": [
            cv2.CHAIN_APPROX_NONE,
            "z",
            "stores absolutely all the contour points. That is, any 2 subsequent points (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors, that is, max(abs(x1-x2),abs(y2-y1))==1.",
        ],  # 1
        "SIMPLE": [
            cv2.CHAIN_APPROX_SIMPLE,
            "x",
            "compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points.",
        ],  # 2
        "TC89_L1": [
            cv2.CHAIN_APPROX_TC89_L1,
            "c",
            "applies one of the flavors of the Teh-Chin chain approximation algorithm.",
        ],  # 3
        "TC89_KCOS": [
            cv2.CHAIN_APPROX_TC89_KCOS,
            "v",
            "applies one of the flavors of the Teh-Chin chain approximation algorithm.",
        ],  # 4
    },
    "MORPH": {
        "info": "morphological operations to apply to the mask before finding the contours",
        "kbtemplate": "$flag: toggle $label",
        "ERODE": [cv2.MORPH_ERODE, "q", "Erodes away the boundaries of foreground object"],  # 0
        "DILATE": [cv2.MORPH_DILATE, "w", "Increases the object area"],  # 1
        "OPEN": [cv2.MORPH_OPEN, "e", "Remove small noise"],  # 2
        "CLOSE": [cv2.MORPH_CLOSE, "r", "Fill small holes"],  # 3
        "GRADIENT": [cv2.MORPH_GRADIENT, "t", "Difference between dilation and erosion of an image."],  # 4
        "TOPHAT": [cv2.MORPH_TOPHAT, "y", "Difference between input image and Opening of the image"],  # 5
        "BLACKHAT": [cv2.MORPH_BLACKHAT, "u", "Difference between the closing of the input image and input image"],  # 6
        "HITMISS": [cv2.MORPH_HITMISS, "i", "Extracts a particular structure from the image"],  # 7
    },
    "KSIZE": {
        "info": "kernel size for the morphological operations. Larger values will smooth out the contours more",
        "kbtemplate": "Change kernel size: $label",
        "Increase": [True, "=", "Increase the kernel size"],  # 0
        "Decrease": [False, "-", "Decrease the kernel size"],
    },
    "CVH": {
        "info": "whether to use convex hulls of the contours. Reduces the complexity of the contours",
        "kbtemplate": "Toggle $label",
        "Convex Hull": [0, "f7", "whether to use convex hulls of the contours. Reduces the complexity of the contours"],
    },
    "ACTION": {
        "info": "Select the action when clicking on a point in the image",
        "kbtemplate": "Click mode: $label",
        "trans": (0, 1, -1, 2, -2),
        "None": [0, "0", ""],
        "Add Luminosity": [1, "1", "Add a luminosity sample at the clicked point"],
        "Remove Luminosity": [-1, "2", "Remove a luminosity sample closest to the clicked point"],
        "Add IDPX": [2, "3", "Add an ID pixel at the clicked point"],
        "Remove IDPX": [-2, "4", "Remove an ID pixel closest to the clicked point"],
    },
    "CLOSE": {"info": "Close the viewer", "kbtemplate": "$info", "CLOSE": [0, "escape", "Close the viewer"]},
    "SAVE": {
        "info": "Save the selected contour, either to the current fits file or to a new file if a path has been provided.",
        "kbtemplate": "$tooltip",
        "SAVE": [0, "insert", "Save the selected contour"],
    },
    "SAVECLOSE": {
        "info": "Save the selected contour, and then close the GUI.",
        "kbtemplate": "$tooltip",
        "SAVECLOSE": [0, "enter", "Save the selected contour, and close the viewer"],
    },
    "RESET": {
        "info": "Reset the viewer pixel selections",
        "kbtemplate": "Reset selections",
        "RESET": [0, "f10", "Reset the viewer pixel selections"],
    },
    "FSCRN": {
        "info": "Toggle fullscreen mode",
        "kbtemplate": "Toggle Fullscreen",
        "FSCRN": [0, "f11", "Toggle Fullscreen"],
    },
    "KILL": {
        "info": "Kill the current process",
        "kbtemplate": "$tooltip",
        "KILL": [0, "delete", "Kill the current process"],
    },
    "MASK": {"info": "Cycle mask display", "kbtemplate": "$tooltip", "MASK": [0, "f3", "Cycle mask display"]},
    "CMAP": {"info": "Cycle colormap", "kbtemplate": "$tooltip", "CMAP": [0, "f4", "Cycle colormap"]},
    "NOTES": {"info": "Toggle note textbox", "kbtemplate": "$tooltip", "NOTES": [0, "f1", "Toggle note textbox"]},
    "TOOLTIP": {
        "info": "Toggle tooltips",
        "kbtemplate": "$tooltip",
        "TOOLTIP": [0, "f2", "Toggle tooltips"],
        "ONSCREEN": [1, "f12", "Toggle onscreen tooltips"],
        "CLI": [2, "`", "Print CLI help"],
    },
    "FIXLRANGE": {
        "info": "Controls for altering the fixed luminance range, which is the lower and upper limit of valid luminances",
        "kbtemplate": "$flag: $tooltip",
        "Lower+": [0, "]", "Increase lower limit"],
        "Lower-": [1, "[", "Decrease lower limit"],
        "Upper+": [2, "#", "Increase upper limit"],
        "Upper-": [3, "'", "Decrease upper limit "],
        "Reset": [4, "f6", "Reset range to initial"],
        "Toggle": [5, "f5", "Toggle fixed luminance range"],
        "Cycle": [6, "f8", "Cycle through steps"],
    },
}
stepmap = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
HELPKEYS = [k for k in [_cvtrans["TOOLTIP"].get(k, [None, None])[1] for k in ["ONSCREEN", "CLI"]] if k is not None]


def __getflaglabels(flag: str):
    """Returns the ordered list of labels for a given flag"""
    ident = list(_cvtrans[flag].keys())
    ident.remove("info")
    ident.remove("kbtemplate")
    if len(ident) == 1:
        return [ident[0]]

    if "trans" not in ident:
        ret = ["" for _ in range(len(ident))]
        for k, v in zip(ident, [_cvtrans[flag][i][0] for i in ident]):
            ret[v] = k
    else:
        ident.remove("trans")
        ret = ["" for _ in range(len(ident))]
        for k, v in zip(ident, [_cvtrans[flag][i][0] for i in ident]):
            ret[_cvtrans[flag]["trans"].index(v)] = k
    return ret


def __getflagindex(flag: str, label: str):
    """Returns the correct index of a given label for a given flag"""
    ident = _cvtrans[flag]
    # print(f"{flag=}, {label=}, {ident=}")
    if ident.get("trans", None) is None:
        # print(f"{ident[label][0]=}")
        return ident[label][0]
    return ident["trans"].index(ident[label][0])


# --------------------------------- Keybindings --------------------------------#
def __evalkbt(flag, label=None):
    """Evaluates the keybinding tooltip template for a given flag dictionary"""
    temp = _cvtrans[flag].get("kbtemplate", None)
    if "$label" in temp:
        if label is None:
            label = flag
        temp = temp.replace("$label", label)
    if "$flag" in temp:
        temp = temp.replace("$flag", flag)
    if "$extension" in temp:
        temp = temp.replace("$extension", _cvtrans[flag][label][-1] if len(_cvtrans[flag][label]) > 3 else "")
    if "$tooltip" in temp:
        temp = temp.replace("$tooltip", _cvtrans[flag][label][2])
    if "$info" in temp:
        temp = temp.replace("$info", _cvtrans[flag]["info"])
    return temp


_keybindings = {}  # Structure: {'key':('flag',value,'tooltip')}
for k, v in _cvtrans.items():
    for label in __getflaglabels(k):
        _keybindings[v[label][1]] = (k, label, __evalkbt(k, label))
        # 'key':('flag',value,'kb_tooltip')


# print(_keybindings)
def __gettooltip(flag, value=None, label=None):
    """Returns the tooltip for a flag (and/or label/value combination), along with the correct keybinding, if available"""
    if label is not None:
        tt = _cvtrans[flag][label][2]
    elif value is not None:
        tt = _cvtrans[flag][__getflaglabels(flag)[value]][2]
    else:
        tt = _cvtrans[flag]["info"]
    kb = [
        k
        for k, v in _keybindings.items()
        if v[0] == flag and (label is None or v[1] == label) and (value is None or v[1] == value)
    ]
    return f"{tt} ({kb[0]})" if kb else tt


# ----------------------------- Style Definitions ------------------------------#
_bgs = {"main": "#000", "sidebar": "#fff", "legend": "#000"}  # > Background colors (hierarchical) based on axes labels
CSEL = "RGB"  #'DEFAULT' #> Default color pallete choice
COLORPALLETTE = {
    "RGB": ["#FF0000FF", "#00FF00FF", "#FF0000FF", "#0000FFFF"],
    "DEFAULT": ["#FF0000FF", "#FF5500FF", "#FF0000FF", "#FFFF0055"],
    "BLUE": ["#0077FFFF", "#7700FFFF", "#0000FFFF", "#00FFFF55"],
    "GREEN": ["#00FF00FF", "#00FF55FF", "#00FF00FF", "#55FF0055"],
    "BW": ["#000", "#FFF", "#000000FF", "#FFFFFF55"],
}


def __get_bg(ax):
    """Returns the background color for a given axes"""
    return _bgs.get(ax.get_label(), _bgs.get("sidebar", _bgs.get("main", "#fff")))


_legendkws = {
    "loc": "lower left",
    "fontsize": 8,
    "labelcolor": "linecolor",
    "frameon": False,
    "mode": "expand",
    "ncol": 3,
}  # > active contour list styling
_idpxkws = {"s": 20, "color": COLORPALLETTE[CSEL][0], "zorder": 12, "marker": "x"}  # Identifier Pixel scatter style
_clickedkws = {"s": 20, "color": COLORPALLETTE[CSEL][1], "zorder": 12, "marker": "x"}  # Clicked Pixel scatter style
_selectclinekws = {"s": 0.3, "color": COLORPALLETTE[CSEL][2], "zorder": 10}  # Selected Contour line style
_selectctextkws = {"fontsize": 8, "color": _selectclinekws["color"]}  # Selected Contour text style
_otherclinekws = {"s": 0.2, "color": COLORPALLETTE[CSEL][3], "zorder": 10}  # Other Contour line style
_otherctextkws = {"fontsize": 8, "color": _otherclinekws["color"]}  # Other Contour text style
_defclinekws = {
    "s": _selectclinekws["s"],
    "color": _selectclinekws["color"],
    "zorder": _selectclinekws["zorder"],
}  # Default Contour line style
_defctextkws = {"fontsize": 8, "color": _defclinekws["color"]}  # Default Contour text style
_handles = {  # > Legend handle styling and static elements (unused)
    "selectedc": {"color": _selectclinekws["color"], "lw": 2},
    "otherc": {"color": _otherclinekws["color"], "lw": 2},
    "defc": {"color": _defclinekws["color"], "lw": 2},
}
cmap_cycler = [
    cmr.neutral,
    cmr.neutral_r,
    cmr.toxic,
    cmr.nuclear,
    cmr.emerald,
    cmr.lavender,
    cmr.dusk,
    cmr.torch,
    cmr.eclipse,
]
# App Icon and Font Path
_iconpath = fpath("python/jarvis/resources/aa_asC_icon.ico")
_fontpath = fpath("python/jarvis/resources/FiraCodeNerdFont-Regular.ttf")
global FIRST_RUN
FIRST_RUN = True


# ----------------------------- Configuration Validation ------------------------#
def __validatecfg():
    """"""
    for k, v in _cvtrans.items():
        assert "info" in v, f"{k} must have an info key"
        assert "kbtemplate" in v, f"{k} must have a kbtemplate key"
        trans = v.get("trans", False)
        if trans:
            assert (
                len(v) - 3 == len(trans)
            ), f'{k}\'s index translation length does not match the number of flags: "trans": {trans}, flags: {list(v.keys())}'
        else:
            maxflag = max(val[0] for label, val in v.items() if label not in ["info", "kbtemplate"])
            lenflags = len(v) - 3  # zero indexed, and accounting for`info','kbtemplate' and no `trans'
            assert (
                lenflags == maxflag
            ), f"{k}'s index translation length does not match the number of flags, highest flag value: {maxflag}, flags: {list(v.keys())} (length: {lenflags}+2)"
        for label, val in v.items():
            if label not in ["trans", "info", "kbtemplate"]:
                assert (
                    len(val) >= 3
                ), f"{k}'s index translation values must be a list of length 3 or more, containing the flag value (or index),keybinding, and description"
                assert isinstance(
                    val[0], (int, bool),
                ), f"{k}'s index translation values must be integers or booleans, not {type(val[0])}: {val[0]}"
                assert isinstance(
                    val[1], str,
                ), f"{k}'s index translation keybindings must be strings, not {type(val[1])}: {val[1]}"
                assert isinstance(
                    val[2], str,
                ), f"{k}'s index translation descriptions must be strings, not {type(val[2])}: {val[2]}"
                if len(val) > 3:
                    assert isinstance(
                        val[3], str,
                    ), f"{k}'s index translation extensions must be strings, not {type(val[3])}: {val[3]}"
    for k, v in _keybindings.items():
        assert (
            len(v) == 3
        ), f"{k}'s keybinding must be a tuple of length 3 containing the flag, value, and a short description"
    assert exists(_iconpath), "Icon file not found"
    assert exists(_fontpath), "Font file not found"


if FIRST_RUN:
    __validatecfg()


# ------------------------- pathfinder (Function) -----------------------------#
# changelog
# - port over cvis functions and params, ksize,morphex,fcmethod,cvh,lrange,fcmode
# - matplotlib figure and axes setup
# add in interactive elements
# add in pathsaving and button funcs
# add in tooltips
# add in clicktype
# add in keybindings
# add in fits descriptor
# add in persistence between consecutive runs
# add in falsecolor, mask
# add in saving of config to fits file
# add ability to change headername
# add in notes
# unify the config flags and keybindings with tooltips
# add in the ability to change the fixed luminance range [current]
# add in autoselect contour [current]
# add in config loading
# FUTURE
# integrate the hierarchy into the gui


def pathfinder(
    fits_dir: HDUList,
    show_tooltips: bool = True,
    headername: str = "BOUNDARY",
    steps: bool = True,
    show: bool = True,
    **persists,
) -> List[bool, Union[np.ndarray, dict]]:
    """### *JAR:VIS* Pathfinder
> ***Requires PyQt6 (or PyQt5)***

A GUI tool to select contours from a fits file. The tool allows the user to select luminosity samples from the image, and then to select a contour that encloses all of the selected samples. The user can then save the contour to the fits file, or to a new file if a save location is provided. The user can also change the morphological operations applied to the mask before finding the contours, the method used to find the contours, the method used to approximate the contours, and whether to use the convex hull of the contours. The user can also change the kernel size for the morphological operations.

#### Parameters: (most may be changed during the session using the GUI buttons.)
- `fits_dir`: The fits file to open.
- `show_tooltips`: Whether to show tooltips when hovering over buttons.
- `steps`: whether to save snapshots of the image at each step.
- `show`: whether to show the GUI.
- `persists`: Configuration options.
> **the following options in `persists` are case insensitive:**\\
>   `"REGISTER_KEYS":bool=True`--> whether to register keybindings [internal]\\
>   `"cmap":int=0`--> the initial image colormap flag\\
>   `"tooltips":bool=True` --> whether to show tooltips when hovering over buttons\\
>   `"mask":int=0` --> the initial mask display flag\\
>   `"ACTION":int=0` --> the initial click action flag\\
>   `"MORPH":list[int]=[cv2.MORPH_OPEN,cv2.MORPH_CLOSE]` --> the initial flags for morphological operations to apply to the mask before finding the contours\\
>   `"CHAIN":int=cv2.CHAIN_APPROX_SIMPLE` --> the initial flag for the method used to approximate the contours\\
>   `"CVH":bool=False` --> whether to use convex hulls of the contours\\
>   `"KSIZE" :int=5 ` --> the initial kernel size for the morphological operations\\
>   `"RETR":int=cv2.RETR_EXTERNAL` --> the initial flag for the method used to find the contours\\
>   `"FIXEDRANGE":list[float]=None` --> the initial fixed luminance range, which is the lower and upper limit of valid luminances\\
>   `"LUMXY":list[list[float,[float]]]=[]` --> the initial luminosity samples. Each sample is a list of the form [lum, (x, y)]\\
>   `"IDXY": list[list[float]]=[]` --> the initial ID pixel samples. This chooses the contour by finding first contour that contains the pixel samples, largest first.\\
>   `"AUTOID":list[float]= None` --> the initial autoselect contour sample. This chooses the contour by finding the nearest point to the autoselect point.\\
>   `"EXTNAME":str="BOUNDARY"` --> the initial header name to save the contour to\\
>   `"path":np.ndarray=None` --> the initial path of the contour\\
>   `"snapshots":dict={}` --> the initial snapshot arrays of the image at each step. [internal]\\
>   `"attrs":dict={}` --> attributes to save to the header of the contour array when saved. [internal]\\
>**the following options in persists are case sensitive:**\\
>   `"conf"`:dict: contains configuration that can be loaded easier from a fits file previously saved.\\
>- conf keys:
>>`'LUMXY*'`: string of form ({lum},({x},{y})) --> appends\\
>>`'IDXY*'`: string of form ({x},{y})  --> appends\\
>>`'AUTOID'|'AUTOCL'`: string of form ({x},{y}) --> overwrites\\
>>`'MORPH'`: string of 01s,"" --> overwrites\\
>>`"ACTION"` --> overwrites\\
>>`"CHAIN"` --> overwrites\\
>>`"CVH"` --> overwrites\\
>>`"KSIZE"` --> overwrites\\
>>`"RETR"` --> overwrites\\
>>`"FIXEDRANGE"` --> overwrites

Note:
the priority of methods is as follows:
- luminosity: `FIXEDRANGE`(FR), `LUMXY`(XY)
    - lower bound: max(minimum XY, FR lower bound)
    - upper bound: min(maximum XY, FR upper bound)
- contour identification: `IDXY`, `AUTOID`

minimum parameters required to run in `show=False` mode (assuming defaults are used):
- either FIXEDRANGE or >=2 LUMXY points
- either AUTOID or >=1 IDXY points
#### Returns:
- None or the selected contour path, if `show=False`.

    """
    use("QtAgg")
    global ACTIVE_FITS  #:HDUList
    ACTIVE_FITS = fopen(fits_dir, "update")
    fontManager.addfont(fpath("python/jarvis/resources/FiraCodeNerdFont-Regular.ttf"))
    with rc_context(
        rc={
            "font.family": "FiraCode Nerd Font",
            "axes.unicode_minus": False,
            "toolbar": "None",
            "font.size": 8,
            "keymap.fullscreen": "",
            "keymap.home": "",
            "keymap.back": "",
            "keymap.forward": "",
            "keymap.pan": "",
            "keymap.zoom": "",
            "keymap.save": "",
            "keymap.quit": "",
            "keymap.grid": "",
            "keymap.yscale": "",
            "keymap.xscale": "",
            "keymap.copy": "",
        },
    ):
        # generate a stripped down, grey scale image of the fits file, and make normalised imagempl.rcParams['toolbar'] = 'None'
        proc = prep_polarfits(assign_params(ACTIVE_FITS, fixed="LON", full=True))
        img = azimuthal_equidistant(fits_obj=ACTIVE_FITS, clip_negatives=True)
        imagexy(proc, cmap=cmr.neutral, ax_background="white", img_background="black")
        global view_config, cv_config, active_selections, result, FIRST_RUN

        result = {
            "EXTNAME": persists.get("EXTNAME", headername),
            "path": persists.get("path"),
            "snapshots": {},
            "attrs": {},
        }
        if steps:
            result["snapshots"]["img_0.png"] = img.copy()
        oldlen = len(ACTIVE_FITS)

        if FIRST_RUN:
            if show:
                _get_pathfinderhead()  # tqdm.write('(Press # to print keybindings)')
            FIRST_RUN = False
            if show:
                view_config = {"REGISTER_KEYS": True, "cmap": 0, "tooltips": show_tooltips, "mask": 0}
            cv_config = {
                "ACTION": 0,
                "MORPH": [cv2.MORPH_OPEN, cv2.MORPH_CLOSE],
                "CHAIN": cv2.CHAIN_APPROX_SIMPLE,
                "CVH": False,
                "KSIZE": 5,
                "RETR": cv2.RETR_EXTERNAL,
                "FIXEDRANGE": None,
            }
            active_selections = {"LUMXY": [], "IDXY": [], "AUTOID": None}
        for k, v in cv_config.items():
            cv_config[k] = persists.get(k, persists.get(k.lower(), persists.get(k.upper(), v)))
        if show:
            for k, v in view_config.items():
                view_config[k] = persists.get(k, persists.get(k.lower(), persists.get(k.upper(), v)))
        for k, v in active_selections.items():
            active_selections[k] = persists.get(k, persists.get(k.lower(), persists.get(k.upper(), v)))
        for k, v in result.items():
            result[k] = persists.get(k, persists.get(k.lower(), persists.get(k.upper(), v)))

        global fixedrange_ghost  #:list[float]|None
        fixedrange_ghost = None
        normed = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        # set up global variables needed for the click event and configuration options
        result["attrs"] = {
            "LMIN": -np.inf,
            "LMAX": np.inf,
            "NUMPTS": np.nan,
            "XYA_CT": np.nan,
            "XYA_CTP": np.nan,
            "XYA_IMG": (np.pi * (normed.shape[0] / 2) ** 2) / 2,
        }

        def __approx_contour_area_pct(contourarea):
            num = contourarea / result["attrs"]["XYA_IMG"] * 100
            if num < 0.01:
                return f"{num:.2e}%"
            return f"{num:.2f}%"

        # set up the figure and axes
        if show:
            fig = plt.figure(figsize=(12.8, 7.2))
            qapp = fig.canvas.manager.window
            qapp.setWindowTitle(f"JAR:VIS Pathfinder ({fits_dir})")
            iconpath = fpath("python/jarvis/resources/aa_asC_icon.ico")
            qapp.setWindowIcon(QtGui.QIcon(iconpath))
            # gs: 1 row, 2 columns, splitting the figure into plot and options
            gs = fig.add_gridspec(1, 2, wspace=0.02, hspace=0, width_ratios=[4, 1.5], left=0, right=1, top=1, bottom=0)
            # mgs: 17 rows, 12 columns, to layout the options
            mgs = gs[1].subgridspec(
                12, 12, wspace=0, hspace=0, height_ratios=[0.2, 0.2, 0.15, 0.2, 0.2, 0.5, 0.5, 0.15, 0.05, 1, 1.5, 1.5],
            )
            # layout the axes in the gridspecs

            linfax = fig.add_subplot(mgs[0:2, :8], label="infotext")  # text box for information, top right
            clax = fig.add_subplot(mgs[0, 8:], label="CLOSE")  # close button
            sbax = fig.add_subplot(mgs[1, 8:], label="SAVE")  # save button
            headernameax = fig.add_subplot(mgs[2, 8:], label="headername")  # text box for header name
            rbax = fig.add_subplot(mgs[3, 8:], label="RESET")  # reset button
            fsax = fig.add_subplot(mgs[4, 8:], label="FSCRN")  # fullscreen button
            flax = fig.add_subplot(mgs[2:5, :8], label="ACTION", facecolor=_bgs["sidebar"])  # click mode selection
            retrax = fig.add_subplot(
                mgs[5, :8], label="RETR", facecolor=_bgs["sidebar"],
            )  # contour retrieval mode selection
            morphax = fig.add_subplot(
                mgs[5:7, 8:], label="MORPH", facecolor=_bgs["sidebar"],
            )  # morphological operations selection
            chainax = fig.add_subplot(
                mgs[6, :8], label="CHAIN", facecolor=_bgs["sidebar"],
            )  # contour approximation method selection
            ksizeax = fig.add_subplot(mgs[8, :8], label="KSIZE", facecolor=_bgs["sidebar"])  # kernel size selection
            cvhax = fig.add_subplot(mgs[7:9, 8:], label="CHAIN", facecolor=_bgs["sidebar"])  # convex hull selection
            ksizesubax = fig.add_subplot(mgs[7, :8], label="KSIZE", facecolor=_bgs["sidebar"])  # kernel size display
            fitinfoax = fig.add_subplot(
                mgs[9, :], label="fitinfo", facecolor=_bgs["sidebar"],
            )  # fit information display
            lax = fig.add_subplot(mgs[-1, :], label="legend", facecolor=_bgs["legend"])  # legend
            ax = fig.add_subplot(gs[0], label="main", zorder=11)  # main plot

            mmgs = mgs[10, :].subgridspec(2, 1, wspace=0, hspace=0, height_ratios=[10, 1])  # note textbox
            noteax = fig.add_subplot(
                mmgs[0], label="notes", zorder=-10, facecolor=_bgs["legend"],
            )  # fig.add_axes(ax.bbox.bounds, label='note', facecolor='#fff5', zorder=10)

            global noteattrs
            noteattrs = {"active": False, "text": "", "cursor": 0}
            kmap = {"enter": "\n", "space": " ", "tab": "\t"}
            _modind = ("caps", "shift", "ctrl", "alt")
            notebox = noteax.text(
                0,
                0.5,
                noteattrs["text"],
                fontsize=8,
                color="black",
                ha="left",
                va="center",
                wrap=True,
                bbox={"lw": 0},
                zorder=200,
            )
            noteax.set_axis_off()

            if show_tooltips:
                tooltipax = fig.add_subplot(mgs[10, :], label="tooltip", facecolor="#000")
                tooltip = tooltipax.text(
                    0.012,
                    0.98,
                    "",
                    fontsize=6,
                    color="black",
                    ha="left",
                    va="top",
                    wrap=True,
                    bbox={"facecolor": "white", "alpha": 0.5, "lw": 0},
                    zorder=10,
                )
            axs = [
                ax,
                lax,
                rbax,
                linfax,
                retrax,
                morphax,
                chainax,
                flax,
                sbax,
                cvhax,
                ksizeax,
                fitinfoax,
                clax,
                ksizesubax,
                fsax,
                noteax,
            ] + ([tooltipax] if show_tooltips else [])
            axgroups = {
                "buttons": [rbax, sbax, clax, fsax],
                "textboxes": [linfax, fitinfoax],
                "options": [retrax, morphax, chainax, cvhax, ksizeax, ksizesubax, flax],
                "legends": [lax],
            }

            linfax.text(
                0.5,
                0.5,
                "Select at least 2 or more\nLuminosity Samples",
                fontsize=8,
                color="black",
                ha="center",
                va="center",
            )
            # final plot initialisation
            ksizeax.set_facecolor("#fff")
            for a in axs:
                a.xaxis.set_visible(False)
                a.yaxis.set_visible(False)
            # fig.canvas.setWindowTitle('Luminance Viewer')
            fig.set_facecolor("black")
            ax.imshow(normed, cmap=cmap_cycler[view_config["cmap"]], zorder=0)

            def __redraw_displayed_attrs():
                global ACTIVE_FITS
                visit, tel, inst, filt, pln, hem, cml, doy = fitsheader(
                    ACTIVE_FITS, "VISIT", "TELESCOP", "INSTRUME", "FILTER", "PLANET", "HEMISPH", "CML", "DOY",
                )
                dt = get_datetime(ACTIVE_FITS)
                fitinfoax.text(0.04, 0.96, "FITS File Information", fontsize=10, color="black", ha="left", va="top")
                pth = str(fits_dir)
                pthparts = []
                if len(pth) > 30:
                    parts = split_path(rpath(pth))
                    while len(parts) > 0:
                        stick = parts.pop(-1)
                        if len(parts) > 0:
                            if len(stick) + len(parts[-1]) < 30:
                                parts[-1] += stick
                            else:
                                pthparts.append(stick)
                        else:
                            pthparts.append(stick)
                else:
                    pthparts.append(pth)
                pthparts.reverse()
                hdulinf = hdulinfo(ACTIVE_FITS, include_ver=True)

                def hdupart(hdul):
                    template = "{i} {name} ({type})"
                    lent = len(hdul)
                    hduc = [template.format(i=i, name=h["name"], type=h["type"]) for i, h in enumerate(hdul)]
                    # cols = lent//4 + (1 if lent%4 > 0 else 0)
                    rows = 4 if lent >= 7 else 3 if lent >= 3 else lent
                    maxwidth = max([len(h) for h in hduc])
                    hduc = [h.ljust(maxwidth) for h in hduc]
                    addempties = rows - (lent % rows)
                    if addempties < rows:
                        hduc += [" ".ljust(maxwidth) for _ in range(addempties)]

                    # split into 4 parts
                    hduc = [hduc[i::rows] for i in range(rows)]
                    hduc = [" ".join(hduc[i]) for i in range(rows)]
                    return [["HDUs", hduc[0]]] + [[" ", hduc[i]] for i in range(1, rows)]

                hdup = hdupart(hdulinf)
                ctxt = [
                    ["File", f"{pthparts[0]}"],
                    *[[" ", f"{p}"] for p in pthparts[1:]],
                    ["Inst.", f"{inst} ({tel}, {filt} filter)"],
                    ["Obs.", f"visit {visit} of {pln} ({hem})"],
                    ["Date", f'{dt.strftime("%d/%m/%Y")}, day {doy}'],
                    ["Time", f'{dt.strftime("%H:%M:%S")}(ICRS)'],
                    ["CML", f"{cml:.4f}°"],
                    *hdup,
                ]
                # ['HDUs',f"{0} {hdulinf[0]['name']}({hdulinf[0]['type']})"], *[[" ", f"{i} {h['name']}({h['type']})"] for i,h in enumerate(hdulinf[1:],1)]]
                tl = fitinfoax.table(
                    cellText=ctxt,
                    cellLoc="right",
                    cellColours=[[_bgs["sidebar"], _bgs["sidebar"]] for i in range(len(ctxt))],
                    colWidths=[0.14, 1],
                    colLabels=None,
                    rowLabels=None,
                    colColours=None,
                    rowColours=None,
                    colLoc="center",
                    rowLoc="center",
                    loc="bottom left",
                    bbox=[0.02, 0.02, 0.96, 0.8],
                    zorder=0,
                )
                tl.visible_edges = ""
                tl.auto_set_font_size(False)
                tl.set_fontsize(8)
                for key, cell in tl.get_celld().items():
                    cell.set_linewidth(0)
                    cell.PAD = 0
                    if key[0] < len(pthparts) and key[1] == 1:
                        cell.set_fontsize(max(8 - len(pthparts) + 1, 5))
                    if key[0] >= len(pthparts) + 5 and key[1] == 1:
                        cell.set_fontsize(max(8 - len(hdulinf) + 1, 5))
                fig.canvas.blit(fitinfoax.bbox)

            __redraw_displayed_attrs()

        def __generate_conf_output():
            """Returns the current configuration options in a dictionary, encoding the values as integers or binary strings to save to the fits file."""
            global cv_config
            m_ = [0] * 8
            for m in cv_config["MORPH"]:
                m_[m] = 1
            m_ = "".join([str(i) for i in m_])
            returnparts = {x: cv_config.get(x) for x in ["RETR", "CHAIN", "CVH", "KSIZE"]}
            returnparts |= {"MORPH": m_, "CONTTIME": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            return returnparts

        def __load_conf_output(conf):
            """Loads the configuration options from a dictionary, decoding the values from integers or binary strings."""
            global cv_config
            lrange_ = [0, 1]
            for k, v in conf.items():
                if k == "MORPH":
                    cv_config["MORPH"] = [i for i in range(8) if v[i] == "1"]
                elif k in cv_config:
                    cv_config[k] = v
                if k == "LMIN":
                    lrange_[0] = v
                elif k == "LMAX":
                    lrange_[1] = v
            if lrange_ != [0, 1]:
                cv_config["FIXEDRANGE"] = lrange_

        def __generate_coord_output():
            """Returns the current selected coordinates to save to the fits file."""
            global active_selections
            ret = {}
            for i, coord in enumerate(active_selections["LUMXY"]):
                ret[f"LUMXY_{i}"] = f"{coord[0]:.3f},({coord[1][0]},{coord[1][1]})"
            for i, coord in enumerate(active_selections["IDXY"]):
                ret[f"IDXY_{i}"] = f"({coord[0]},{coord[1]})"
            if active_selections["AUTOID"] is not None:
                ret["AUTOID"] = f"({active_selections['AUTOID'][0]},{active_selections['AUTOID'][1]})"
            return ret

        def __load_coord_output(conf):
            """Loads the selected coordinates from a dictionary."""
            global active_selections
            for k in active_selections:
                active_selections[k] = []
            for k, v in conf.items():
                if k.startswith("LUMXY"):
                    lum = float(v.split(",")[0])
                    x = int(v.split("(")[1].split(",")[0])
                    y = int(v.split(",")[1].split(")")[0])
                    active_selections["LUMXY"].append([lum, [x, y]])
                elif k.startswith("IDXY"):
                    xy = v.replace("(", "").replace(")", "").split(",")
                    active_selections["IDXY"].append([int(xy[0]), int(xy[1])])
                elif k == "AUTOID" or k == "AUTOCL":
                    xy = v.replace("(", "").replace(")", "").split(",")
                    active_selections["AUTOID"] = [int(xy[0]), int(xy[1])]

        if show:

            def __draw_contours(axes, contours, clinekws, ctextkws, limit=True):
                for i, contour in contours:
                    global __MAXCDRAWS
                    if i == 0:
                        __MAXCDRAWS = 0
                    if limit:
                        if __MAXCDRAWS > MAXCONTOURDRAWS and len(contour) < 2 and len(contour) > MAXPOINTSPERCONTOUR:
                            continue
                        __MAXCDRAWS += 1
                    axes.scatter(*contour.T, **clinekws)
                    # add the first point to the end to close the contour
                    contour = np.append(contour, [contour[0]], axis=0)
                    cpath = mPath(contour.reshape(-1, 2))
                    axes.plot(*cpath.vertices.T, color=clinekws["color"], lw=clinekws["s"])
                    axes.text(contour[0][0][0], contour[0][0][1], f"{i}", **ctextkws)

        # main update function
        def __redraw_main(_):
            global active_selections, cv_config, view_config, result
            if steps:
                result["snapshots"]["img_1.png"] = normed.copy()
            if show:
                # debug print(cl)
                for a in [ax, lax, linfax]:
                    a.clear()
                    a.set_facecolor(__get_bg(a))
                ax.imshow(normed, cmap=cmap_cycler[view_config["cmap"]], zorder=0)
                if (
                    len(active_selections["IDXY"]) > 0
                ):  # pxsc = [idpixels[0] for idpixels in active_selections["IDXY"]], [idpixels[1] for idpixels in active_selections["IDXY"]]
                    ax.scatter(*zip(*active_selections["IDXY"]), **_idpxkws)
                if (
                    len(active_selections["LUMXY"]) > 0
                ):  # scpc = [c[1][0] for c in active_selections["LUMXY"]], [c[1][1] for c in active_selections["LUMXY"]]
                    ax.scatter(*zip(*[c[1] for c in active_selections["LUMXY"]]), **_clickedkws)
            if len(active_selections["LUMXY"]) > 1 or cv_config["FIXEDRANGE"] is not None:
                lrange = cv_config["FIXEDRANGE"][0:2] if cv_config["FIXEDRANGE"] not in [None, "", []] else [0, 1]
                # print(lrange, active_selections["LUMXY"], cv_config["FIXEDRANGE"])
                if len(active_selections["LUMXY"]) > 1:
                    lrange = [
                        max(min(c[0] for c in active_selections["LUMXY"]), lrange[0]),
                        min(max(c[0] for c in active_selections["LUMXY"]), lrange[1]),
                    ]

                mask = cv2.inRange(normed, lrange[0] * 255, lrange[1] * 255)
                if steps:
                    result["snapshots"]["img_2.png"] = mask.copy()
                result["attrs"].update(
                    {
                        "LMIN": lrange[0],
                        "LMAX": lrange[1],
                        "LMETHOD": "FIXEDL" if lrange == cv_config["FIXEDRANGE"] else "IDPX",
                    },
                )
                # smooth out the mask to remove noise
                kernel = np.ones([cv_config["KSIZE"]] * 2, np.uint8)  # Small kernel to smooth edges
                for i, morph in enumerate(cv_config["MORPH"]):
                    mask = cv2.morphologyEx(mask, morph, kernel)
                    if steps:
                        result["snapshots"][f"img_2_{i}.png"] = mask.copy()

                if view_config["mask"] != 0 and show:
                    ax.imshow(
                        mask,
                        cmap=cmr.neutral
                        if view_config["mask"] in [1, 2]
                        else cmr.neutral_r
                        if view_config["mask"] in [3, 4]
                        else cmr.neon,
                        alpha=0.5 if view_config["mask"] in [1, 3, 5] else 1,
                        zorder=0.1,
                    )
                # find the contours of the mask
                contours, hierarchy = cv2.findContours(image=mask, mode=cv_config["RETR"], method=cv_config["CHAIN"])
                if cv_config["CVH"]:  # Convex hull of the contours which reduces the complexity of the contours
                    contours = [cv2.convexHull(cnt) for cnt in contours]
                sortedcs = sorted(contours, key=lambda x: cv2.contourArea(x))
                sortedcs.reverse()
                if show:
                    txt = ""
                    if cv_config["FIXEDRANGE"] is not None:
                        txt += (
                            ("FIXED" + f"({cv_config["FIXEDRANGE"][2]:.0e})")
                            if len(cv_config["FIXEDRANGE"]) > 2
                            else ""
                        )

                    linfax.text(
                        0.5,
                        0.75,
                        f"LRG=({lrange[0]:.4f},{lrange[1]:.4f})" + txt,
                        fontsize=8,
                        color="black",
                        ha="center",
                        va="center",
                    )
                # ------- Contour Selection -------#
                if len(active_selections["IDXY"]) > 0 or isinstance(
                    active_selections["AUTOID"], (list, tuple),
                ):  # if an id_pixel is selected, find the contour that encloses it
                    selected_contours = []
                    other_contours = []
                    if len(active_selections["IDXY"]) > 0:  # ------------ IDPX METHOD ------------#
                        for i, contour in enumerate(sortedcs):
                            if all(
                                cv2.pointPolygonTest(contour, id_pixel, False) > 0
                                for id_pixel in active_selections["IDXY"]
                            ):  # >0 means inside
                                selected_contours.append([i, contour])
                            else:
                                other_contours.append([i, contour])
                        # if a contour is found, convert the contour points to polar coordinates and return them
                    else:  # ------------ CLICK0 METHOD ------------#
                        co = [
                            (i, abs(cv2.pointPolygonTest(contour, active_selections["AUTOID"], True)))
                            for i, contour in enumerate(sortedcs)
                        ] + [(None, np.inf)]
                        chosen = min(co, key=lambda x: x[1])
                        if chosen[0] is not None:
                            selected_contours.append([chosen[0], sortedcs[chosen[0]]])
                        other_contours = [[i, contour] for i, contour in enumerate(sortedcs) if i != chosen[0]]
                    # ------- Contour Drawing (with selections) -------#
                    if show:
                        __draw_contours(ax, selected_contours, _selectclinekws, _selectctextkws)
                        __draw_contours(ax, other_contours, _otherclinekws, _otherctextkws)
                        linfax.text(
                            0.5,
                            0.5,
                            f"Contours: {len(selected_contours)}({len(other_contours)} invalid)",
                            fontsize=8,
                            color="black",
                            ha="center",
                            va="center",
                        )
                        if len(selected_contours) > 0:
                            result["path"] = selected_contours[0][1]
                            result["attrs"].update(
                                {
                                    "NUMPTS": len(result["path"]),
                                    "XYA_CT": cv2.contourArea(result["path"]),
                                    "XYA_CTP": __approx_contour_area_pct(cv2.contourArea(result["path"])),
                                },
                            )
                            linfax.text(
                                0.5,
                                0.25,
                                f'Area: {result["attrs"]["XYA_CTP"]}, N:{result["attrs"]['NUMPTS']}',
                                fontsize=10,
                                color="black",
                                ha="center",
                                va="center",
                            )
                        else:
                            result["path"] = None
                        selectedhandles = [
                            Line2D(
                                [0],
                                [0],
                                label=f"{i}, {__approx_contour_area_pct(cv2.contourArea(sortedcs[i]))}",
                                **_handles["selectedc"],
                            )
                            for i in [c[0] for c in selected_contours]
                        ]
                        otherhandles = [
                            Line2D(
                                [0],
                                [0],
                                label=f"{i}, {__approx_contour_area_pct(cv2.contourArea(sortedcs[i]))}",
                                **_handles["otherc"],
                            )
                            for i in [c[0] for c in other_contours]
                        ]
                        lax.legend(
                            handles=[*selectedhandles, *otherhandles][: min(MAXLEGENDITEMS, len(otherhandles) - 1)],
                            **_legendkws,
                        )
                else:  # ------- Contour Drawing (without selections) -------#
                    if show:
                        __draw_contours(ax, list(enumerate(sortedcs)), _otherclinekws, _otherctextkws)
                        linfax.text(
                            0.5, 0.5, f"Contours: {len(sortedcs)}", fontsize=8, color="black", ha="center", va="center",
                        )
                        handles = [
                            Line2D(
                                [0],
                                [0],
                                label=f"{i}, {__approx_contour_area_pct(cv2.contourArea(sortedcs[i]))}",
                                **_handles["defc"],
                            )
                            for i in range(len(sortedcs))
                        ][: min(MAXLEGENDITEMS, len(sortedcs) - 1)]
                        lax.legend(handles=handles, **_legendkws)
                        lax.set_facecolor(__get_bg(lax))
            else:  # ------- No Contours -------#
                if show:
                    linfax.text(
                        0.5,
                        0.5,
                        "Select at least 2 or more\nLuminosity Samples",
                        fontsize=8,
                        color="black",
                        ha="center",
                        va="center",
                    )
                    lax.set_facecolor(__get_bg(ax))
            if steps:
                fig.savefig(fpath("figures/img_3.png"))
            if show:
                fig.canvas.draw()
                fig.canvas.flush_events()

        def __load_conf(conf):
            """Loads the configuration and selected coordinates from a dictionary."""
            __load_conf_output(conf)
            __load_coord_output(conf)
            if show:
                __redraw_main(None)

        if "conf" in persists:
            __load_conf(persists["conf"])
        # config gui elements
        # ---- SAVE button ----#

        def __event_save(*_):
            global result, ACTIVE_FITS
            if result["path"] is not None:
                pth = result["path"].reshape(-1, 2)
                pth = fullxy_to_polar_arr(pth, normed, 40)
                if steps:
                    for k, v in result["snapshots"].items():
                        cv2.imwrite(fpath(f"figures/{k}"), v)
                # gcopy = G_fits_obj.copy()
                nhattr = result["attrs"]
                nhattr |= {}
                nhattr |= __generate_conf_output()
                nhattr |= __generate_coord_output()
                nhattr |= {m: fitsheader(ACTIVE_FITS, m) for m in ["UDATE", "YEAR", "VISIT", "DOY", "EXPT"]}
                if noteattrs["text"] not in ["", None]:
                    notes = noteattrs["text"].replace("\n", " " * 32)
                    notes = "".join([char if ord(char) < 128 else f"\\x{ord(char):02x}" for char in notes])
                    nhattr["NOTES"] = notes
                for k, v in nhattr.items():
                    if v in [np.nan, np.inf, -np.inf]:
                        raise KeyError(f"Key Mismatch: {v} is not a valid number, please check the value of {k}")
                header = Header(nhattr)
                extname = f'{result["EXTNAME"].upper()}'
                if extname in [hdu.name for hdu in ACTIVE_FITS]:
                    ver = max([hdu.ver for hdu in ACTIVE_FITS if hdu.name == extname]) + 1
                    ch = contourhdu(pth, name=extname, header=header, ver=ver)
                else:
                    ch = contourhdu(pth, name=extname, header=header)
                ACTIVE_FITS.append(ch)
                ACTIVE_FITS.flush()
                assert len(ACTIVE_FITS) == oldlen + 1, f"Save Failed: {len(ACTIVE_FITS)} != {oldlen+1}\n"
                tqdm.write(f"Save Successful, contour added to fits file at index {len(ACTIVE_FITS)-1}")
                linfax.clear()
                linfax.text(
                    0.5,
                    0.5,
                    "Save Successful, contour added\n to fits file at index" + str(len(ACTIVE_FITS) - 1),
                    fontsize=8,
                    color="black",
                    ha="center",
                    va="center",
                )
                fig.canvas.blit(linfax.bbox)
                __redraw_displayed_attrs()
            else:
                tqdm.write("Save Failed: No contour selected to save")

        if show:
            bsave = Button(sbax, "Save       " + "\ueb4b")
            bsave.on_clicked(__event_save)
            # ---- RESET button ----#
            breset = Button(rbax, "Reset      " + "\uf0e2")

            def __event_reset(*_):
                global active_selections, view_config, cv_config
                for key in active_selections:
                    active_selections[key] = []
                for key in cv_config:
                    if key in ["MORPH", "RETR", "CHAIN", "CVH", "KSIZE"]:
                        cv_config[key] = persists.get(key, cv_config[key])
                for key in view_config:
                    if key in ["mask", "cmap"]:
                        view_config[key] = persists.get(key, view_config[key])

                __redraw_main(None)

            breset.on_clicked(__event_reset)
            # ---- RETR options ----#
            retrbts = RadioButtons(retrax, __getflaglabels("RETR"), active=0)

            def __event_retr(label):
                """set the global variable G_fcmode to the correct value based on the label"""
                global cv_config
                cv_config["RETR"] = _cvtrans["RETR"][label][0]

            retrbts.on_clicked(__event_retr)
            # ---- MORPH options ----#
            bopts = __getflaglabels("MORPH")
            acts = [_cvtrans["MORPH"][b][0] in cv_config["MORPH"] for b in bopts]
            morphbts = CheckButtons(morphax, bopts, acts)

            def __event_morphex(lb):
                """toggle the morphological operations in G_morphex based on the label, add if not present, remove if present"""
                global cv_config
                if _cvtrans["MORPH"][lb][0] in cv_config["MORPH"]:
                    cv_config["MORPH"].remove(_cvtrans["MORPH"][lb][0])
                else:
                    cv_config["MORPH"].append(_cvtrans["MORPH"][lb][0])

            morphbts.on_clicked(__event_morphex)
            # ---- CHAIN options ----#
            chainbts = RadioButtons(
                chainax, __getflaglabels("CHAIN"), active=_cvtrans["CHAIN"]["trans"][cv_config["CHAIN"]],
            )

            def __event_chain(label):
                """set the global variable G_fcmethod to the correct value based on the label"""
                global cv_config
                cv_config["CHAIN"] = _cvtrans["CHAIN"][label][0]

            chainbts.on_clicked(__event_chain)
            # ---- CLOSE button ----#
            bclose = Button(clax, "Close      " + "\ue20d", color="#ffaaaa", hovercolor="#ff6666")

            # bclose.label.set(fontsize=14)
            def __event_close(*_):
                """close the figure"""
                plt.close()

            bclose.on_clicked(__event_close)
            # ---- CVH options ----#
            cvhbts = CheckButtons(cvhax, ["Convex Hull"], [cv_config["CVH"]])

            def __event_cvh(*_):
                """toggle the G_cvh variable"""
                global cv_config
                cv_config["CVH"] = not cv_config["CVH"]

            cvhbts.on_clicked(__event_cvh)
            # ---- KSIZE options ----#
            ksizeslider = Slider(
                ksizeax,
                "",
                valmin=1,
                valmax=11,
                valinit=cv_config["KSIZE"],
                valstep=2,
                color="orange",
                initcolor="None",
            )
            ksizeax.add_patch(
                Rectangle(
                    (0, 0), 1, 1, facecolor="#fff", lw=1, zorder=-1, transform=ksizeax.transAxes, edgecolor="black",
                ),
            )
            ksizeax.add_patch(
                Rectangle((0, 0.995), 5 / 12, 1.005, facecolor="#fff", lw=0, zorder=10, transform=ksizeax.transAxes),
            )

            def __event_ksize(val):
                """set the global variable G_ksize to the slider value, and update the displayed value"""
                global cv_config
                ksizeax.set_facecolor("#fff")
                cv_config["KSIZE"] = int(val)
                ksizesubax.clear()
                ksizesubax.text(
                    0.1, 0.45, f'ksize: {cv_config["KSIZE"]}', fontsize=8, color="black", ha="left", va="center",
                )
                fig.canvas.blit(ksizesubax.bbox)
                __redraw_main(None)

            # __event_ksize(cv_config["KSIZE"])
            ksizeslider.on_changed(__event_ksize)
            # ---- SELECTMODE options ----#
            _radioprops = {
                "facecolor": ["#000", _clickedkws["color"], _clickedkws["color"], _idpxkws["color"], _idpxkws["color"]],
                "edgecolor": ["#000", _clickedkws["color"], _clickedkws["color"], _idpxkws["color"], _idpxkws["color"]],
                "marker": ["o", "o", "X", "o", "X"],
            }
            _labelprops = {
                "color": ["#000", _clickedkws["color"], _clickedkws["color"], _idpxkws["color"], _idpxkws["color"]],
            }
            selradio = RadioButtons(
                flax,
                ["None", "Add Luminosity", "Remove Luminosity", "Add IDPX", "Remove IDPX"],
                active=0,
                radio_props=_radioprops,
                label_props=_labelprops,
            )

            def __event_select(label):
                """set the global variable G_clicktype to the correct value based on the label"""
                global cv_config
                ret = 0
                if "Luminosity" in label:
                    ret = 1
                elif "IDPX" in label:
                    ret = 2
                if "Remove" in label:
                    ret = -ret
                cv_config["ACTION"] = ret

            selradio.on_clicked(__event_select)
            # ---- FULLSCREEN button ----#
            bfullscreen = Button(fsax, "Fullscreen " + "\uf50c")

            # bfullscreen.label.set(fontsize=14)
            def __event_fullscreen(*_):
                fig.canvas.manager.full_screen_toggle()

            bfullscreen.on_clicked(__event_fullscreen)
            if "conf" in persists:
                __load_conf(persists["conf"])

            # ---- CLICK options ----#
            def __on_click_event(event):
                global active_selections, cv_config, REGISTER_KEYS
                REGISTER_KEYS = event.inaxes != headernameax
                if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
                    click_coords = (int(event.xdata), int(event.ydata))
                    # get the luminance value of the clicked pixel
                    if cv_config["ACTION"] == 1:  # add luminosity
                        active_selections["LUMXY"].append([img[int(event.ydata), int(event.xdata)] / 255, click_coords])
                    elif cv_config["ACTION"] == -1:  # remove luminosity
                        if len(active_selections["LUMXY"]) > 0:
                            active_selections["LUMXY"].pop(
                                np.argmin(
                                    [
                                        np.linalg.norm(np.array(c[1]) - np.array(click_coords))
                                        for c in active_selections["LUMXY"]
                                    ],
                                ),
                            )
                    elif cv_config["ACTION"] == 2:  # add id pixel
                        active_selections["IDXY"].append(click_coords)
                    elif cv_config["ACTION"] == -2:  # remove id pixel
                        if len(active_selections["IDXY"]) > 0:
                            active_selections["IDXY"].pop(
                                np.argmin(
                                    [
                                        np.linalg.norm(np.array(c) - np.array(click_coords))
                                        for c in active_selections["IDXY"]
                                    ],
                                ),
                            )
                    elif (
                        cv_config["ACTION"] == 0 and event.button == 3
                    ):  # if not in any mode, right click to auto select the closest contour
                        active_selections["AUTOID"] = click_coords
                __redraw_main(None)

            click_event = fig.canvas.mpl_connect("button_press_event", __on_click_event)  # noqa: F841

            # ---- HOVER options ----#
            def __on_mouse_move_event(event):
                inax = event.inaxes
                global view_config
                if view_config["tooltips"] and inax in axgroups["options"] + axgroups["buttons"]:
                    axlabel = inax.get_label()
                    txts = [t for t in [t for t in inax.get_children() if isinstance(t, Text)] if t.get_text() != ""]
                    # now find if the mouse is over any of the texts
                    txt = None
                    for t in txts:
                        if t.contains(event)[0]:
                            txt = t
                            break
                    if txt is not None:
                        item = txt.get_text()
                        try:
                            lbl, inf = [
                                f"{axlabel}.{item}",
                                f"{_cvtrans[axlabel][item][2]} [key: { _cvtrans[axlabel][item][1]}]",
                            ]
                        except KeyError:
                            trans = _cvtrans[axlabel]
                            lbl, inf = [axlabel, trans["info"]]
                            if len(trans) == 3:
                                inf += f"[key: {trans[axlabel][1]}]"
                    else:
                        trans = _cvtrans[axlabel]
                        lbl, inf = [axlabel, trans["info"]]
                        if len(trans) == 3:
                            inf += f"[key: {trans[axlabel][1]}]"

                    tooltip.set_text(f"{lbl}: {inf}")
                    fig.canvas.draw_idle()
                    fig.canvas.blit(tooltipax.bbox)
                if inax == ax:
                    fig.canvas.set_cursor(3)
                elif inax in axgroups["options"] + axgroups["buttons"]:
                    fig.canvas.set_cursor(2)
                else:
                    fig.canvas.set_cursor(1)

            hover_event = fig.canvas.mpl_connect("motion_notify_event", __on_mouse_move_event)  # noqa: F841
            # ----TEXTENTRY----#
            textbox = TextBox(headernameax, "Header Name:", initial=result["EXTNAME"])
            delchars = "".join(c for c in map(chr, range(1114111)) if not c.isalnum() and c not in [" ", "_", "-"])
            delmap = str.maketrans("", "", delchars)

            def __on_headername_text_change(text: str):
                global result
                text = text.translate(delmap)
                result["EXTNAME"] = text.replace(" ", "_").replace()

            textbox.on_submit(__on_headername_text_change)
            # ----NOTES----#

            # ----KEYEVENTS----#

            def __on_key_press_event(event):
                global view_config, noteattrs
                if noteattrs["active"]:
                    if event.key == [k for k, v in _keybindings.items() if v[0] == "NOTES"][0]:
                        noteattrs["active"] = False  # process the keybinding while in notes mode to exit notes mode
                        noteax.set(zorder=-10)
                        notebox.set(zorder=-10)
                        fig.canvas.blit(noteax.bbox)
                        notebox.set(bbox={"facecolor": "#fff0", "edgecolor": "#fff0"}, color="#fff0")
                        noteax.set(facecolor="#fff0")
                        view_config["REGISTER_KEYS"] = True
                    elif event.key in _modind:  # process modifiers
                        pass
                    else:  # process the keypress if not modifier or exit key
                        if notebox.zorder < ax.zorder:
                            notebox.set_zorder(ax.zorder + 2)
                        if noteax.zorder < ax.zorder:
                            noteax.set_zorder(
                                ax.zorder + 1,
                            )  # (this is a hack to make sure the text is on top of the axes when it is important)
                        elif event.key == "backspace":
                            if noteattrs["cursor"] != len(noteattrs["text"]):
                                noteattrs["text"] = (
                                    noteattrs["text"][: len(noteattrs["text"]) - noteattrs["cursor"] - 1]
                                    + noteattrs["text"][len(noteattrs["text"]) - noteattrs["cursor"] :]
                                )
                            noteattrs["cursor"] = min(noteattrs["cursor"], len(noteattrs["text"]))
                        elif event.key in ["left", "right"]:
                            if event.key == "left":
                                noteattrs["cursor"] = min(noteattrs["cursor"] + 1, len(noteattrs["text"]))
                            elif event.key == "right":
                                noteattrs["cursor"] = max(noteattrs["cursor"] - 1, 0)
                        elif len(event.key) == 1 or event.key in kmap:
                            k = kmap.get(event.key, event.key)
                            noteattrs["text"] = (
                                noteattrs["text"][: len(noteattrs["text"]) - noteattrs["cursor"]]
                                + k
                                + noteattrs["text"][len(noteattrs["text"]) - noteattrs["cursor"] :]
                            )
                            noteattrs["cursor"] = min(noteattrs["cursor"], len(noteattrs["text"]))
                        text = (
                            noteattrs["text"][: len(noteattrs["text"]) - noteattrs["cursor"]]
                            + "|"
                            + noteattrs["text"][len(noteattrs["text"]) - noteattrs["cursor"] :]
                        )
                        notebox.set_text("Notes:\n" + text)
                        fig.canvas.blit(noteax.bbox)
                        fig.canvas.draw()

                elif view_config["REGISTER_KEYS"]:
                    if event.key in _keybindings:
                        action, val, _ = _keybindings[event.key]
                        if action == "ACTION":
                            selradio.set_active(__getflagindex("ACTION", val))
                        elif action == "RETR":
                            retrbts.set_active(__getflagindex("RETR", val))
                        elif action == "MORPH":
                            morphbts.set_active(__getflagindex("MORPH", val))
                        elif action == "CHAIN":
                            chainbts.set_active(__getflagindex("CHAIN", val))
                        elif action == "CVH":
                            cvhbts.set_active(0)
                        elif action == "KSIZE":
                            ksizeslider.set_val(min(max(cv_config["KSIZE"] + (2 if val == "Increase" else -2), 1), 11))
                        elif action == "SAVE":
                            __event_save(None)
                        elif action == "SAVECLOSE":
                            __event_save(None)
                            __event_close(None)
                        elif action == "RESET":
                            __event_reset(None)
                        elif action == "FSCRN":
                            __event_fullscreen(None)
                        elif action == "CLOSE":
                            __event_close(None)
                        elif action == "TOOLTIP":
                            if val == "TOOLTIP":
                                view_config["tooltips"] = not view_config["tooltips"]
                            elif val == "ONSCREEN":
                                __show_key_menu(True)
                                return
                            else:
                                return
                        elif action == "KILL":
                            tqdm.write("PATHFINDER: closing process.")
                            plt.close()
                            exit()
                        elif action == "MASK":
                            view_config["mask"] += 1
                            if view_config["mask"] >= 7:
                                view_config["mask"] = 0
                        elif action == "CMAP":
                            view_config["cmap"] += 1
                            if view_config["cmap"] >= len(cmap_cycler):
                                view_config["cmap"] = 0

                        elif action == "NOTES":
                            noteattrs["active"] = True
                            view_config["REGISTER_KEYS"] = True
                            if notebox.zorder < ax.zorder:
                                notebox.set_zorder(ax.zorder + 2)
                            if noteax.zorder < ax.zorder:
                                noteax.set_zorder(ax.zorder + 1)
                            notebox.set_text("Notes:\n" + noteattrs["text"] + "_")
                            noteax.set_facecolor("#fff0")
                            notebox.set(bbox={"facecolor": "#fff0", "edgecolor": "#fff"}, color="#fff")
                            fig.canvas.blit(noteax.bbox)
                            fig.canvas.draw()
                        elif action == "FIXLRANGE":
                            val = __getflagindex("FIXLRANGE", val)
                            if val == 4:
                                global fixedrange_ghost
                                if cv_config["FIXEDRANGE"] is None:
                                    cv_config["FIXEDRANGE"] = (
                                        fixedrange_ghost if fixedrange_ghost is not None else [0, 1, 0.01]
                                    )
                                else:
                                    fixedrange_ghost = cv_config["FIXEDRANGE"]
                                    cv_config["FIXEDRANGE"] = None
                            elif val < 4:
                                ind = int(val > 1)  # 0 for min, 1 for max val::{L+,L-,U+,U-,T,R}
                                if cv_config["FIXEDRANGE"] is None:
                                    if fixedrange_ghost is None:
                                        cv_config["FIXEDRANGE"] = [0, 0.3, 0.01]
                                    else:
                                        return
                                if len(cv_config["FIXEDRANGE"]) == 2:
                                    cv_config["FIXEDRANGE"].append(0.01)
                                step = cv_config["FIXEDRANGE"][2]
                                cv_config["FIXEDRANGE"][ind] += step * (1 - 2 * (val % 2))
                                cv_config["FIXEDRANGE"][ind] = min(max(cv_config["FIXEDRANGE"][ind], 0), 1)

                            elif val == 5:
                                if cv_config["FIXEDRANGE"] is None:
                                    fixedrange_ghost = persists.get("FIXEDRANGE", [0, 0.3, 0.01])
                                else:
                                    fixedrange_ghost = None
                                    cv_config["FIXEDRANGE"] = persists.get("FIXEDRANGE", [0, 0.3, 0.01])
                            elif val == 6:
                                if len(cv_config["FIXEDRANGE"]) == 2:
                                    cv_config["FIXEDRANGE"].append(0.01)
                                cv_config["FIXEDRANGE"][2] = stepmap[
                                    stepmap.index(cv_config["FIXEDRANGE"][2]) + 1
                                    if stepmap.index(cv_config["FIXEDRANGE"][2]) < len(stepmap) - 1
                                    else 0
                                ]
                    if event.key in HELPKEYS:
                        tt = [(k, v[2]) for k, v in _keybindings.items()]
                        maxlenk = max([len(t[0]) for t in tt])
                        maxlenv = max([len(t[1]) for t in tt])
                        template = f"║ {{k:^{maxlenk+2}}} ║ {{action:<{maxlenv+2}}} ║"
                        top = f'╔═{"═"*(maxlenk+2) }═╦═{"═"*(maxlenv+2)      }═╗'
                        bottom = f'╚═{"═"*(maxlenk+2) }═╩═{"═"*(maxlenv+2)      }═╝'
                        mid = f'╠═{"═"*(maxlenk+2) }═╬═{"═"*(maxlenv+2)      }═╣'
                        if TWOCOLS:
                            lenh = len(tt) // 2
                            tt1 = tt[:lenh]
                            tt2 = tt[lenh:]
                            sep = "\t"
                            while len(tt1) != len(tt2):
                                if len(tt1) > len(tt2):
                                    tt2.append(("", ""))
                                elif len(tt1) < len(tt2):
                                    tt1.append(("", ""))
                                else:
                                    break
                            title = template.format(k="Key", action="Action")
                            forms = [p + sep + p for p in [top, title, mid]]
                            for i in range(len(tt1)):
                                forms.append(
                                    template.format(k=tt1[i][0], action=tt1[i][1])
                                    + sep
                                    + template.format(k=tt2[i][0], action=tt2[i][1]),
                                )
                            forms.append(bottom)
                            for f in forms:
                                tqdm.write(f)
                        else:
                            tqdm.write(top)
                            tqdm.write(template.format(k="Key", action="Action"))
                            tqdm.write(mid)
                            for k, action in tt:
                                tqdm.write(template.format(k=k, action=action))
                            tqdm.write(bottom + sep + bottom)
                    __redraw_main(None)

            global menulock
            menulock = False

            def __show_key_menu(_):
                global keybind, tab, menulock
                menulock = not menulock
                if menulock:
                    bboxord = [-0.1, 0, 1.1, 1]
                    keybind = ax.add_patch(
                        Rectangle(
                            bboxord[0:2],
                            bboxord[2],
                            bboxord[3],
                            facecolor="#fffa",
                            lw=1,
                            zorder=10,
                            edgecolor="black",
                            transform=ax.transAxes,
                            clip_on=False,
                        ),
                    )
                    keys, values = zip(*_keybindings.items())
                    coltexts = [
                        keys[: len(keys) // 2],
                        [v[2] for v in values[: len(keys) // 2]],
                        keys[len(keys) // 2 :],
                        [v[2] for v in values[len(keys) // 2 :]],
                    ]
                    coltexts = [list(c) for c in coltexts]
                    coltexts[0] = [f"{k} " "\u21e8" for k in coltexts[0]]
                    coltexts[2] = [f"{k} " "\u21e8" for k in coltexts[2]]
                    keylen = max([len(k) for k in keys])
                    v2len = max([len(v[2]) for v in values])
                    coltexts = [
                        [f"{k:>{keylen}}" for k in coltexts[0]],
                        [f" {v:<{v2len}}" for v in coltexts[1]],
                        [f"{k:<{keylen}}" for k in coltexts[2]],
                        [f" {v:<{v2len}}" for v in coltexts[3]],
                    ]
                    while len(coltexts[0]) != len(coltexts[2]):
                        if len(coltexts[0]) > len(coltexts[2]):
                            coltexts[2].append("")
                            coltexts[3].append("")
                        elif len(coltexts[0]) < len(coltexts[2]):
                            coltexts[0].append("")
                            coltexts[1].append("")
                        else:
                            break
                    v = [*zip(*coltexts)]
                    tab = ax.table(
                        cellText=v[1:],
                        cellLoc="left",
                        colLoc="left",
                        cellColours=[["#fff5"] * len(coltexts)] * len(coltexts[0][1:]),
                        colWidths=[0.1, 0.4] * (len(coltexts) // 2),
                        colColours=["#fff5"] * len(coltexts),
                        zorder=11,
                        bbox=bboxord,
                        edges="",
                        colLabels=v[0],
                        transform=ax.transAxes,
                    )
                    tab.AXESPAD = 0
                    tab.auto_set_column_width([0, 1])
                    tab.auto_set_column_width([2, 3])
                    tab.auto_set_font_size(True)

                    for (i, j), cell in tab.get_celld().items():
                        if i in [0, 2]:
                            cell.set_text_props(weight="800")
                        cell.PAD = 0
                        cell.set_fontsize(cell.get_fontsize() * 1.2)
                    tab.auto_set_font_size(False)

                    fig.canvas.blit(ax.bbox)
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                else:
                    __redraw_main(None)

            key_event = fig.canvas.mpl_connect("key_press_event", __on_key_press_event)  # noqa: F841

            plt.show()

        else:
            __redraw_main(None)
            if result["path"] is None:
                plt.close()
            else:
                __event_save(None)
                plt.close()

        if result["path"] is not None:
            pth = result["path"].reshape(-1, 2)
            ret = [True, fullxy_to_polar_arr(pth, normed, 40)]
        else:
            ret = [False, [__generate_conf_output(), __generate_coord_output()]]
        if show and noteattrs["text"] not in ["", None]:
            tqdm.write(f'{filename_from_path(fits_dir)}:\n {noteattrs["text"]}')
        from contextlib import suppress

        with suppress(Exception):
            ACTIVE_FITS.close()

        return ret


# ------------ QuickPlot Class (for power.py optional plotting) ----------------#
class QuickPlot:
    """Helper class for plotting image arrays in various formats, used in the power.py module when the plotting flag is set."""

    titles = {
        "raw": "Image array in fits file.",
        "limbtrimmed": "Image centred, limb-trimmed.",
        "extracted": "Auroral region extracted.",
        "polar": "Polar projection.",
        "sqr": "ROI",
        "mask": "ROI mask in image space",
        "brojected_full": "Brojected full image.",
        "brojected_roi": "Brojected ROI image.",
    }
    labelpairs = {
        "pixels": {"xlabel": "pixels", "ylabel": "pixels"},
        "degpixels": {"xlabel": "longitude pixels", "ylabel": "co-latitude pixels"},
        "deg": {"xlabel": "SIII longitude [deg]", "ylabel": "co-latitude [deg]"},
    }

    def __init__(self):
        """Helper class for plotting image arrays in various formats, used in the power.py module when the plotting flag is set."""
        pass

    def __imcbar(
        self, ax, img, cmap="cubehelix", origin="lower", vlim=(1.0, 1000.0), clabel="Intensity [kR]", fs=12, pad=0.05,
    ):
        im = ax.imshow(img, cmap=cmap, origin=origin, vmin=vlim[0], vmax=vlim[1])
        cbar = plt.colorbar(pad=0.05, ax=ax, mappable=im)
        cbar.ax.set_ylabel(clabel, fontsize=fs, labelpad=pad)

    def _plot_(
        self, ax, image, cmap="cubehelix", origin="lower", vlim=(1.0, 1000.0),
    ):  # make a quick-look plot to check image array content:
        ax.set(title="Image array in fits file.", **self.labelpairs["degpixels"])
        self.__imcbar(ax, image, cmap=cmap, origin=origin, vlim=vlim)
        return ax

    def _plot_raw_limbtrimmed(
        self, ax, image, lons=np.arange(0, 1440, 1), lats=np.arange(0, 720, 1), cmap="cubehelix", vlim=(0.0, 1000.0),
    ):  # Quick plot check of the centred, limb-trimmed image:
        ax.set(title="Image centred, limb-trimmed.", **self.labelpairs["degpixels"])
        ax.pcolormesh(lons, lats, np.flip(np.flip(image, axis=1)), cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        return ax

    def _plot_extracted_wcoords(
        self, ax: plt.Axes, image: np.ndarray, cmap: str = "cubehelix", vlim: list = (0.0, 1000.0),
    ) -> plt.Axes:
        ax.set(title="Auroral region extracted.", **self.labelpairs["deg"])
        self.__imcbar(ax, image, cmap=cmap, vlim=vlim)
        ax.set_yticks(ticks=[0 * 4, 10 * 4, 20 * 4, 30 * 4, 40 * 4], labels=["0", "10", "20", "30", "40"])
        ax.set_xticks(ticks=[0 * 4, 90 * 4, 180 * 4, 270 * 4, 360 * 4], labels=["360", "270", "180", "90", "0"])
        return ax

    def _plot_polar_wregions(
        self,
        ax: plt.Axes,
        image: np.ndarray,
        cpath: np.ndarray,
        rho: Iterable[float] = np.linspace(0, 40, num=160),
        theta: Iterable[float] = np.linspace(0, 2 * np.pi, num=1440),
        fs: int = 12,
        cmap: str = "cubehelix",
        vlim: list = (1.0, 1000.0),
    ) -> plt.Axes:
        # ==============================================================================
        # make polar projection plot
        # ==============================================================================
        # set up polar coords
        # rho    # colat vector with image pixel resolution steps
        # theta  # longitude vector in radian space and image pixel resolution steps
        dr = [[i[0] for i in cpath], [i[1] for i in cpath]]
        ax.set(title="Polar projection.", theta_zero_location="N", ylim=[0, 40], **self.labelpairs["deg"])
        ax.set_yticks(ticks=[0, 10, 20, 30, 40], labels=["", "", "", "", ""])
        ax.set_xticklabels(
            ["0", "315", "270", "225", "180", "135", "90", "45"],  # reverse these! ###################
            fontweight="bold",
            fontsize=fs,
        )
        ax.tick_params(axis="x", pad=-1.0)
        ax.fill_between(theta, 0, 40, alpha=0.2, hatch="/", color="gray")
        cm = ax.pcolormesh(theta, rho, image, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        ax.plot([np.radians(360 - r) for r in dr[1]], dr[0], color="red", linewidth=3.0)
        # -> OVERPLOT THE ROI IN POLAR PROJECTION HERE. <-
        # Add colourbar: ---------------------------------------------------------------
        cbar = plt.colorbar(ticks=[0.0, 100.0, 500.0, 900.0], pad=0.05, mappable=cm, ax=ax)
        cbar.ax.set_yticklabels(["0", "100", "500", "900"])
        cbar.ax.set_ylabel("Intensity [kR]", fontsize=12)
        return ax

    def _plot_sqr_wregions(
        self,
        ax,
        image,
        cpath,
        testlons=np.arange(0, 360, 0.25),
        testcolats=np.arange(0, 40, 0.25),
        cmap="cubehelix",
        vlim=(1.0, 1000.0),
    ):  # ?sq
        # # === Plot full image array using pcolormesh so we can understand the masking
        # # process i.e. isolating pixels that fall inside the ROI =======================
        ax.set(title="ROI", xlabel="SIII longitude [deg]", ylabel="co-latitude [deg]")
        ax.set_xticks(ticks=[0, 90, 180, 270, 360], labels=["360", "270", "180", "90", "0"])
        dr = [[i[0] for i in cpath], [i[1] for i in cpath]]
        ax.pcolormesh(testlons, testcolats, image, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
        ax.plot([360 - r for r in dr[1]], dr[0], color="red", linewidth=3.0)
        return ax

    def _plot_mask(self, ax, image, loc="ROI"):  # ?
        ax.imshow(image, origin="lower")
        ax.set(title=f"{loc} mask in image space", xlabel="longitude pixels", ylabel="co-latitude pixels")
        return ax

    def _plot_brojected(
        self, ax, image, loc="ROI", cmap="cubehelix", origin="lower", vlim=(1.0, 1000.0), saveto=None,
    ):  # ?
        ax.set(title=f"Brojected {loc} image.", **self.labelpairs["pixels"])
        self.__imcbar(ax, image, cmap=cmap, origin=origin, vlim=vlim)
        if saveto is not None:
            plt.savefig(saveto, dpi=350)
        return ax


def _get_pathfinderhead():
    # stdscr=curses.initscr()
    # curses.noecho()
    # curses.cbreak()
    # rows, cols = stdscr.getmaxyx()
    lines = [
        "₊------------------------------------------------------₊",
        "|                   JAR:VIS Pathfinder                 |",
        "⁺------------------------------------------------------⁺",
        f"(press {" or ".join(HELPKEYS)} for keybindings.)",
    ]
    for line in lines:
        tqdm.write(line)


def extract_conf(header: Header, ignore: list = ()) -> dict:
    """Extracts the configuration from the header of a fits file, into a dictionary with unique keys. values like lists are not parsed until loaded, and will still be strings."""
    ignore = [ignore] if isinstance(ignore, str) else ignore
    conf = {}
    for keyword in header:
        # print(keyword, header[keyword])
        if any(keyword.startswith(i) for i in ignore):
            continue
        if keyword not in conf:
            conf[keyword] = header[keyword]
        else:
            conf[keyword + str(len([k for k in conf if k.startswith(keyword)]))] = header[keyword]
    return conf

    # header ===extract_conf===> dict with unique keys ===pathfinder--load_conf===> converts to values
