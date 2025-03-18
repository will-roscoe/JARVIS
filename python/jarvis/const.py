#!/usr/bin/env python3
"""Constants and default values used throughout the project."""
#ruff: noqa
import logging
from pathlib import Path
import cmasher as cmr
import cv2
from os.path import exists
from astropy import units as u
FITSINDEX = 1 # DEFAULT FITSINDEX: the default target HDU within a fits file.
# ~ defines the project root directory (as the root of the gh repo)
GHROOT = Path(__file__).parents[2]# ~ if you move this file/folder, you need to change this line to match the new location.
# ~ index matches the number of folders to go up from where THIS file is located: /[2] python/[1] jarvis/[0] const.py
class ConfigLike:

    """A class containing constants for a file or group of functions."""

    def __init__(self,doc=""):
        """Initialize a config object, assigning a doc."""
        self.__doc__ = doc
        self.__dict__ = {}  # Stores attributes dynamically

    def __setattr__(self, name, value):
        """Set an attribute, eg abc.xyz = '123'."""
        if not name.startswith("__"):
            self.__dict__[name] = value  # Set attributes dynamically
            self.__doc__ += f"\n{name}: {str(value).replace("\n", "")}"

    def __getattr__(self, name):
        """Get an attribute, eg x= abc.xyz."""
        if name in self.__dict__:
            return self.__dict__[name]  # Get attributes dynamically
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    def __repr__(self) -> str:
        return 'Config:'+{k:v for k,v, in self.__dict__.items if not k.startswith("__") }
    def __str__(self):
        return "Config:\n"+self.__doc__

class ExpandingList:
    def __init__(self, items, overflow):
        self._data = tuple(items)
        self._overflow = overflow
    def __getitem__(self, index):
        if index >= len(self._data):
            return self._overflow
        return self._data[index]
    def __len__(self):
        return len(self._data)
    
    

    

######## Dirs ##################################################################
# DEFAULT PATHS
Dirs = ConfigLike("Default paths for the project")
Dirs.ROOT = GHROOT
Dirs.DATA = GHROOT / "datasets"
Dirs.PY = GHROOT / "python"
Dirs.JARVIS = Dirs.PY / "jarvis"
Dirs.KERNEL = "datasets/kernels/"
Dirs.HST = Dirs.DATA / "HST"
Dirs.HISAKI = Dirs.DATA / "Hisaki"
Dirs.TORUS = Dirs.HISAKI / "Torus Power"
Dirs.AURORA = Dirs.HISAKI / "Aurora Power"
Dirs.TEMP = GHROOT / "temp"
Dirs.GEN = Dirs.DATA / "Generated"
Dirs.gv_map ={"group": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "visit": [1, 2, 5, 4, 3, 8, 9, 10, 13, 15, 11, 12, 16, 18, 19, 20, 21, 23, 24, 29]}
######## DPR ###################################################################
DPR = ConfigLike("DPR region coordinates for HST images")
# XY COORDS FOR IN DPR REGION OF IMAGES
DPR.IMXY = { # dictionary containing manual DPR identifying poinits per group
    "01": (584, 1098),
    "02": (592, 1150),
    "03": (742, 1413),
    "04": (600, 1233),
    "05": ("", ""),
    "06": (674, 1233),
    "07": (607, 1173),
    "08": (742, 1390),
    "09": ("", ""),
    "10": (600, 910),
    "11": (622, 1061),
    "12": (584, 1046),
    "13": (570, 1083),
    "14": (614, 1241),
    "15": (660, 1309),
    "16": (664, 1264),
    "17": (750, 1391),
    "18": (614, 971),
    "19": (607, 1159),
    "20": ("", ""),
}
DPR.JSON = Dirs.DATA / "HST" / "custom" / "boundarypaths.json"
######## CONST #################################################################
CONST = ConfigLike("Specific scientific constants for the project")
CONST.au_to_km = 1.495978707e8
CONST.gustin_factor = 9.04e-10 # 1.02e-9 #> "Conversion factor to be multiplied by the squared HST-planet distance (km) to determine the total emitted power (Watts) from observed counts per second."
# > If 1 / conversion factor is ~3994, this implies a colour ratio of 1.10. for Saturn with a STIS SrF2 image (see Gustin+ 2012 Table 1):
# > And this in turn means that the counts-per-second to total emitted power (Watts). conversion factor is 9.04e-10 (Gustin+2012 Table 2), for STIS SrF2:
CONST.delrp_jup = 240#km
CONST.kr_per_count = 1/CONST.gustin_factor
CONST.SI_exponents = { # SI prefixes from https://www.nist.gov/pml/owm/metric-si-prefixes. commented out any undesired prefixes.
                    #"q":-30,          # quecto
                    #"r":-27,          # ronto
                    #"y":-24,          # yocto
                    #"z":-21,          # zepto
                    #"a":-18,          # atto
                    #"f":-15,          # femto
                    "p":-12,          # pico
                    "n":-9,           # nano
                    "μ":-6,"u":-6,    # micro
                    "m":-3,           # milli
                    #"d":-1,           # deci
                    #"_":0,            # -
                    #"da":1,           # deca
                    #"h":2,            # hecto
                    "k":3,            # kilo
                    "M":6,            # mega
                    "G":9,            # giga
                    "T":12,           # tera
                    #"P":15,           # peta
                    #"E":18,           # exa
                    #"Z":21,           # zetta
                    #"Y":24,           # yotta
                    #"R":27,           # ronna
                    #"Q":30,           # quetta
                      }
CONST.calib_cts2kr = 7.890258788035481e-05 # Calibrated conversion factor for counts to kilorayleighs for the HST STIS SrF2 filter.
# == cts2kr(dist_earth[AU] x CONST.au_to_km)^2 x CONST.gustin_factor / 1e9
######## Power #################################################################
Power = ConfigLike("power subpackage configurations")
Power.WRITETO = "powers.txt"  # > file to write power results to
Power.DISPLAY_PLOTS = False  # > whether to output plots to screen
Power.DISPLAY_MSGS = False  # > whether to output messages to screen
######## PF ####################################################################
#  naming convention:
#    - flag (RETR, CHAIN, MORPH, KSIZE, ...) -> axes label name, configuration parameter
#    - label (EXTERNAL, LIST, SIMPLE, ...) -> value name, configuration option
#    - value -> the value of the configuration option, at [0] in _cvtrans[flag][label]
#    - index -> the index of the configuration option, usually the same as value, otherwise the index of the value in the translation list, trans
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
PF = ConfigLike("pathfinder configurations")
PF.MAXPOINTSPERCONTOUR = 2000
PF.MAXCONTOURDRAWS = 125
PF.MAXLEGENDITEMS = 50
PF.TWOCOLS = True
PF._cvtrans = {
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
PF._legendkws = {"loc": "lower left", "fontsize": 8,"labelcolor": "linecolor","frameon": False,"mode": "expand","ncol": 3,}  # > active contour list styling
PF.cmap_cycler = [cmr.neutral,cmr.neutral_r,cmr.toxic,cmr.nuclear,cmr.emerald,cmr.lavender,cmr.dusk,cmr.torch,cmr.eclipse,]
# App Icon and Font Path
PF._iconpath = GHROOT/"python/jarvis/resources/aa_asC_icon.ico"
PF._fontpath = GHROOT/"python/jarvis/resources/FiraCodeNerdFont-Regular.ttf"
PF._bgs = {"main": "#000", "sidebar": "#fff", "legend": "#000"}  # > Background colors (hierarchical) based on axes labels
PF.PALLETE_SELECT = "RGB"  #'DEFAULT' #> Default color pallete choice
PF.COLORPALLETTE = {
    "RGB": ["#FF0000FF", "#00FF00FF", "#FF0000FF", "#0000FFFF"],
    "DEFAULT": ["#FF0000FF", "#FF5500FF", "#FF0000FF", "#FFFF0055"],
    "BLUE": ["#0077FFFF", "#7700FFFF", "#0000FFFF", "#00FFFF55"],
    "GREEN": ["#00FF00FF", "#00FF55FF", "#00FF00FF", "#55FF0055"],
    "BW": ["#000", "#FFF", "#000000FF", "#FFFFFF55"]}
PF.HELPKEYS = [k for k in [PF._cvtrans["TOOLTIP"].get(k, [None, None])[1] for k in ["ONSCREEN", "CLI"]] if k is not None]
PF._idpxkws = {"s": 20, "color": PF.COLORPALLETTE[PF.PALLETE_SELECT][0], "zorder": 12, "marker": "x"}  # Identifier Pixel scatter style
PF._clickedkws = {"s": 20, "color": PF.COLORPALLETTE[PF.PALLETE_SELECT][1], "zorder": 12, "marker": "x"}  # Clicked Pixel scatter style
PF._selectclinekws = {"s": 0.3, "color": PF.COLORPALLETTE[PF.PALLETE_SELECT][2], "zorder": 10}  # Selected Contour line style
PF._selectctextkws = {"fontsize": 8, "color": PF._selectclinekws["color"]}  # Selected Contour text style
PF._otherclinekws = {"s": 0.2, "color": PF.COLORPALLETTE[PF.PALLETE_SELECT][3], "zorder": 10}  # Other Contour line style
PF._otherctextkws = {"fontsize": 8, "color": PF._otherclinekws["color"]}  # Other Contour text style
PF._defclinekws = {
    "s": PF._selectclinekws["s"],
    "color": PF._selectclinekws["color"],
    "zorder": PF._selectclinekws["zorder"],
}  # Default Contour line style
PF._defctextkws = {"fontsize": 8, "color": PF._defclinekws["color"]}  # Default Contour text style
PF._handles = {  # > Legend handle styling and static elements (unused)
    "selectedc": {"color": PF._selectclinekws["color"], "lw": 2},
    "otherc": {"color": PF._otherclinekws["color"], "lw": 2},
    "defc": {"color": PF._defclinekws["color"], "lw": 2},
}
PF._infotextkws = {"fontsize": 8,"color": "black", "ha": "center", "va": "center"}
PF._tablekws = {"colWidths": [0.14, 1],"colLabels": None,"rowLabels": None,
                    "colColours": None,"rowColours": None,"colLoc": "center", "rowLoc": "center","loc": "bottom left","bbox": [0.02, 0.02, 0.96, 0.8], "zorder": 0,"cellLoc": "right",}
PF._radioprops = {
                "facecolor": ["#000", PF._clickedkws["color"], PF._clickedkws["color"], PF._idpxkws["color"], PF._idpxkws["color"]],
                "edgecolor": ["#000", PF._clickedkws["color"], PF._clickedkws["color"], PF._idpxkws["color"], PF._idpxkws["color"]],
                "marker": ["o", "o", "X", "o", "X"],
            }
PF._labelprops = {
                "color": ["#000", PF._clickedkws["color"], PF._clickedkws["color"], PF._idpxkws["color"], PF._idpxkws["color"]],
            }
def __validatecfg():
    """Validate the configuration. This is run on initialization of extensions.py."""
    for k, v in PF._cvtrans.items():
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
    for k, v in PF._keybindings.items():
        assert (
            len(v) == 3
        ), f"{k}'s keybinding must be a tuple of length 3 containing the flag, value, and a short description"
    assert exists(PF._iconpath), "Icon file not found"
    assert exists(PF._fontpath), "Font file not found"
PF.validatecfg = __validatecfg
PF.defaults = ConfigLike()
PF.defaults.view_config = {"REGISTER_KEYS": True, "cmap": 0, "mask": 0}
PF.defaults.cv_config ={
                "ACTION": 0,
                "MORPH": [cv2.MORPH_OPEN, cv2.MORPH_CLOSE],
                "CHAIN": cv2.CHAIN_APPROX_SIMPLE,
                "CVH": False,
                "KSIZE": 5,
                "RETR": cv2.RETR_EXTERNAL,
                "FIXEDRANGE": {"ACTIVE":True,"RANGE":[0.0001,0.01], "STEP":0.001}
            }

IMG = ConfigLike("Image configurations")
IMG.contrast = ConfigLike("Func and configurations for the contrast enchancement of images")
IMG.contrast.power = 1.4


log = ConfigLike("Logging")
log.ACTIVE = True
log.logger = logging.getLogger("jarvis")
log.logger.setLevel(logging.DEBUG)
log.logger.addHandler(logging.FileHandler(GHROOT / "jarvis.log", mode="w"))
#log.logger.addHandler(logging.StreamHandler())
log.write = log.logger.debug
log.write("Logging initialized")


plot = ConfigLike("Plotting configurations")
plot.maps = ConfigLike("Default property maps for mpl properties.")
plot.maps.color=ExpandingList(["#060","#0a0","#0f0","#dd0","#fa0","#f50","#f00","#b00","#f08","#b0b","#80f","#00f","#0af"], "#aaa")
plot.maps.marker = ExpandingList(["1","2","3","4","x","+","1","2","3","4","x","+"], ".")
plot.maps.hatch = ExpandingList(["\\\\","//","--","||","oo","xx","**"], "++")
plot.size_a4 = {"width": 8.3, "height": 11.7}
plot.margin = 1
plot.inset_annotate_kws = dict(xy=(0, 1),xycoords="axes fraction",xytext=(+0.5, -0.5),textcoords="offset fontsize",fontsize="medium",verticalalignment="top",weight="bold",bbox={"facecolor": "#0000", "edgecolor": "none", "pad": 3.0},)
plot.meta_annotate_kws = dict(xy=(1, 1),xycoords="axes fraction",xytext=(-0.05, +0.1),textcoords="offset fontsize",fontsize="medium",verticalalignment="bottom",horizontalalignment="right",annotation_clip=False,bbox={"facecolor": "#0000", "edgecolor": "none", "pad": 3.0},)
plot.visit_annotate_kws = dict(xytext=(0, -9), 
                                           textcoords="offset points", ha="center", va="center", 
                                           fontsize="small", 
                                           arrowprops=dict(arrowstyle="-", color="black", 
                                            lw=0.5, shrinkA=0,shrinkB=0), bbox=dict(fc="#fff0", ec="#fff0",lw=0.2, pad=0),clip_on=False,)
plot.gen = ConfigLike("Figure generation configurations")
plot.gen.stacked = ConfigLike("Stacked figure configurations")
plot.defaults = ConfigLike("Default figure configurations")
plot.defaults.rcfile = str(GHROOT/"python/jarvis/resources/jarvis.mplstyle")
plot.defaults.DOY_label = "Day of Year"




plot.gen.overlaid = ConfigLike("Overlaid figure configurations")
plot.gen.mega = ConfigLike("Mega figure configurations")
plot.gen.hist = ConfigLike("Histogram figure configurations")
plot.gen.gif = ConfigLike("GIF figure configurations")






HISAKI = ConfigLike("Mappings for Hisaki/SW dataset")
HISAKI._desc = dict(                 # (Units from HISAKI fits files, rest were guessed)
    jup_sw_pdyn =       ["Pdyn", "$P_{SW,dyn}$", "nPa",],  # Dynamic pressure of the solar wind in nanoPascals.
    # Shared Columns
    RADMON =            ["Rad_Mon", "$R_{rad}$", "counts/min",],  # Radiation monitor for measuring radiation in counts per minute.
    JUPLOC =            ["Jup_Y", "$y_{J}$", "pixel",],  # Y-coordinate position of Jupiter in the image in pixels.
    JPFWHM =            ["Jup_FWHM", "$FWHM_{J}$", "pixel",],  # Full width at half maximum (FWHM) of Jupiter's image in pixels.
    INT_TIME =          ["Intg_Time", "$T_{int}$", "min", ],  # Total time for image integration in minutes.
    SLIT1Y =            ["Slit_B140", "$y_{slit1}$", "pixel", ],  # Y-position of the bottom of slit 1 in the image in pixels.
    SLIT2Y =            ["Slit_B20", "$y_{slit2}$", "pixel",],  # Y-position of the bottom of slit 2 in the image in pixels.
    SLIT3Y =            ["Slit_T20", "$y_{slit3}$", "pixel", ],  # Y-position of the top of slit 3 in the image in pixels.
    SLIT4Y =            ["Slit_T140", "$y_{slit4}$", "pixel", u],  # Y-position of the top of slit 4 in the image in pixels.
    JPFLAG =            ["Jup_Flag", "$JP_{flag}$", "pixel"],  # Flag indicating the position of Jupiter in the image in pixels.
    AURPFLG =           ["Aur_Flag", "$Aurora_{flag}$", "pixel"],  # Flag indicating the position of an aurora in the image in pixels.
    YEAR =              ["YEAR", "$Year$", "years"],  # Year of the observation.
    DAYOFYEAR =         ["DAYOFYEAR", "$DayOfYear$", "days"],  # Day of the year (1 to 365/366).
    SECOFDAY =          ["SECOFDAY", "$t_{sec}$", "sec"],  # Time of day in seconds (from 00:00:00).
    XPOS1 =             ["X_Dawn", "$x_{dawn}$", "pixel"],  # X-position of the dawn (eastern horizon) in the image in pixels.
    XPOS2 =             ["X_Dusk", "$x_{dusk}$", "pixel"],  # X-position of the dusk (western horizon) in the image in pixels.
    XPOS3 =             ["X_Aur", "$x_{aurora}$", "pixel"],  # X-position of the aurora in the image in pixels.
    CML =               ["CML", "$CML$", "degree"],  # Central Meridian Longitude (CML) of Jupiter in degrees.
    DISK =              ["Disk_Size", "$D_{disk}$", "asec"],  # Apparent size of Jupiter's disk in arcseconds.
    Y_POL =             ["Y_Polarization", "$P_{y}$", "0:N 1:S"],  # Polarization state of the observation (0 = North, 1 = South).
    # Torus Columns
    TPOW0710ADAWN =     ["Torus_Power_Dawn", "$F_{torus, dawn}$", "GW"],  # Average flux in the torus region at dawn in gigawatts.
    TERR0710ADAWN =     ["Torus_Power_Dawn_Err", "$F_{err, torus, dawn}$", "GW"],  # Error in the torus flux at dawn in gigawatts.
    CONT0710ADAWN =     ["Torus_Cont_Dawn", "$C_{torus, dawn}$", "counts"],  # Continuum intensity in the torus region at dawn in counts.
    LINT0710ADAWN =     ["Torus_LINT_Dawn", "$L_{torus, dawn}$", "counts/min"],  # LINT (line integral) in the torus region at dawn in counts per minute.
    EFLX0710ADAWN =     ["Torus_Flux_Dawn", "$F_{torus, flux, dawn}$", "eV/cm^2/s"],  # Flux in the torus region at dawn in eV/cm^2/s.
    EERR0710ADAWN =     ["Err_Torus_Flux_Dawn", "$F_{err, torus, flux, dawn}$", "eV/cm^2/s"],  # Error in the torus flux at dawn in eV/cm^2/s.
    PPOS0710ADAWN =     ["Torus_Dawn_Pos", "$P_{torus, dawn}$", "pixel"],  # Position of the torus region at dawn in pixels.
    TPOW0710ADUSK =     ["Torus_Power_Dusk", "$F_{torus, dusk}$", "GW"],  # Average flux in the torus region at dusk in gigawatts.
    TERR0710ADUSK =     ["Torus_Power_Dusk_Err", "$F_{err, torus, dusk}$", "GW"],  # Error in the torus flux at dusk in gigawatts.
    CONT0710ADUSK =     ["Torus_Cont_Dusk", "$C_{torus, dusk}$", "counts"],  # Continuum intensity in the torus region at dusk in counts.
    LINT0710ADUSK =     ["Torus_LINT_Dusk", "$L_{torus, dusk}$", "counts/min"],  # LINT (line integral) in the torus region at dusk in counts per minute.
    EFLX0710ADUSK =     ["Torus_Flux_Dusk", "$F_{torus, flux, dusk}$", "eV/cm^2/s"],  # Flux in the torus region at dusk in eV/cm^2/s.
    EERR0710ADUSK =     ["Torus_Flux_Dusk_Error", "$F_{err, torus, flux, dusk}$", "eV/cm^2/s"],  # Error in the torus flux at dusk in eV/cm^2/s.
    PPOS0710ADUSK =     ["Torus_Dusk_Pos", "$P_{torus, dusk}$", "pixel"],  # Position of the torus region at dusk in pixels.
    # Aurora Columns
    TPOW1190A =         ["Aurora_Power", "$F_{aurora}$", "GW"],#4  # Average auroral flux in gigawatts.
    TERR1190A =         ["Aurora_Power_Err", "$F_{err, aurora}$", "GW"],#5  # Error in the auroral flux in gigawatts.
    CONT1190A =         ["Aurora_Cont", "$C_{aurora}$", "counts"],#6  # Continuum intensity in the aurora region in counts.
    LINT1190A =         ["Aurora_LINT", "$L_{aurora}$", "counts/min"],#7  # LINT (line integral) in the aurora region in counts per minute.
    EFLX1190A =         ["Aurora_Flux", "$F_{aurora, flux}$", "eV/cm^2/s"],#8  # Flux in the aurora region in eV/cm^2/s.
    EERR1190A =         ["Aurora_Flux_Err", "$F_{err, aurora, flux}$", "eV/cm^2/s"],#9  # Error in the auroral flux in eV/cm^2/s.
    PPOS1190A =         ["Aurora_Pos", "$P_{aurora}$", "pixel"],#10  # Position of the aurora region in the image in pixels.
    TPOW1190ARAD1 =     ["Aurora_R1_Power", "$F_{aurora,r1}$", "GW"],  # Average auroral flux in gigawatts.
    TERR1190ARAD1 =     ["Aurora_R1_Power_Err", "$F_{err, aurora,r1}$", "GW"],  # Error in the auroral flux in gigawatts.
    CONT1190ARAD1 =     ["Aurora_R1_Cont", "$C_{aurora,r1}$", "counts"],  # Continuum intensity in the aurora region in counts.
    LINT1190ARAD1 =     ["Aurora_R1_LINT", "$L_{aurora,r1}$", "counts/min"],  # LINT (line integral) in the aurora region in counts per minute.
    EFLX1190ARAD1 =     ["Aurora_R1_Flux", "$F_{aurora, flux,r1}$", "eV/cm^2/s"],  # Flux in the aurora region in eV/cm^2/s.
    EERR1190ARAD1 =     ["Aurora_R1_Flux_Err", "$F_{err, aurora, flux,r1}$", "eV/cm^2/s"],  # Error in the auroral flux in eV/cm^2/s.
    PPOS1190ARAD1 =     ["Aurora_R1_Pos", "$P_{aurora,r1}$", "pixel"],  # Position of the aurora region in the image in pixels.
    TPOW1190ARAD2 =     ["Aurora_R2_Power", "$F_{aurora,r2}$", "GW"],  # Average auroral flux in gigawatts.
    TERR1190ARAD2 =     ["Aurora_R2_Power_Err", "$F_{err, aurora,r2}$", "GW"],  # Error in the auroral flux in gigawatts.
    CONT1190ARAD2 =     ["Aurora_R2_Cont", "$C_{aurora,r2}$", "counts"],  # Continuum intensity in the aurora region in counts.
    LINT1190ARAD2 =     ["Aurora_R2_LINT", "$L_{aurora,r2}$", "counts/min"],  # LINT (line integral) in the aurora region in counts per minute.
    EFLX1190ARAD2 =     ["Aurora_R2_Flux", "$F_{aurora, flux,r2}$", "eV/cm^2/s"],  # Flux in the aurora region in eV/cm^2/s.
    EERR1190ARAD2 =     ["Aurora_R2_Flux_Err", "$F_{err, aurora, flux,r2}$", "eV/cm^2/s"],  # Error in the auroral flux in eV/cm^2/s.
    PPOS1190ARAD2 =     ["Aurora_R2_Pos", "$P_{aurora,r2}$", "pixel"]  # Position of the aurora region in the image in pixels.
)
HISAKI.colnames = {k: v[0] for k, v in HISAKI._desc.items()}
HISAKI.colunits = {k: v[2] for k, v in HISAKI._desc.items()}
HISAKI.tex = {k: v[1] for k, v in HISAKI._desc.items()}
HISAKI.df = ConfigLike("Descriptors mapped to new column names")
HISAKI.df._desc = {v[0]:[k, *v[1:]] for k,v in HISAKI._desc.items()}
HISAKI.df.names = {k: v[0] for k, v in HISAKI.df._desc.items()}
HISAKI.df.units = {k: v[2] for k, v in HISAKI.df._desc.items()}
HISAKI.df.tex = {k: v[1] for k, v in HISAKI.df._desc.items()}
def mapval(inp,output, _desc):
    # input is a string, matchin an item in one of the lists in _desc. output is a string or int or list of combinations of these
    strmap  = {"init":"key", "colname":0,"label":1,"unit":2, "tex":1}
    # turn into indices/"key"
    output = [strmap.get(o, o) for o in (output if isinstance(output, (list, tuple)) else [output])]
    # find the list containing the input, and assign the items and key to found
    for k,v in _desc.items():
        if inp in v:
            found = v +[k]
            break
    # return the requested parts (if numeric, return at index, else return last)
    ret = [found[o] if isinstance(o,int) else found[-1] for o in output ] if isinstance(output, (list,tuple)) else found[output]
    return ret 
        
HISAKI.mapval = lambda x,y: mapval(x,y,HISAKI._desc)

HST = ConfigLike("Mappings for HST dataset")
HST._desc = dict(
    Visit = ["Visit", "v", ""], # Visit number.
    Obs_Date = ["Obs_Date", "", ""], # Observation date. YYYY-MM-DD.
    Obs_Time = ["Obs_Time", "", ""], # Observation time. HH:MM:SS.
    Total_Power = ["Total_Power", "$P_{Tot}$", "GW"], # Total power in Gigawatts.
    Avg_Flux = ["Avg_Flux", "$F_{avg}$", "GW/km^2"], # Average flux in Gigawatts per square kilometer.
    Area = ["Area", "$A$", "km^2"], # Area in square kilometers.
    L_min = ["L_min", "$L_{min}$", ""], # Minimum Luminosity chosen for the calculation.
    L_max = ["L_max", "$L_{max}$", ""], # Maximum Luminosity chosen for the calculation.
    N_pts = ["N_pts", "$N_{path}$", ""], # Number of points in the boundary.
    Date_Created = ["Date_Created", "", ""], # Date the boundary was created. YYYY-MM-DDTHH:MM:SS.
    EXT = ["EXT", "", ""], # Identifier for the boundary. (Default: BOUNDARY)
)
HST.colnames = {k: v[0] for k, v in HST._desc.items()}
HST.colunits = {k: v[2] for k, v in HST._desc.items()}
HST.tex = {k: v[1] for k, v in HST._desc.items()}
HST.df = ConfigLike("Descriptors mapped to new column names")
HST.df._desc = {v[0]:[k, *v[1:]] for k,v in HST._desc.items()}
HST.df.names = {k: v[0] for k, v in HST.df._desc.items()}
HST.df.units = {k: v[2] for k, v in HST.df._desc.items()}
HST.df.tex = {k: v[1] for k, v in HST.df._desc.items()}
HST.mapval = lambda x,y: mapval(x,y,HST._desc)


