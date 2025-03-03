"""jarvis.extensions
- This module contains functions and classes that extend the functionality of the jarvis package.
- The functions and classes in this module are designed to be used in conjunction with the core functionality of the jarvis package.

Functions:
- pathfinder: A GUI tool to select contours from a fits file, for use in defining auroral boundaries.

Classes:
- QuickPlot: Predefined plotting functions for quick visualization of data in power.py [internal]
"""

from multiprocessing import context
import cli
import numpy as np
import matplotlib.pyplot as plt
import time
import cmasher as cmr
import cv2
import os
from astropy.io import fits
from regex import T
from tqdm import tqdm
from .utils import filename_from_path, fpath, hdulinfo
try: # this is to ensure that this file can be imported without needing PyQt6 or PyQt5, eg for QuickPlot
    from PyQt6 import QtGui # type: ignore #
except ImportError:
    try:
        from PyQt5 import QtGui # type: ignore #
    except ImportError:
        tqdm.write('Warning: PyQt6 or PyQt5 not found, pathfinder app may not function correctly')


from matplotlib import font_manager 
from datetime import datetime
from .transforms import fullxy_to_polar_arr
from .utils import fitsheader, get_datetime, assign_params, rpath, split_path

from .polar import prep_polarfits
from .cvis import contourhdu, imagexy, save_contour

import matplotlib as mpl

MAXPOINTSPERCONTOUR = 2000
MAXCONTOURDRAWS = 125
MAXLEGENDITEMS = 50
TWOCOLS = True
#------------------- Tooltips and Info about Config Flags ---------------------#
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

'RETR':{
    'info':'modes to find the contours',
    'kbtemplate': '$flag mode: $label',
        'EXTERNAL':         [cv2.RETR_EXTERNAL,         'a', 'retrieves only the extreme outer contours'], #0
        'LIST':             [cv2.RETR_LIST,             's', 'retrieves all of the contours without establishing any hierarchical relationships'], #1
        'CCOMP':            [cv2.RETR_CCOMP,            'd', 'retrieves all of the contours and organizes them into a two-level hierarchy, where external contours are on the top level and internal contours are on the second level'], #2
        'TREE':             [cv2.RETR_TREE,             'f', 'retrieves all of the contours and reconstructs a full hierarchy of nested contours'] #3
    },
'CHAIN':{
    'info':'methods to approximate the contours',
    'kbtemplate': '$flag mode: $label',
    'trans':(1,2,3,4), #0 is floodfill, but not used here
        'NONE':             [cv2.CHAIN_APPROX_NONE,     'z','stores absolutely all the contour points. That is, any 2 subsequent points (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors, that is, max(abs(x1-x2),abs(y2-y1))==1.'], #1
        'SIMPLE':           [cv2.CHAIN_APPROX_SIMPLE,   'x','compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points.'], #2
        'TC89_L1':          [cv2.CHAIN_APPROX_TC89_L1,  'c','applies one of the flavors of the Teh-Chin chain approximation algorithm.'], #3
        'TC89_KCOS':        [cv2.CHAIN_APPROX_TC89_KCOS,'v','applies one of the flavors of the Teh-Chin chain approximation algorithm.'], #4
    },
'MORPH':{
    'info':'morphological operations to apply to the mask before finding the contours',
    "kbtemplate":"$flag: toggle $label",
        'ERODE':            [cv2.MORPH_ERODE,           'q','Erodes away the boundaries of foreground object'], #0
        'DILATE':           [cv2.MORPH_DILATE,          'w', 'Increases the object area'], #1
        'OPEN':             [cv2.MORPH_OPEN,            'e','Remove small noise'], #2
        'CLOSE':            [cv2.MORPH_CLOSE,           'r', 'Fill small holes'], #3
        'GRADIENT':         [cv2.MORPH_GRADIENT,        't','Difference between dilation and erosion of an image.'], #4
        'TOPHAT':           [cv2.MORPH_TOPHAT,          'y', 'Difference between input image and Opening of the image'],    #5
        'BLACKHAT':         [cv2.MORPH_BLACKHAT,        'u', 'Difference between the closing of the input image and input image'], #6
        'HITMISS':          [cv2.MORPH_HITMISS,         'i', 'Extracts a particular structure from the image'], #7
    },
'KSIZE':{
    'info':'kernel size for the morphological operations. Larger values will smooth out the contours more',
    'kbtemplate': 'Change kernel size: $label',
        "Increase":         [True,                      '=', 'Increase the kernel size'], #0
        "Decrease":         [False,                     '-','Decrease the kernel size'],
    },
'CVH':{
    'info':'whether to use convex hulls of the contours. Reduces the complexity of the contours',
    'kbtemplate': 'Toggle $label',
        'Convex Hull':      [0,                         'f7', 'whether to use convex hulls of the contours. Reduces the complexity of the contours'],
    },
'ACTION':{
    'info':'Select the action when clicking on a point in the image', 
    'kbtemplate': 'Click mode: $label',
    'trans':(0,1,-1,2,-2),
        'None':             [0,                         '0',''],
        'Add Luminosity':   [1,                         '1', 'Add a luminosity sample at the clicked point'],
        'Remove Luminosity':[-1,                        '2', 'Remove a luminosity sample closest to the clicked point'],
        'Add IDPX':         [2,                         '3','Add an ID pixel at the clicked point'],
        'Remove IDPX':      [-2,                        '4','Remove an ID pixel closest to the clicked point'],
    },
'CLOSE':{
    'info':'Close the viewer',
    'kbtemplate': '$info',
        "CLOSE":            [0,                    'escape', 'Close the viewer']
    },
'SAVE':{
    'info':'Save the selected contour, either to the current fits file or to a new file if a path has been provided.',
    'kbtemplate':'$tooltip',
        "SAVE":             [0,                     'enter', 'Save the selected contour']
    },
'RESET':{
    'info':'Reset the viewer pixel selections',
    'kbtemplate': 'Reset selections',
        "RESET":            [0,                       'f10', 'Reset the viewer pixel selections']
    },
'FSCRN':{
    'info':'Toggle fullscreen mode', 
    'kbtemplate': 'Toggle Fullscreen',
        "FULLSCREEN":       [0,                       'f11', 'Toggle Fullscreen']
    } ,
'KILL':{
    'info':'Kill the current process', 
    'kbtemplate': '$tooltip',
        "KILL":             [0,                    'delete','Kill the current process']
    },
'MASK':{
    'info':'Cycle mask display', 
    'kbtemplate': '$tooltip',
        "MASK":             [0,                         'f3','Cycle mask display']
    },
'CMAP':{
    'info':'Cycle colormap', 
    'kbtemplate': '$tooltip',
        "CMAP":             [0,                         'f4','Cycle colormap']
    },
'NOTES':{
    'info':'Toggle note textbox', 
    'kbtemplate': '$tooltip',
        "NOTES":            [0,                        'f1','Toggle note textbox']
    },
'TOOLTIP':{
    'info':'Toggle tooltips', 
    'kbtemplate': '$tooltip',
        "TOOLTIP":          [0,                        'f2','Toggle tooltips'],
        "ONSCREEN":         [1,                        'f12','Toggle onscreen tooltips'],
        "CLI":              [2,                        '`','Print CLI help']

    },
'FIXLRANGE':{'info':'Controls for altering the fixed luminance range, which is the lower and upper limit of valid luminances', 'kbtemplate':'$flag: $tooltip',
             "Lower+":          [0,                      ']','Increase lower limit'],
             "Lower-":          [1,                     '[','Decrease lower limit'],
             "Upper+":          [2,                      '#','Increase upper limit'],
             "Upper-":          [3,                     '\'','Decrease upper limit '],
             "Reset":           [4,                         'f6','Reset range to initial'],
             "Toggle":          [5,                         'f5','Toggle fixed luminance range'],
             "Cycle":           [6,                         'f8','Cycle through steps'],}, 
}      
stepmap = [0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]     
HELPKEYS = [k for k in [_cvtrans['TOOLTIP'].get(k,[None,None])[1] for k in ['ONSCREEN','CLI']] if k is not None]
def __getflaglabels(flag:str):
    '''Returns the ordered list of labels for a given flag'''
    ident= list(_cvtrans[flag].keys())
    ident.remove('info')
    ident.remove('kbtemplate')
    if len(ident) == 1:
        return [ident[0]]
    
    if 'trans' not in ident:
        ret =['' for _ in range(len(ident))]
        for k,v in zip(ident,[_cvtrans[flag][i][0] for i in ident]):
            ret[v] = k
    else:
        ident.remove('trans')
        ret =['' for _ in range(len(ident))]
        for k,v in zip(ident,[_cvtrans[flag][i][0] for i in ident]):
            ret[_cvtrans[flag]['trans'].index(v)]=k
    return ret
def __getflagindex(flag:str,label:str):
    '''Returns the correct index of a given label for a given flag'''
    ident = _cvtrans[flag]
    if ident.get('trans',None) is None:
        return _cvtrans[flag][label][0]
    else:
        v= ident['trans'].index(ident[label][0])
        return v

#--------------------------------- Keybindings --------------------------------#
def __evalkbt(flag,label=None):
    '''Evaluates the keybinding tooltip template for a given flag dictionary'''
    temp = _cvtrans[flag].get('kbtemplate',None)
    if '$label' in temp:
        if label is None:
            label = flag
        temp = temp.replace('$label',label)
    if '$flag' in temp:
        temp = temp.replace('$flag',flag)
    if '$extension' in temp:
        temp = temp.replace('$extension',_cvtrans[flag][label][-1] if len(_cvtrans[flag][label])>3 else "")
    if '$tooltip' in temp:
        temp = temp.replace('$tooltip',_cvtrans[flag][label][2])
    if '$info' in temp:
        temp = temp.replace('$info',_cvtrans[flag]['info'])
    return temp

_keybindings = {} # Structure: {'key':('flag',value,'tooltip')}
for k,v in _cvtrans.items():
    for label in __getflaglabels(k):
        _keybindings[v[label][1]] = (k,v[label][0],__evalkbt(k,label))
                # 'key':('flag',value,'kb_tooltip')

def __gettooltip(flag, value=None, label=None):
    '''Returns the tooltip for a flag (and/or label/value combination), along with the correct keybinding, if available'''
    if label is not None:
        tt = _cvtrans[flag][label][2]
    elif value is not None:
        tt = _cvtrans[flag][__getflaglabels(flag)[value]][2]
    else:
        tt = _cvtrans[flag]['info']
    kb = [k for k,v in _keybindings.items() if v[0] == flag and (label is None or v[1] == label) and (value is None or v[1] == value)] 
    return f"{tt} ({kb[0]})" if kb else tt

    

#----------------------------- Style Definitions ------------------------------#
_bgs ={'main':'#000','sidebar':'#fff','legend':'#000'} #> Background colors (hierarchical) based on axes labels
CSEL = 'RGB'#'DEFAULT' #> Default color pallete choice
COLORPALLETTE = {'RGB':['#FF0000FF','#00FF00FF','#FF0000FF','#0000FFFF'],
                'DEFAULT':['#FF0000FF','#FF5500FF','#FF0000FF','#FFFF0055'],
                'BLUE':['#0077FFFF','#7700FFFF','#0000FFFF','#00FFFF55'],
                'GREEN':['#00FF00FF','#00FF55FF','#00FF00FF','#55FF0055'],
                'BW':['#000','#FFF','#000000FF','#FFFFFF55']}
def __get_bg(ax):
    """Returns the background color for a given axes"""
    return _bgs.get(ax.get_label(), _bgs.get('sidebar', _bgs.get('main', '#fff')))
_legendkws =dict(loc='lower left', fontsize=8, labelcolor='linecolor', frameon=False, mode='expand', ncol=3) #> active contour list styling
_idpxkws = dict(s=20, color=COLORPALLETTE[CSEL][0], zorder=12, marker='x') # Identifier Pixel scatter style 
_clickedkws = dict(s=20, color=COLORPALLETTE[CSEL][1], zorder=12, marker='x') # Clicked Pixel scatter style   
_selectclinekws = dict(s=0.3, color=COLORPALLETTE[CSEL][2], zorder=10) # Selected Contour line style
_selectctextkws = dict(fontsize=8, color=_selectclinekws['color']) # Selected Contour text style
_otherclinekws = dict(s=0.2, color=COLORPALLETTE[CSEL][3], zorder=10) # Other Contour line style
_otherctextkws = dict(fontsize=8, color=_otherclinekws['color']) # Other Contour text style
_defclinekws = dict(s=_selectclinekws['s'], color=_selectclinekws['color'], zorder=_selectclinekws['zorder']) # Default Contour line style
_defctextkws = dict(fontsize=8, color=_defclinekws['color']) # Default Contour text style
_handles =dict( #> Legend handle styling and static elements (unused)
    selectedc = dict(color=_selectclinekws['color'], lw=2),
    otherc = dict(color=_otherclinekws['color'], lw=2),
    defc = dict(color=_defclinekws['color'], lw=2))
cmap_cycler = [cmr.neutral,cmr.neutral_r, cmr.toxic,cmr.nuclear,cmr.emerald,cmr.lavender,cmr.dusk,cmr.torch,cmr.eclipse]
# App Icon and Font Path
_iconpath = fpath('python/jarvis/resources/aa_asC_icon.ico')
_fontpath = fpath('python/jarvis/resources/FiraCodeNerdFont-Regular.ttf')
global FIRST_RUN
FIRST_RUN = True    
#----------------------------- Configuration Validation ------------------------#
def __validatecfg():
    """"""
    for k, v in _cvtrans.items():
        assert 'info' in v, f'{k} must have an info key'
        assert 'kbtemplate' in v, f'{k} must have a kbtemplate key'
        trans = v.get('trans',False)
        if trans:
            assert len(v) - 3 == len(trans), f'{k}\'s index translation length does not match the number of flags: "trans": {trans}, flags: {list(v.keys())}'
        else:
            maxflag = max(val[0] for label, val in v.items() if label not in ['info','kbtemplate'])
            lenflags = len(v) - 3  # zero indexed, and accounting for`info','kbtemplate' and no `trans'
            assert lenflags == maxflag, f'{k}\'s index translation length does not match the number of flags, highest flag value: {maxflag}, flags: {list(v.keys())} (length: {lenflags}+2)'
        for label, val in v.items():
            if label not in ['trans', 'info', 'kbtemplate']:
                assert len(val) >=3, f'{k}\'s index translation values must be a list of length 3 or more, containing the flag value (or index),keybinding, and description'
                assert isinstance(val[0], (int, bool)), f'{k}\'s index translation values must be integers or booleans, not {type(val[0])}: {val[0]}'
                assert isinstance(val[1], str), f'{k}\'s index translation keybindings must be strings, not {type(val[1])}: {val[1]}'
                assert isinstance(val[2], str), f'{k}\'s index translation descriptions must be strings, not {type(val[2])}: {val[2]}'
                if len(val) > 3:
                    assert isinstance(val[3], str), f'{k}\'s index translation extensions must be strings, not {type(val[3])}: {val[3]}'
    for k, v in _keybindings.items():
        assert len(v) == 3, f'{k}\'s keybinding must be a tuple of length 3 containing the flag, value, and a short description'
    assert os.path.exists(_iconpath), 'Icon file not found'
    assert os.path.exists(_fontpath), 'Font file not found'
if FIRST_RUN:
    __validatecfg()


# ------------------------- pathfinder (Function) -----------------------------#
#changelog
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
#FUTURE
# integrate the hierarchy into the gui

def pathfinder(fits_dir: fits.HDUList,saveloc=None,show_tooltips=True, headername='BOUNDARY',morphex=(cv2.MORPH_CLOSE,cv2.MORPH_OPEN),fixlrange=None, fcmode=cv2.RETR_EXTERNAL, fcmethod=cv2.CHAIN_APPROX_SIMPLE, cvh=False, ksize=5,writeresult=True,**persists):
    """### *JAR:VIS* Pathfinder
    > ***Requires PyQt6 (or PyQt5)***

    A GUI tool to select contours from a fits file. The tool allows the user to select luminosity samples from the image, and then to select a contour that encloses all of the selected samples. The user can then save the contour to the fits file, or to a new file if a save location is provided. The user can also change the morphological operations applied to the mask before finding the contours, the method used to find the contours, the method used to approximate the contours, and whether to use the convex hull of the contours. The user can also change the kernel size for the morphological operations.

    #### Parameters:
    - `fits_dir`: The fits file to open.
    - `saveloc`: The location to save the selected contour to. If `None`, the contour will be saved to the original fits file if the user chooses to save it.
    - `show_tooltips`: Whether to show tooltips when hovering over buttons.
    - Initial Config Options:
        - `morphex`: The morphological operations to apply to the mask before finding the contours. Default is `(cv2.MORPH_CLOSE,cv2.MORPH_OPEN)`.
        - `fcmode`: The method to find the contours. Default is `cv2.RETR_EXTERNAL`.
        - `fcmethod`: The method to approximate the contours. Default is `cv2.CHAIN_APPROX_SIMPLE`.
        - `cvh`: Whether to use convex hulls of the contours. Default is `False`.
        - `ksize`: The kernel size for the morphological operations. Default is `5`.

        These may be changed during the session using the GUI buttons.
    """
    mpl.use('QtAgg')
    global G_fits_obj#:fits.HDUList
    G_fits_obj = fits.open(fits_dir, 'update')
    font_manager.fontManager.addfont(fpath('python/jarvis/resources/FiraCodeNerdFont-Regular.ttf'))
    with mpl.rc_context(rc={'font.family': 'FiraCode Nerd Font', 'axes.unicode_minus': False, 'toolbar': 'None', 'font.size': 8,
                            'keymap.fullscreen': '', 'keymap.home': '', 'keymap.back': '', 'keymap.forward': '', 'keymap.pan': '',
                            'keymap.zoom': '', 'keymap.save': '', 'keymap.quit': '', 'keymap.grid': '', 'keymap.yscale': '',
                            'keymap.xscale': '', 'keymap.copy':''}):
        #generate a stripped down, grey scale image of the fits file, and make normalised imagempl.rcParams['toolbar'] = 'None'
        proc =prep_polarfits(assign_params(G_fits_obj, fixed='LON', full=True))
        img = imagexy(proc, cmap=cmr.neutral, ax_background='white', img_background='black') 
        oldlen = len(G_fits_obj)
        global show_mask, G_clicktype, G_path, G_headername, G_show_tooltips, G_morphex, G_fcmode, G_fcmethod, G_cvh, G_ksize, falsecolor, G_fixrange, REGISTER_KEYS,cl
        REGISTER_KEYS = True # control whether to listen for keypresses and run each key's function, set to False when we are in a text box.
        global FIRST_RUN
        if FIRST_RUN:
            _get_pathfinderhead()# tqdm.write('(Press # to print keybindings)')
            G_show_tooltips = show_tooltips #:bool # whether to show tooltips
            G_clicktype =persists.get('G_clicktype', 0) #:int # the type of click event, [0,1,2,3,4]
            G_morphex= [m for m in morphex] #:list[int] # list of activated morphological operations
            show_mask = 0 #:int # mask value, 0 is no mask, 1+ is different visibility levels
            G_fcmode = fcmode #:int # selected contour mode aka cv2.RETR_*
            G_fcmethod = fcmethod #:int # selected contour method aka cv2.CHAIN_APPROX_*
            G_cvh= cvh #:bool # whether to use convex hulls of the contours
            G_ksize = ksize #:int, 2N+1 only # kernel size for morphological operations
            falsecolor = 0 #:int # what colormap to use for the mask, 0 is default normal, 1 is inverted, 2+ are false color maps
            G_headername = 'BOUNDARY' if headername is None else headername #:str # the selected header name to save the contour to, changeable by the user using the text box
            G_fixrange= [i for i in fixlrange] if fixlrange is not None else None #:list[float]|None # the fixed luminance range, if any. selected pixels take priority over this range.
            cl = persists.get('cl', None) # the auto selected contour picker point
            FIRST_RUN = False
        global G_ofixrange#:list[float]|None
        G_ofixrange = G_fixrange
        normed = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        #set up global variables needed for the click event and configuration options
        global imarea#:float
        imarea = (np.pi*(normed.shape[0]/2)**2)/2 
        global clicked_coords#:list[list[float,list[int]]]
        clicked_coords= persists.get('clicked_coords', []) # members are of the form [luminance, [x,y]]
        global id_pixels#:list[list[int]]
        id_pixels = persists.get('id_pixels', []) # members are of the form [x,y]
        G_path = persists.get('G_path', None)
        global retattrs#:dict
        retattrs = dict(LMIN =-np.inf, LMAX=np.inf, NUMPTS=np.nan, XYA_CT=np.nan, XYA_CTP=np.nan, XYA_IMG=imarea)
        
        def __approx_contour_area_pct(contourarea):
            num = contourarea/imarea * 100
            if num < 0.01:
                return f'{num:.2e}%'
            return f'{num:.2f}%'    
        # set up the figure and axes
        
        fig = plt.figure(figsize = (12.8,7.2))
        qapp = fig.canvas.manager.window
        qapp.setWindowTitle(f'JAR:VIS Pathfinder ({fits_dir})')
        iconpath = fpath('python/jarvis/resources/aa_asC_icon.ico')
        qapp.setWindowIcon(QtGui.QIcon(iconpath))
        # gs: 1 row, 2 columns, splitting the figure into plot and options
        gs = fig.add_gridspec(1,2, wspace=0.02, hspace=0, width_ratios=[4,1.5], left=0, right=1, top=1, bottom=0)
        # mgs: 17 rows, 12 columns, to layout the options
        mgs = gs[1].subgridspec(12,12, wspace=0, hspace=0, height_ratios=[0.2,0.2,0.15, 0.2,0.2,0.5, 0.5,0.15, 0.05,1,1.5,1.5]) 
        # layout the axes in the gridspecs                  
        
        linfax = fig.add_subplot(mgs[0:2,:8], label='infotext') # text box for information, top right
        clax = fig.add_subplot(mgs[0,8:], label='CLOSE') # close button
        sbax = fig.add_subplot(mgs[1,8:], label='SAVE') # save button
        headernameax = fig.add_subplot(mgs[2,8:], label='headername') # text box for header name
        rbax = fig.add_subplot(mgs[3,8:], label='RESET') # reset button
        fsax = fig.add_subplot(mgs[4,8:], label='FSCRN') # fullscreen button
        flax = fig.add_subplot(mgs[2:5,:8], label='ACTION', facecolor=_bgs['sidebar']) # click mode selection
        retrax = fig.add_subplot(mgs[5,:8], label='RETR', facecolor=_bgs['sidebar']) # contour retrieval mode selection
        morphax = fig.add_subplot(mgs[5:7,8:], label='MORPH', facecolor=_bgs['sidebar']) # morphological operations selection
        chainax = fig.add_subplot(mgs[6,:8], label='CHAIN', facecolor=_bgs['sidebar']) # contour approximation method selection
        ksizeax = fig.add_subplot(mgs[8,:8], label='KSIZE', facecolor=_bgs['sidebar']) # kernel size selection
        cvhax = fig.add_subplot(mgs[7:9,8:], label='CHAIN', facecolor=_bgs['sidebar']) # convex hull selection
        ksizesubax = fig.add_subplot(mgs[7,:8], label='KSIZE', facecolor=_bgs['sidebar']) # kernel size display
        fitinfoax = fig.add_subplot(mgs[9,:], label='fitinfo', facecolor=_bgs['sidebar']) # fit information display
        lax = fig.add_subplot(mgs[-1,:], label='legend', facecolor=_bgs['legend']) # legend
        ax = fig.add_subplot(gs[0],label='main', zorder=11) # main plot

        mmgs = mgs[10,:].subgridspec(2,1, wspace=0, hspace=0, height_ratios=[10,1])  # note textbox
        noteax = fig.add_subplot(mmgs[0], label='notes',zorder=-10,facecolor=_bgs['legend'])#fig.add_axes(ax.bbox.bounds, label='note', facecolor='#fff5', zorder=10)
        global notebox,notemode_active, G_notes
        notemode_active = False
        G_notes = ''
        notebox = noteax.text(0,0.5, G_notes, fontsize=8, color='black', ha='left', va='center', wrap=True, bbox=dict(lw=0), zorder=200)
        noteax.set_axis_off()
        #noteax.set(zorder=-10)#position=[0,0,0,0],
        if show_tooltips:
            tooltipax = fig.add_subplot(mgs[10,:], label='tooltip', facecolor='#000')
            tooltip = tooltipax.text(0.012, 0.98, '', fontsize=6, color='black', ha='left', va='top', wrap=True, bbox=dict(facecolor='white', alpha=0.5, lw=0), zorder=10)    
        axs = [ax, lax, rbax, linfax,  retrax, morphax, chainax, flax, sbax, cvhax, ksizeax, fitinfoax, clax, ksizesubax, fsax, noteax]+([tooltipax] if show_tooltips else [])
        axgroups ={'buttons':[rbax, sbax, clax, fsax],'textboxes':[linfax, fitinfoax],'options':[retrax, morphax, chainax, cvhax, ksizeax, ksizesubax, flax],'legends':[lax]}
        global lastax
        lastax = None
        
        linfax.text(0.5, 0.5, 'Select at least 2 or more\nLuminosity Samples', fontsize=8, color='black', ha='center', va='center')    
        #final plot initialisation
        ksizeax.set_facecolor('#fff')
        for a in axs:
            a.xaxis.set_visible(False)
            a.yaxis.set_visible(False)
        #fig.canvas.setWindowTitle('Luminance Viewer')
        fig.set_facecolor('black')
        ax.imshow(normed, cmap=cmap_cycler[falsecolor], zorder=0)
        def __redraw_displayed_attrs():
            global G_fits_obj
            visit, tel, inst, filt, pln, hem,cml,doy  = *fitsheader(G_fits_obj, 'VISIT', 'TELESCOP', 'INSTRUME', 'FILTER', 'PLANET', 'HEMISPH', 'CML', 'DOY'), 
            dt = get_datetime(G_fits_obj)
            fitinfoax.text(0.04, 0.96, "FITS File Information" , fontsize=10, color='black', ha='left', va='top')
            pth = str(fits_dir)
            pthparts = []
            if len(pth) > 30:
                parts = split_path(rpath(pth))
                while len(parts) > 0:
                    stick = parts.pop(-1)
                    if len(parts)>0:
                        if len(stick)+len(parts[-1]) < 30:
                            parts[-1] += stick
                        else:
                            pthparts.append(stick)
                    else:
                        pthparts.append(stick)
            else:
                pthparts.append(pth)
            pthparts.reverse()
            hdulinf = hdulinfo(G_fits_obj,include_ver=True)
            def hdupart(hdul):
                template = "{i} {name} ({type})"
                lent = len(hdul)
                hduc = [template.format(i=i, name=h['name'], type=h['type']) for i,h in enumerate(hdul)]
                #cols = lent//4 + (1 if lent%4 > 0 else 0)
                rows = 4 if lent >= 7 else 3 if lent>=3 else lent
                maxwidth = max([len(h) for h in hduc])
                hduc = [h.ljust(maxwidth) for h in hduc]
                addempties = rows - (lent%rows)
                if addempties < rows:
                    hduc += [' '.ljust(maxwidth) for _ in range(addempties)]
                
                # split into 4 parts
                hduc = [hduc[i::rows] for i in range(rows)]
                hduc = [' '.join(hduc[i]) for i in range(rows)]
                ret = [['HDUs', hduc[0]]]+[[' ', hduc[i]] for i in range(1,rows)]
                return ret

            hdup = hdupart(hdulinf)
            ctxt = [['File',f'{pthparts[0]}'], *[[' ', f"{p}"] for p in pthparts[1:]],
                                    ['Inst.',f'{inst} ({tel}, {filt} filter)'],
                                    ['Obs.',f'visit {visit} of {pln} ({hem})'],
                                    ['Date',f'{dt.strftime("%d/%m/%Y")}, day {doy}'],
                                    ['Time',f'{dt.strftime("%H:%M:%S")}(ICRS)'],
                                    ['CML',f'{cml:.4f}°'],*hdup]
                                    # ['HDUs',f"{0} {hdulinf[0]['name']}({hdulinf[0]['type']})"], *[[" ", f"{i} {h['name']}({h['type']})"] for i,h in enumerate(hdulinf[1:],1)]]
            tl=fitinfoax.table(cellText=ctxt,cellLoc='right', cellColours=[[_bgs['sidebar'], _bgs['sidebar']] for i in range(len(ctxt))], colWidths=[0.14,1], colLabels=None, rowLabels=None, colColours=None, rowColours=None, colLoc='center', rowLoc='center', loc='bottom left', bbox=[0.02, 0.02, 0.96, 0.8], zorder=0)
            tl.visible_edges = ''
            tl.auto_set_font_size(False)
            tl.set_fontsize(8)
            for key, cell in tl.get_celld().items():
                cell.set_linewidth(0)
                cell.PAD =0
                if key[0]<len(pthparts) and key[1] == 1:
                    cell.set_fontsize(max(8-len(pthparts)+1,5))
                if key[0]>=len(pthparts)+5 and key[1] == 1:
                    cell.set_fontsize(max(8-len(hdulinf)+1,5))
            fig.canvas.blit(fitinfoax.bbox)
        __redraw_displayed_attrs()
        def __generate_conf_output():
            """Returns the current configuration options in a dictionary, encoding the values as integers or binary strings to save to the fits file."""
            global G_morphex, G_fcmode, G_fcmethod, G_cvh, G_ksize
            m_ = [0,]*8
            for m in G_morphex:
                m_[m]=1
            m_ = "".join([str(i) for i in m_])
            return dict(MORPH=m_, RETR=G_fcmode, CHAIN=G_fcmethod, CVH=int(G_cvh), KSIZE=G_ksize, CONTTIME=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        def __generate_coord_output():
            """Returns the current selected coordinates to save to the fits file."""
            global clicked_coords, id_pixels,cl
            ret = dict()
            for i, coord in enumerate(clicked_coords):
                ret[f'LUMXY_{i}'] = f"{coord[0]:.3f},({coord[1][0]},{coord[1][1]})"
            for i, coord in enumerate(id_pixels):
                ret[f'IDXY_{i}'] = f"({coord[0]},{coord[1]})"
            if  cl is not None:
                ret['AUTOCL'] = f"({cl[0]},{cl[1]})"
            
            return ret
        #main update function
        def __redraw_main(_):
            global id_pixels, G_clicked_coords, G_morphex, G_fcmode, G_fcmethod, G_cvh, G_ksize, G_path, retattrs,  show_mask, falsecolor, G_fixrange,cl
            #debug print(cl)
            for a in [ax, lax]:
                a.clear()
                a.set_facecolor(__get_bg(a))
            ax.imshow(normed, cmap=cmap_cycler[falsecolor], zorder=0)
            if len (id_pixels) > 0:
                pxsc = [idpixels[0] for idpixels in id_pixels], [idpixels[1] for idpixels in id_pixels]
                ax.scatter(*pxsc, **_idpxkws)
            if len(clicked_coords) > 0:
                scpc = [c[1][0] for c in clicked_coords], [c[1][1] for c in clicked_coords]
                ax.scatter(*scpc, **_clickedkws)
            if len(clicked_coords) >1 or G_fixrange is not None:
                lrange = G_fixrange[0:2] if G_fixrange is not None else (0,1)
                if len(clicked_coords) >1:
                    lrange = [min(*[c[0] for c in clicked_coords], *lrange), max(*[c[0] for c in clicked_coords], *lrange)]

                mask = cv2.inRange(normed, lrange[0]*255, lrange[1]*255)
                retattrs.update({'LMIN':lrange[0], 'LMAX':lrange[1]})
                retattrs.update({"LMETHOD":'FIXEDL' if lrange==G_fixrange else 'IDPX'})
                # smooth out the mask to remove noise
                kernel = np.ones((G_ksize, G_ksize), np.uint8)  # Small kernel to smooth edges
                for morph in G_morphex:
                    mask = cv2.morphologyEx(mask, morph, kernel)
                if show_mask !=0:
                    a = 0.5 if show_mask in [1,3,5] else 1
                    cmap = cmr.neutral if show_mask in [1,2] else cmr.neutral_r if show_mask in [3,4] else cmr.neon
                    ax.imshow(mask, cmap=cmap, alpha=a, zorder=0.1)
                # find the contours of the mask
                contours, hierarchy = cv2.findContours(image=mask, mode=G_fcmode, method=G_fcmethod)
                if G_cvh: # Convex hull of the contours which reduces the complexity of the contours
                    contours = [cv2.convexHull(cnt) for cnt in contours]
                sortedcs = sorted(contours, key=lambda x: cv2.contourArea(x))
                sortedcs.reverse()
                linfax.clear()
                txt = ""
                if G_fixrange is not None:
                    txt += "FIXED"
                    if len(G_fixrange)> 2:

                        txt += f"({G_fixrange[2]:.0e})"
                linfax.text(0.5,0.75, f'LRG=({lrange[0]:.4f},{lrange[1]:.4f})'+txt, fontsize=8, color='black', ha='center', va='center')
                #------- Contour Selection -------#
                if len(id_pixels) > 0 or  isinstance(cl,(list,tuple)): # if an id_pixel is selected, find the contour that encloses it
                    selected_contours = []
                    other_contours = []
                    if len(id_pixels) > 0: #------------ IDPX METHOD ------------#
                        for i,contour in enumerate(sortedcs):
                            if all([cv2.pointPolygonTest(contour, id_pixel, False) > 0 for id_pixel in id_pixels]):  # >0 means inside
                                selected_contours.append([i,contour])
                            else:
                                other_contours.append([i,contour])
                        #if a contour is found, convert the contour points to polar coordinates and return them
                    else: #------------ CLICK0 METHOD ------------#
                       
                        co = [(i,abs(cv2.pointPolygonTest(contour, cl, True))) for i,contour in enumerate(sortedcs)] + [(None,np.inf)]
                        print(co,cl)
                        chosen = min(co, key=lambda x: x[1])
                           
                        print(chosen)
                        if chosen[0] is not None:
                            selected_contours.append([chosen[0],sortedcs[chosen[0]]])
                        other_contours = [[i,contour] for i,contour in enumerate(sortedcs) if i != chosen[0]]
                    #------- Contour Drawing (with selections) -------#
                    for (i, contour) in selected_contours:
                        ax.scatter(*contour.T,**_selectclinekws)
                        cpath = mpl.path.Path(contour.reshape(-1, 2))
                        ax.plot(*cpath.vertices.T, color=_selectclinekws['color'], lw=_selectclinekws['s'])
                        ax.text(contour[0][0][0], contour[0][0][1], f"{i}", **_selectctextkws)
                    for (i, contour) in other_contours:
                        if i <MAXCONTOURDRAWS and len(contour)>2 and len(contour)<MAXPOINTSPERCONTOUR:
                            ax.scatter(*contour.T,**_otherclinekws)
                            cpath = mpl.path.Path(contour.reshape(-1, 2))
                            ax.plot(*cpath.vertices.T, color=_otherclinekws['color'], lw=_otherclinekws['s'])
                            ax.text(contour[0][0][0], contour[0][0][1], f"{i}", **_otherctextkws)
                    linfax.text(0.5, 0.5, f'Contours: {len(selected_contours)}({len(other_contours)} invalid)', fontsize=8, color='black', ha='center', va='center')
                    if len(selected_contours) >0:
                        G_path = selected_contours[0][1]
                        retattrs.update({'NUMPTS':len(G_path), 'XYA_CT':cv2.contourArea(G_path), 'XYA_CTP':__approx_contour_area_pct(cv2.contourArea(G_path))})
                        linfax.text(0.5, 0.25, f'Area: {retattrs["XYA_CTP"]}, N:{retattrs['NUMPTS']}', fontsize=10, color='black', ha='center', va='center')

                    selectedhandles = [mpl.lines.Line2D([0], [0], label=f'{i}, {__approx_contour_area_pct(cv2.contourArea(sortedcs[i]))}', **_handles['selectedc']) for i in [c[0] for c in selected_contours]]
                    otherhandles = [mpl.lines.Line2D([0], [0], label=f'{i}, {__approx_contour_area_pct(cv2.contourArea(sortedcs[i]))}', **_handles['otherc']) for i in [c[0] for c in other_contours]]
                    lax.legend(handles=[*selectedhandles,*otherhandles][:min(MAXLEGENDITEMS,len(otherhandles)-1)], **_legendkws)
                else: #------- Contour Drawing (without selections) -------#
                    for i,contour in enumerate(sortedcs):
                            if i <MAXCONTOURDRAWS and len(contour)>2 and len(contour)<MAXPOINTSPERCONTOUR:
                                ax.scatter(*contour.T,**_defclinekws)
                                cpath = mpl.path.Path(contour.reshape(-1, 2))
                                ax.plot(*cpath.vertices.T, color=_defclinekws['color'], lw=_defclinekws['s'])
                                ax.text(contour[0][0][0], contour[0][0][1], f"{i}", **_defctextkws)
                    linfax.text(0.5, 0.5, f'Contours: {len(sortedcs)}', fontsize=8, color='black', ha='center', va='center')
                    handles = [mpl.lines.Line2D([0], [0], label=f'{i}, {__approx_contour_area_pct(cv2.contourArea(sortedcs[i]))}',**_handles['defc'] ) for i in range(len(sortedcs))][:min(MAXLEGENDITEMS,len(sortedcs)-1)]
                    lax.legend(handles=handles, **_legendkws)
                    lax.set_facecolor(__get_bg(lax))
            else: #------- No Contours -------#
                linfax.clear()
                linfax.text(0.5, 0.5, 'Select at least 2 or more\nLuminosity Samples', fontsize=8, color='black', ha='center', va='center')
                lax.set_facecolor(__get_bg(ax))
            fig.canvas.draw()
            fig.canvas.flush_events()
        #config gui elements
        #---- RESET button ----#
        breset = mpl.widgets.Button(rbax, "Reset      "+u'\uF0E2')
        def __event_reset(event):
            global clicked_coords, id_pixels,falsecolor,cl,show_mask
            clicked_coords = []
            id_pixels = []
            falsecolor = 0
            cl = None
            show_mask = 0

            __redraw_main(None)
        breset.on_clicked(__event_reset)
        #---- SAVE button ----#
        bsave = mpl.widgets.Button(sbax, "Save       "+u'\uEB4B')
        def __event_save(event):
            global G_path, G_fits_obj,retattrs,G_headername
            if G_path is not None:
                pth = G_path.reshape(-1, 2)
                pth = fullxy_to_polar_arr(pth, normed, 40)
                if saveloc is not None:
                    raise NotImplementedError("Saving to a new fits file is not yet implemented") #todo
                    n_fits_obj = save_contour(fits_obj=G_fits_obj, cont=pth)
                    n_fits_obj.writeto(saveloc, overwrite=True)
                    tqdm.write(f"Saved to {saveloc}, restarting viewer with new data")
                    linfax.clear()
                    linfax.text(0.5, 0.5, "Saved to "+str(saveloc)+"\n restarting viewer with new data", fontsize=8, color='black', ha='center', va='center')
                    fig.canvas.blit(linfax.bbox)
                    #wait for 1 second
                    time.sleep(1)
                    plt.close()
                    G_fits_obj.close()
                    G_fits_obj = n_fits_obj
                    pathfinder(saveloc, None, G_morphex, G_fcmode, G_fcmethod
                                    , G_cvh, G_ksize,persists={'clicked_coords':clicked_coords, 'id_pixels':id_pixels, 'G_clicktype':G_clicktype, 'G_path':G_path})
                else:
                    
                    gcopy = G_fits_obj.copy()
                    nhattr = retattrs
                    nhattr |=dict()
                    nhattr |= __generate_conf_output()
                    nhattr |= __generate_coord_output()
                    nhattr |= {m:fitsheader(G_fits_obj,m) for m in ['UDATE','YEAR','VISIT','DOY']}
                    if G_notes not in ['', None]:
                        notes = G_notes.replace('\n', " "*32)
                        notes= "".join([char if ord(char) < 128 else f'\\x{ord(char):02x}' for char in notes])
                        nhattr['NOTES'] = notes
                    for k,v in nhattr.items():
                        if v in [np.nan, np.inf, -np.inf]:
                            raise KeyError(f'Key Mismatch: {v} is not a valid number, please check the value of {k}')
                    header = fits.Header(nhattr)
                    extname = f'{G_headername.upper()}'
                    if extname in [hdu.name for hdu in G_fits_obj]:
                        ver = max([hdu.ver for hdu in G_fits_obj if hdu.name == extname])+1
                        ch = contourhdu(pth, name=extname, header=header, ver=ver)
                    else:
                        ch = contourhdu(pth, name=extname, header=header)
                    G_fits_obj.append(ch)
                    #repr(G_fits_obj[-1])
                    G_fits_obj.flush()
                    #G_fits_obj.writeto("temp/temp.fits", overwrite=True, output_verify='fix')

                    #G_fits_obj.close()
                    #k_fits_obj = fits.open("temp/temp.fits")
                    #k_fits_obj.writeto(fits_dir, overwrite=True, output_verify='fix')
                    assert len(G_fits_obj) == oldlen+1, f"Save Failed: {len(G_fits_obj)} != {oldlen+1}\n"
                    #fits.diff(gcopy, k_fits_obj)
                    tqdm.write(f"Save Successful, contour added to fits file at index {len(G_fits_obj)-1}")
                    linfax.clear()
                    linfax.text(0.5, 0.5, "Save Successful, contour added\n to fits file at index"+str(len(G_fits_obj)-1), fontsize=8, color='black', ha='center', va='center')
                    fig.canvas.blit(linfax.bbox)
                    __redraw_displayed_attrs()
            else:
                tqdm.write("Save Failed: No contour selected to save")
        bsave.on_clicked(__event_save)
        #---- RETR options ----#
        retrbts = mpl.widgets.RadioButtons(retrax,__getflaglabels('RETR'), active=0)
        def __event_retr(label):
            """set the global variable G_fcmode to the correct value based on the label"""
            global G_fcmode
            G_fcmode = _cvtrans['RETR'][label][0]
        retrbts.on_clicked(__event_retr)
        #---- MORPH options ----#
        bopts = __getflaglabels('MORPH')
        acts = [True if _cvtrans['MORPH'][b][0] in G_morphex else False for b in bopts]
        morphbts = mpl.widgets.CheckButtons(morphax, bopts, acts)
        def __event_morphex(lb):
            """toggle the morphological operations in G_morphex based on the label, add if not present, remove if present"""
            global G_morphex
            if _cvtrans['MORPH'][lb][0] in G_morphex:
                G_morphex.remove(_cvtrans['MORPH'][lb][0])
            else:
                G_morphex.append(_cvtrans['MORPH'][lb][0])
        morphbts.on_clicked(__event_morphex)
        #---- CHAIN options ----#
        chainbts = mpl.widgets.RadioButtons(chainax,__getflaglabels('CHAIN'), active=1)
        def __event_chain(label):
            """set the global variable G_fcmethod to the correct value based on the label"""
            global G_fcmethod
            G_fcmethod = _cvtrans['CHAIN'][label][0]
        chainbts.on_clicked(__event_chain)
        #---- CLOSE button ----#
        bclose = mpl.widgets.Button(clax, "Close      "+u'\uE20D', color='#ffaaaa', hovercolor='#ff6666')
        #bclose.label.set(fontsize=14)
        def __event_close(event):
            """close the figure"""
            plt.close()
        bclose.on_clicked(__event_close)
        #---- CVH options ----#
        cvhbts = mpl.widgets.CheckButtons(cvhax, ['Convex Hull'], [G_cvh],)
        def __event_cvh(label):
            """toggle the G_cvh variable"""
            global G_cvh
            G_cvh = not G_cvh
        cvhbts.on_clicked(__event_cvh)
        #---- KSIZE options ----#
        ksizeslider = mpl.widgets.Slider(ksizeax,"", valmin=1, valmax=11, valinit=G_ksize, valstep=2, color='orange', initcolor='None')
        ksizeax.add_patch(mpl.patches.Rectangle((0,0), 1, 1, facecolor='#fff', lw=1, zorder=-1, transform=ksizeax.transAxes, edgecolor='black'))
        ksizeax.add_patch(mpl.patches.Rectangle((0,0.995), 5/12, 1.005, facecolor='#fff', lw=0, zorder=10, transform=ksizeax.transAxes))
        def __event_ksize(val):
            """set the global variable G_ksize to the slider value, and update the displayed value"""
            global G_ksize
            global clicked_coords
            ksizeax.set_facecolor('#fff')
            G_ksize = int(val)
            ksizesubax.clear()
            ksizesubax.text(0.1, 0.45, f'ksize: {G_ksize}', fontsize=8, color='black', ha='left', va='center')
            fig.canvas.blit(ksizesubax.bbox)
            __redraw_main(None)
        __event_ksize(G_ksize)
        ksizeslider.on_changed(__event_ksize)
        #---- SELECTMODE options ----#
        _radioprops={'facecolor':['#000', _clickedkws['color'], _clickedkws['color'], _idpxkws['color'], _idpxkws['color']], 
                    'edgecolor':['#000', _clickedkws['color'], _clickedkws['color'], _idpxkws['color'], _idpxkws['color']],
                    'marker': ['o','o','X','o','X']}
        _labelprops = {'color': ['#000', _clickedkws['color'], _clickedkws['color'], _idpxkws['color'], _idpxkws['color']]}
        selradio =  mpl.widgets.RadioButtons(flax, ['None', 'Add Luminosity', 'Remove Luminosity', 'Add IDPX', 'Remove IDPX'], active=0, radio_props=_radioprops, label_props=_labelprops) 
        def __event_select(label):
            """set the global variable G_clicktype to the correct value based on the label"""
            global G_clicktype
            global clicked_coords
            ret = 0
            if 'Luminosity' in label:
                ret = 1
            elif 'IDPX' in label:
                ret = 2
            if 'Remove' in label:
                ret = -ret
            G_clicktype = ret
        selradio.on_clicked(__event_select)
        #---- FULLSCREEN button ----#
        bfullscreen = mpl.widgets.Button(fsax, 'Fullscreen '+u'\uF50C')
        #bfullscreen.label.set(fontsize=14)  
        def __event_fullscreen(event):
            fig.canvas.manager.full_screen_toggle()
        bfullscreen.on_clicked(__event_fullscreen)

        #---- CLICK options ----#
        def __on_click_event(event):
            global clicked_coords, id_pixels, G_clicktype, REGISTER_KEYS,cl
            if event.inaxes == headernameax:
                REGISTER_KEYS = False
            else:
                REGISTER_KEYS = True
            if event.inaxes == ax:
                if event.xdata is not None and event.ydata is not None:
                    click_coords = (int(event.xdata), int(event.ydata))
                    # get the luminance value of the clicked pixel
                    if G_clicktype == 1: # add luminosity
                        clicked_coords.append([img[int(event.ydata), int(event.xdata)]/255, click_coords])
                    elif G_clicktype == -1: # remove luminosity
                        if len(clicked_coords) > 0:
                            clicked_coords.pop(np.argmin([np.linalg.norm(np.array(c[1])-np.array(click_coords)) for c in clicked_coords]))
                    elif G_clicktype == 2: # add id pixel
                        id_pixels.append(click_coords)
                    elif G_clicktype == -2: # remove id pixel
                        if len(id_pixels) > 0:
                            id_pixels.pop(np.argmin([np.linalg.norm(np.array(c)-np.array(click_coords)) for c in id_pixels]))   
                    elif G_clicktype == 0 and event.button == 3: #if not in any mode, right click to auto select the closest contour
                        cl = click_coords
            __redraw_main(cl)
        click_event = fig.canvas.mpl_connect('button_press_event', __on_click_event) #noqa: F841
        #---- HOVER options ----#
        def __on_mouse_move_event(event):
            global lastax
            inax = event.inaxes
            global G_show_tooltips    
            if G_show_tooltips:
                if inax in axgroups['options']+axgroups['buttons']:
                    if True:#lastax != inax:
                        lastax = inax
                        axlabel = inax.get_label()
                        txts = [t for t in  inax.get_children() if isinstance(t, mpl.text.Text)]
                        txts = [t for t in txts if t.get_text() != '']
                        # now find if the mouse is over any of the texts
                        txt = None
                        for t in txts:
                            if t.contains(event)[0]:
                                txt = t
                                break
                        if txt is not None:
                            item = txt.get_text()
                            try: 
                                texti = [f'{axlabel}.{item}', _cvtrans[axlabel][item][1]] 
                            except KeyError:
                                texti = [f'{axlabel}', _cvtrans[axlabel]['info']]
                            
                        else:
                            texti = [axlabel,_cvtrans[axlabel]['info']]
                        tooltip.set_text(f'{texti[0]}: {texti[1]}')                    
                        fig.canvas.draw_idle()
                        fig.canvas.blit(tooltipax.bbox)
            if inax == ax:
                fig.canvas.set_cursor(3)
            elif inax in axgroups['options']+axgroups['buttons']:
                fig.canvas.set_cursor(2)
            else:
                fig.canvas.set_cursor(1)
        hover_event = fig.canvas.mpl_connect('motion_notify_event', __on_mouse_move_event) #noqa: F841
        #----TEXTENTRY----#
        textbox = mpl.widgets.TextBox(headernameax, 'Header Name:', initial=G_headername)
        delchars = ''.join(c for c in map(chr, range(1114111)) if not c.isalnum() and c not in [' ','_','-'])
        delmap = str.maketrans("", "", delchars)
        def __on_headername_text_change(text:str):
            global G_headername
            text = text.translate(delmap)
            G_headername = text.replace(" ","_").replace()
        textbox.on_submit(__on_headername_text_change)
        #----NOTES----#
        
        #----KEYEVENTS----#
        global modifiers
        modifiers = bytes((0 for i in range(4))) #caps shift ctrl alt
        kmap = dict(enter='\n',space=' ', tab='\t')
                    
        __modind = tuple(['caps','shift','ctrl','alt'])
        def __modind(key:str)->int:
            return __modind.index(key) #shouldnt need this, flags are fixed so we can just remember the order, is less readable though
        def __getmod(ind:int)->bool:
            global modifiers
            return modifiers & (1<<ind.key)
        def __setmod(ind:int, val:bool)->None:
            global modifiers
            if val:
                modifiers = modifiers | (val<<ind.key)
            else:
                modifiers = modifiers & ~(val<<ind.key)
        def __togglemod(ind:int)->None:
            global modifiers
            modifiers = modifiers ^ (1<<ind.key)
        global cursorloc# where in the text the cursor is. default is -1, which means at the end
        cursorloc=0 # (define as how many characters are after the cursor). 0 is at the end of the text, 1 is one character back, etc. not dependent on the length of the text
        global menulock
        menulock = False
        def __on_key_press_event(event):
            global REGISTER_KEYS, notemode_active, G_notes, notebox
            if notemode_active:
                global G_notes,notebox, modifiers,cursorloc
                if event.key == [k for k,v in _keybindings.items() if v[0] == 'NOTES'][0]:
                    notemode_active = False #process the keybinding while in notes mode to exit notes mode
                    noteax.set(zorder=-10)
                    notebox.set(zorder=-10)
                    fig.canvas.blit(noteax.bbox)
                    notebox.set(bbox=dict(facecolor='#fff0',edgecolor='#fff0'),color='#fff0')
                    noteax.set(facecolor='#fff0')
                    REGISTER_KEYS = True
                elif event.key in __modind: #process modifiers
                    __setmod(__modind(event.key), True)
                else: #process the keypress if not modifier or exit key
                    if notebox.zorder < ax.zorder:
                        notebox.set_zorder(ax.zorder+2)
                    if noteax.zorder < ax.zorder:
                        noteax.set_zorder(ax.zorder+1) # (this is a hack to make sure the text is on top of the axes when it is important)
                    elif event.key == 'backspace':
                        if cursorloc != len(G_notes):
                            G_notes = G_notes[:len(G_notes)-cursorloc-1]+G_notes[len(G_notes)-cursorloc:]
                        cursorloc = min(cursorloc, len(G_notes))
                    elif event.key in ['left','right']:
                        if event.key == 'left':
                            cursorloc = min(cursorloc+1, len(G_notes))
                        elif event.key == 'right':
                            cursorloc = max(cursorloc-1, 0)
                    elif len(event.key) == 1 or event.key in kmap:
                        k = kmap.get(event.key, event.key)
                        G_notes = G_notes[:len(G_notes)-cursorloc]+k+G_notes[len(G_notes)-cursorloc:]
                        cursorloc = min(cursorloc, len(G_notes))
                    text = G_notes[:len(G_notes)-cursorloc]+'|'+G_notes[len(G_notes)-cursorloc:]
                    notebox.set_text("Notes:\n"+text)
                    fig.canvas.blit(noteax.bbox)
                    fig.canvas.draw()

            elif REGISTER_KEYS:
                if event.key in _keybindings:
                    action, val,_ = _keybindings[event.key]
                    if action == 'ACTION':
                        selradio.set_active(__getflagindex('ACTION', val))
                    elif action == 'RETR':
                        retrbts.set_active(__getflagindex('RETR', val))
                    elif action == 'MORPH':
                        morphbts.set_active(__getflagindex('MORPH', val))
                    elif action == 'CHAIN':
                        chainbts.set_active(__getflagindex('CHAIN', val))
                    elif action == 'CVH':
                        cvhbts.set_active(0)
                    elif action == 'KSIZE':
                        ksizeslider.set_val(min(max(G_ksize+(2 if val else -2),1),11))
                    elif action == 'SAVE':
                        __event_save(None)
                    elif action == 'RESET':
                        __event_reset(None)
                    elif action == 'FSCRN':
                        __event_fullscreen(None)
                    elif action == 'CLOSE':
                        __event_close(None)
                    elif action == 'TOOLTIP':
                        if val != 1:
                            global G_show_tooltips
                            G_show_tooltips = not G_show_tooltips
                        else:
                            __show_key_menu(True)
                            return
                    elif action == 'KILL':
                        tqdm.write('PATHFINDER: closing process.')
                        plt.close()
                        exit()
                    elif action == 'MASK':
                        global show_mask
                        show_mask += 1
                        if show_mask >= 7:
                            show_mask = 0
                    elif action == 'CMAP':
                        global falsecolor
                        falsecolor+= 1
                        if falsecolor >= len(cmap_cycler):
                            falsecolor = 0
                    elif action == 'NOTES':
                        notemode_active = True
                        REGISTER_KEYS = True
                        if notebox.zorder < ax.zorder:
                            notebox.set_zorder(ax.zorder+2)
                        if noteax.zorder < ax.zorder:
                            noteax.set_zorder(ax.zorder+1)
                        notebox.set_text("Notes:\n"+G_notes+"_")
                        noteax.set_facecolor('#fff0')
                        notebox.set(bbox=dict(facecolor='#fff0',edgecolor='#fff'),color='#fff')
                        fig.canvas.blit(noteax.bbox)
                        fig.canvas.draw()
                    elif action == 'FIXLRANGE':
                        if val == 4:
                            global G_fixrange, G_ofixrange
                            if G_fixrange is None:
                                G_fixrange = G_ofixrange
                            else:
                                G_ofixrange = G_fixrange
                                G_fixrange = None
                        elif val< 4:
                            ind = int(val>1) # 0 for min, 1 for max val::{L+,L-,U+,U-,T,R}
                            if G_fixrange is None:
                                if G_ofixrange is None:
                                    G_fixrange = [0,1,0.01]
                                else:
                                    return
                            if len(G_fixrange) == 2:
                                G_fixrange.append(0.01)
                            step = G_fixrange[2] 
                            G_fixrange[ind] += step*(1-2*(val%2))
                            G_fixrange[ind] = min(max(G_fixrange[ind],0),1)
                            
                        elif val == 5:
                            if fixlrange is not None:
                                G_fixrange = fixlrange
                            else:
                                G_fixrange = None
                            G_ofixrange = G_ofixrange
                        elif val == 6:
                            if len(G_fixrange) == 2:
                                G_fixrange.append(0.01)
                            G_fixrange[2] = stepmap[stepmap.index(G_fixrange[2])+1 if stepmap.index(G_fixrange[2]) < len(stepmap)-1 else 0]
                elif event.key in HELPKEYS:
                    tt = [(k,v[2]) for k,v in _keybindings.items()]
                    maxlenk = max([len(t[0]) for t in tt])
                    maxlenv = max([len(t[1]) for t in tt])
                    template = f'║ {{k:^{maxlenk+2}}} ║ {{action:<{maxlenv+2}}} ║'
                    top =      f'╔═{"═"*(maxlenk+2) }═╦═{"═"*(maxlenv+2)      }═╗'  
                    bottom =   f'╚═{"═"*(maxlenk+2) }═╩═{"═"*(maxlenv+2)      }═╝'
                    mid =      f'╠═{"═"*(maxlenk+2) }═╬═{"═"*(maxlenv+2)      }═╣'
                    if TWOCOLS:
                        lenh = len(tt)//2
                        tt1 = tt[:lenh]
                        tt2 = tt[lenh:]
                        sep = "\t"
                        while len(tt1) != len(tt2):
                            if len(tt1) > len(tt2):
                                tt2.append(('',''))
                            elif len(tt1) < len(tt2):
                                tt1.append(('',''))
                            else:
                                break
                        title = template.format(k='Key', action='Action')
                        forms = [p+sep+p for p in [top, title, mid]]
                        for i in range((len(tt1))):
                            forms.append(template.format(k=tt1[i][0], action=tt1[i][1]) + sep + template.format(k=tt2[i][0], action=tt2[i][1]))
                        forms.append(bottom)
                        for f in forms:
                            tqdm.write(f)
                    else:
                        tqdm.write(top)
                        tqdm.write(template.format(k='Key', action='Action'))
                        tqdm.write(mid)
                        for k, action in tt:
                            tqdm.write(template.format(k=k, action=action))
                        tqdm.write(bottom+sep+bottom)
                __redraw_main(None)    
        def __on_key_release_event(event):
            pass
            # global modifiers
            # if event.key in __modind:
            #     __setmod(__modind(event.key), False)

        def __show_key_menu(_):
            global keybind, tab, menulock
            menulock = not menulock
            if menulock:
                bboxord = [-0.1,0, 1.1, 1,]
                keybind = ax.add_patch(mpl.patches.Rectangle(bboxord[0:2],bboxord[2],bboxord[3],facecolor='#fffa', lw=1, zorder=10, edgecolor='black', transform=ax.transAxes, clip_on=False))
                keys,values = zip(*_keybindings.items())
                coltexts = [keys[:len(keys)//2], [v[2] for v in values[:len(keys)//2]], keys[len(keys)//2:], [v[2] for v in values[len(keys)//2:]]]
                coltexts = [list(c) for c in coltexts]
                coltexts[0] = [f"{k} "u"\u21E8" for k in coltexts[0]]
                coltexts[2] = [f"{k} "u"\u21E8" for k in coltexts[2]]
                keylen = max([len(k) for k in keys])
                v2len = max([len(v[2]) for v in values])
                coltexts = [[f"{k:>{keylen}}" for k in coltexts[0]], [f" {v:<{v2len}}" for v in coltexts[1]], [f"{k:<{keylen}}" for k in coltexts[2]], [f" {v:<{v2len}}" for v in coltexts[3]]]
                while len(coltexts[0]) != len(coltexts[2]):
                    if len(coltexts[0]) > len(coltexts[2]):
                        coltexts[2].append('')
                        coltexts[3].append('')
                    elif len(coltexts[0]) < len(coltexts[2]):
                        coltexts[0].append('')
                        coltexts[1].append('')
                    else:
                        break
                v = [*zip(*coltexts)]
                tab = ax.table(cellText=v[1:] , cellLoc='left',colLoc='left', cellColours=[['#fff5']*len(coltexts)]*len(coltexts[0][1:]), colWidths=[0.1,0.4]*(len(coltexts)//2), colColours=['#fff5',]*len(coltexts), zorder=11, bbox=bboxord,edges='',colLabels=v[0],transform=ax.transAxes)
                tab.AXESPAD = 0
                tab.auto_set_column_width([0,1])
                tab.auto_set_column_width([2,3])
                tab.auto_set_font_size(True)
                
                for (i,j), cell in tab.get_celld().items():
                    if i in [0,2]:
                        cell.set_text_props(weight='800')
                    cell.PAD = 0
                    cell.set_fontsize(cell.get_fontsize()*1.2)
                tab.auto_set_font_size(False)
                
                
                
                
                fig.canvas.blit(ax.bbox)
                fig.canvas.draw()
                fig.canvas.flush_events()
            else:
                __redraw_main(None)
                
        key_events = [fig.canvas.mpl_connect('key_press_event', __on_key_press_event),fig.canvas.mpl_connect('key_release_event', __on_key_release_event)] #noqa: F841
        plt.show()
        if G_path is not None:
            pth = G_path.reshape(-1, 2)
            ret= fullxy_to_polar_arr(pth, normed, 40)
        else:
            ret= None
        if G_notes not in ['', None]:
            tqdm.write(f'{filename_from_path(fits_dir)}:\n {G_notes}')
        try:
            G_fits_obj.close()
        except: #noqa: E722
            pass

        
        return ret

#------------ QuickPlot Class (for power.py optional plotting) ----------------#
class QuickPlot:
    """Helper class for plotting image arrays in various formats, used in the power.py module when the plotting flag is set."""
    titles = {'raw': 'Image array in fits file.',
              'limbtrimmed': 'Image centred, limb-trimmed.',
              'extracted': 'Auroral region extracted.',
              'polar': 'Polar projection.',
              'sqr': 'ROI',
              'mask': 'ROI mask in image space',
              'brojected_full': 'Brojected full image.',
              'brojected_roi': 'Brojected ROI image.'}
    labelpairs= {'pixels': {'xlabel': 'pixels', 'ylabel': 'pixels'},
                'degpixels': {'xlabel': 'longitude pixels', 'ylabel': 'co-latitude pixels'},
                'deg': {'xlabel': 'SIII longitude [deg]', 'ylabel': 'co-latitude [deg]'}}
    def __init__(self):
        """Helper class for plotting image arrays in various formats, used in the power.py module when the plotting flag is set."""
        pass
    def __imcbar(self, ax, img, cmap='cubehelix', origin='lower', vlim=(1., 1000.), clabel='Intensity [kR]',fs=12,pad=0.05):
        im = ax.imshow(img, cmap=cmap,origin=origin, vmin=vlim[0], vmax=vlim[1])
        cbar = plt.colorbar(pad=0.05, ax=ax, mappable=im)
        cbar.ax.set_ylabel(clabel,fontsize=fs, labelpad=pad)

    def _plot_(self,ax, image, cmap='cubehelix', origin='lower', vlim=(1., 1000.)):# make a quick-look plot to check image array content:
        ax.set(title='Image array in fits file.', **self.labelpairs['degpixels'])
        self.__imcbar(ax, image, cmap=cmap, origin=origin, vlim=vlim)
        return ax


    def _plot_raw_limbtrimmed(self,ax, image,lons=np.arange(0,1440,1),lats=np.arange(0,720,1), cmap='cubehelix',vlim=(0.,1000.)):# Quick plot check of the centred, limb-trimmed image:
        ax.set(title='Image centred, limb-trimmed.',**self.labelpairs['degpixels'])
        ax.pcolormesh(lons,lats,np.flip(np.flip(image, axis=1)),cmap=cmap,
                    vmin=vlim[0],vmax=vlim[1])
        return ax

    def _plot_extracted_wcoords(self,ax:plt.Axes, image,cmap='cubehelix',vlim=(0., 1000.)):
        ax.set(title='Auroral region extracted.', **self.labelpairs['deg'])
        self.__imcbar(ax, image, cmap=cmap, vlim=vlim)
        ax.set_yticks(ticks=[0*4,10*4,20*4,30*4,40*4], labels=['0','10','20','30','40'])
        ax.set_xticks(ticks=[0*4,90*4,180*4,270*4,360*4], labels=['360','270','180','90','0'])
        return ax

    def _plot_polar_wregions(self,ax: plt.Axes, image,cpath, rho=np.linspace(0,40,num=160), theta=np.linspace(0,2*np.pi,num=1440), fs=12, cmap='cubehelix', vlim=(1.,1000.)):#?p
        # ==============================================================================
        # make polar projection plot
        # ==============================================================================
        # set up polar coords
        #rho    # colat vector with image pixel resolution steps
        #theta  # longitude vector in radian space and image pixel resolution steps
        dr = [[i[0] for i in cpath], [i[1] for i in cpath]]
        ax.set(title='Polar projection.', theta_zero_location='N',
            ylim=[0,40], **self.labelpairs['deg'])
        ax.set_yticks(ticks=[0,10,20,30,40], labels=['','','','',''])
        ax.set_xticklabels(['0','315','270','225','180','135','90','45'], # reverse these! ###################
                        fontweight='bold',fontsize=fs)
        ax.tick_params(axis='x',pad=-1.)  
        ax.fill_between(theta, 0, 40, alpha=0.2,hatch="/",color='gray')
        cm=ax.pcolormesh(theta,rho,image,cmap=cmap,vmin=vlim[0],vmax=vlim[1])
        ax.plot([np.radians(360-r) for r in dr[1]], dr[0], color='red',linewidth=3.)
        # -> OVERPLOT THE ROI IN POLAR PROJECTION HERE. <-
        # Add colourbar: ---------------------------------------------------------------
        cbar = plt.colorbar(ticks=[0.,100.,500.,900.],pad=0.05, mappable=cm, ax=ax)
        cbar.ax.set_yticklabels(['0','100','500','900'])
        cbar.ax.set_ylabel('Intensity [kR]',fontsize=12)
        return ax

    def _plot_sqr_wregions(self,ax,image,cpath,testlons=np.arange(0,360,0.25),testcolats=np.arange(0,40,0.25),cmap='cubehelix', vlim=(1.,1000.) ):#?sq
        # # === Plot full image array using pcolormesh so we can understand the masking
        # # process i.e. isolating pixels that fall inside the ROI =======================
        ax.set(title='ROI', xlabel='SIII longitude [deg]', ylabel='co-latitude [deg]')
        ax.set_xticks(ticks=[0,90,180,270,360], labels=['360','270','180','90','0'])
        dr = [[i[0] for i in cpath], [i[1] for i in cpath]]
        ax.pcolormesh(testlons,testcolats,image,cmap=cmap,vmin=vlim[0],vmax=vlim[1])
        ax.plot([360-r for r in dr[1]], dr[0],color='red',linewidth=3.)
        return ax
    def _plot_mask(self,ax,image,loc='ROI'):#?
        ax.imshow(image, origin='lower')
        ax.set(title=f'{loc} mask in image space', xlabel='longitude pixels', ylabel='co-latitude pixels')
        return ax
    def _plot_brojected(self,ax,image,loc='ROI',cmap='cubehelix', origin='lower', vlim=(1., 1000.), saveto=None):#?
        ax.set(title=f'Brojected {loc} image.', **self.labelpairs['pixels'])
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
