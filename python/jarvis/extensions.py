import numpy as np
import matplotlib.pyplot as plt
import time
import cmasher as cmr
import cv2
import os
from astropy.io import fits
from tqdm import tqdm

from .utils import fpath, hdulinfo
try:
    from PyQt6 import QtGui # type: ignore #
    
except ImportError:
    try:
        from PyQt5 import QtGui # type: ignore #
    except ImportError:
        tqdm.write('Warning: PyQt6 or PyQt5 not found, pathfinder app may not function correctly')


from matplotlib import font_manager 
from datetime import datetime
from .transforms import fullxy_to_polar_arr
from .utils import fitsheader, get_datetime, prepare_fits, rpath, split_path

from .polar import process_fits_file
from .cvis import getcontourhdu, mk_stripped_polar, savecontour_tofits

import matplotlib as mpl

#------------------- Tooltips and Info about Config Flags ---------------------#
_cvtrans = {
    'RETR':{'info':'modes to find the contours',
    'EXTERNAL': [cv2.RETR_EXTERNAL, 'retrieves only the extreme outer contours'], #0
    'LIST': [cv2.RETR_LIST, 'retrieves all of the contours without establishing any hierarchical relationships'], #1
    'CCOMP': [cv2.RETR_CCOMP, 'retrieves all of the contours and organizes them into a two-level hierarchy, where external contours are on the top level and internal contours are on the second level'], #2
    'TREE': [cv2.RETR_TREE, 'retrieves all of the contours and reconstructs a full hierarchy of nested contours'] #3
    },
    'CHAIN':{'info':'methods to approximate the contours', #0 is floodfill, but not used here
    'NONE': [cv2.CHAIN_APPROX_NONE, 'stores absolutely all the contour points. That is, any 2 subsequent points (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors, that is, max(abs(x1-x2),abs(y2-y1))==1.'], #1
    'SIMPLE': [cv2.CHAIN_APPROX_SIMPLE, 'compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points.'], #2
    'TC89_L1': [cv2.CHAIN_APPROX_TC89_L1, 'applies one of the flavors of the Teh-Chin chain approximation algorithm.'], #3
    'TC89_KCOS': [cv2.CHAIN_APPROX_TC89_KCOS, 'applies one of the flavors of the Teh-Chin chain approximation algorithm.'], #4
    'trans':(1,2,3,4)
    },
    'MORPH':{'info':'morphological operations to apply to the mask before finding the contours', 
    'ERODE': [cv2.MORPH_ERODE, 'Erodes away the boundaries of foreground object'], #0
    'DILATE': [cv2.MORPH_DILATE, 'Increases the object area'], #1
    'OPEN': [cv2.MORPH_OPEN, 'Remove small noise'], #2
    'CLOSE': [cv2.MORPH_CLOSE, 'Fill small holes'], #3
    'GRADIENT': [cv2.MORPH_GRADIENT, 'Difference between dilation and erosion of an image.'], #4
    'TOPHAT': [cv2.MORPH_TOPHAT, 'Difference between input image and Opening of the image'],    #5
    'BLACKHAT': [cv2.MORPH_BLACKHAT, 'Difference between the closing of the input image and input image'], #6
    'HITMISS': [cv2.MORPH_HITMISS, 'Extracts a particular structure from the image'], #7
    },
    'KSIZE':{'info':'kernel size for the morphological operations. Larger values will smooth out the contours more',
    },
    'CVH':{'info':'whether to use convex hulls of the contours. Reduces the complexity of the contours',
    'Convex Hull':[True, 'whether to use convex hulls of the contours. Reduces the complexity of the contours'],
    },
    'ACTION':{'info':'Select the action when clicking on a point in the image',
    'None':[0,''],
    'Add Luminosity':[1, 'Add a luminosity sample at the clicked point'],
    'Remove Luminosity':[-1, 'Remove a luminosity sample closest to the clicked point'],
    'Add IDPX':[2, 'Add an ID pixel at the clicked point'],
    'Remove IDPX':[-2, 'Remove an ID pixel closest to the clicked point'],
    'trans':(0,1,-1,2,-2)
    },
    'CLOSE':{'info':'Close the viewer'
    },
    'SAVE':{'info':'Save the selected contour, either to the current fits file or to a new file if a path has been provided.'
    },
    'RESET':{'info':'Reset the viewer pixel selections'
    },
    'FSCRN':{'info':'Toggle fullscreen mode'}}
def getflaglabels(flag:str):
    ident= list(_cvtrans[flag].keys())
    ident.remove('info')
    
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
def getflagindex(flag:str,label:str):
    ident = _cvtrans[flag]
    if ident.get('trans',None) is None:
        return _cvtrans[flag][label][0]
    else:
        v= ident['trans'].index(ident[label][0])
        return v

#--------------------------------- Keybindings --------------------------------#
_keybindings = {}
groups = [
{k:('CLICKMODE',v) for k,v in zip('01234',getflaglabels('ACTION'))},
{k:('MORPH',v) for k,v in zip('qwertyui',getflaglabels('MORPH'))},
{k:('RETR',v) for k,v in zip('asdf',getflaglabels('RETR'))},
{k:('CHAIN',v) for k,v in zip('zxcv',getflaglabels('CHAIN'))},
{k:('KSIZE',v) for k,v in zip('=-',[True,False])},
{k:('CVH',v) for k,v in zip('`','Convex Hull')},
{'enter':('save', 'Save to file'),
 'f10':('reset', 'Reset selections'),
 'f11':('fscreen', 'Toggle Fullscreen'),
 'escape':('close', 'Close the viewer'),
 '/' : ('tooltip', 'Show Tooltips'),
 'delete':('kill', 'Kill the current process'),
 'm':('mask', 'cycle mask display'),
 'n':('cmap', 'cycle colormap')},
 #'b':('bgnd','Toggle Black/White Background')
]
for g in groups:
    _keybindings.update(g)

#----------------------------- Style Definitions ------------------------------#
_bgs ={'main':'#000','sidebar':'#fff','legend':'#000'} #> Background colors (hierarchical) based on axes labels
def get_bg(ax):
    return _bgs.get(ax.get_label(), _bgs.get('sidebar', _bgs.get('main', '#fff')))
_legendkws =dict(loc='lower left', fontsize=8, labelcolor='linecolor', frameon=False, mode='expand', ncol=3) #> active contour list styling
_idpxkws = dict(s=20, color='#ff0000ff', zorder=12, marker='x') # Identifier Pixel scatter style
_clickedkws = dict(s=20, color='#ff5500', zorder=12, marker='x') # Clicked Pixel scatter style
_selectclinekws = dict(s=0.3, color='#ff0000ff', zorder=10) # Selected Contour line style
_selectctextkws = dict(fontsize=8, color=_selectclinekws['color']) # Selected Contour text style
_otherclinekws = dict(s=0.2, color='#ffff0055', zorder=10) # Other Contour line style
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
def validatecfg():
    for k, v in _cvtrans.items():
        if len(v) > 2:
            trans = v.get('trans')
            if trans is not None:
                assert len(v) - 2 == len(trans), f'{k}\'s index translation length does not match the number of flags: "trans": {trans}, flags: {list(v.keys())}'
            else:
                maxflag = max(val[0] for label, val in v.items() if label != 'info')
                lenflags = len(v) - 2  # zero indexed, and accounting for`info' and no `trans'
                assert lenflags == maxflag, f'{k}\'s index translation length does not match the number of flags, highest flag value: {maxflag}, flags: {list(v.keys())} (length: {lenflags})'
        for label, val in v.items():
            if label not in ['trans', 'info']:
                assert len(val) == 2, f'{k}\'s index translation values must be a list of length 2, containing the flag value (or index) and description'
                assert isinstance(val[0], (int, bool)), f'{k}\'s index translation values must be integers or booleans'
                assert isinstance(val[1], str), f'{k}\'s index translation descriptions must be strings'
    for k, v in _keybindings.items():
        assert len(v) == 2, f'{k}\'s keybinding must be a tuple of length 2'
    assert os.path.exists(_iconpath), 'Icon file not found'
    assert os.path.exists(_fontpath), 'Font file not found'
if FIRST_RUN:
    validatecfg()


# ------------------------- pathfinder (Function) -----------------------------#
def pathfinder(fits_dir: fits.HDUList,saveloc=None,show_tooltips=True, headername='BOUNDARY',morphex=(cv2.MORPH_CLOSE,cv2.MORPH_OPEN), fcmode=cv2.RETR_EXTERNAL, fcmethod=cv2.CHAIN_APPROX_SIMPLE, cvh=False, ksize=5,**persists):
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
    global G_fits_obj
    G_fits_obj = fits.open(fits_dir, mode='update')
    font_manager.fontManager.addfont(fpath('python/jarvis/resources/FiraCodeNerdFont-Regular.ttf'))
    with mpl.rc_context(rc={'font.family': 'FiraCode Nerd Font', 'axes.unicode_minus': False, 'toolbar': 'None', 'font.size': 8,
                            'keymap.fullscreen': '', 'keymap.home': '', 'keymap.back': '', 'keymap.forward': '', 'keymap.pan': '',
                            'keymap.zoom': '', 'keymap.save': '', 'keymap.quit': '', 'keymap.grid': '', 'keymap.yscale': '',
                            'keymap.xscale': '', 'keymap.copy':''}):
        #generate a stripped down, grey scale image of the fits file, and make normalised imagempl.rcParams['toolbar'] = 'None'
        proc =process_fits_file(prepare_fits(G_fits_obj, fixed='LON', full=True))
        img = mk_stripped_polar(proc, cmap=cmr.neutral, ax_background='white', img_background='black') 
        global show_mask, G_clicktype, G_path, G_headername, G_show_tooltips, G_morphex, G_fcmode, G_fcmethod, G_cvh, G_ksize, falsecolor
        global FIRST_RUN
        if FIRST_RUN:
            tqdm.write('(Press # to print keybindings)')
            G_show_tooltips = show_tooltips
            G_clicktype=persists.get('G_clicktype', 0)
            G_morphex = [m for m in morphex]
            show_mask = 0
            G_fcmode = fcmode
            G_fcmethod = fcmethod
            G_cvh = cvh
            G_ksize = ksize
            falsecolor = 0
            G_headername = 'BOUNDARY' if headername is None else headername
            FIRST_RUN = False
        normed = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        #set up global variables needed for the click event and configuration options
        global imarea
        imarea = (np.pi*(normed.shape[0]/2)**2)/2
        global clicked_coords
        clicked_coords = persists.get('clicked_coords', [])
        global id_pixels
        id_pixels = persists.get('id_pixels', [])
        G_path = persists.get('G_path', None)
        global retattrs 
        retattrs = dict(LMIN =-np.inf, LMAX=np.inf, NUMPTS=np.nan, XYA_CT=np.nan, XYA_CTP=np.nan, XYA_IMG=imarea)
        def getareapct(contourarea):
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
        linfax = fig.add_subplot(mgs[0:2,:8], label='infotext')
        clax = fig.add_subplot(mgs[0,8:], label='CLOSE')
        sbax = fig.add_subplot(mgs[1,8:], label='SAVE')
        headernameax = fig.add_subplot(mgs[2,8:], label='headername')
        rbax = fig.add_subplot(mgs[3,8:], label='RESET')
        fsax = fig.add_subplot(mgs[4,8:], label='FSCRN')
        flax = fig.add_subplot(mgs[2:5,:8], label='ACTION', facecolor=_bgs['sidebar'])
        retrax = fig.add_subplot(mgs[5,:8], label='RETR', facecolor=_bgs['sidebar'])
        morphax = fig.add_subplot(mgs[5:7,8:], label='MORPH', facecolor=_bgs['sidebar'])
        chainax = fig.add_subplot(mgs[6,:8], label='CHAIN', facecolor=_bgs['sidebar'])
        ksizeax = fig.add_subplot(mgs[8,:8], label='KSIZE', facecolor=_bgs['sidebar'])
        cvhax = fig.add_subplot(mgs[7:9,8:], label='CHAIN', facecolor=_bgs['sidebar'])
        ksizesubax = fig.add_subplot(mgs[7,:8], label='KSIZE', facecolor=_bgs['sidebar'])
        fitinfoax = fig.add_subplot(mgs[9,:], label='fitinfo', facecolor=_bgs['sidebar'])
        lax = fig.add_subplot(mgs[-1,:], label='legend', facecolor=_bgs['legend'])
        ax = fig.add_subplot(gs[0],label='main', zorder=11)
        if show_tooltips:
            tooltipax = fig.add_subplot(mgs[10,:], label='tooltip', facecolor='#000')
            tooltip = tooltipax.text(0.012, 0.98, '', fontsize=6, color='black', ha='left', va='top', wrap=True, bbox=dict(facecolor='white', alpha=0.5, lw=0), zorder=10)    
        axs = [ax, lax, rbax, linfax,  retrax, morphax, chainax, flax, sbax, cvhax, ksizeax, fitinfoax, clax, ksizesubax, fsax]+([tooltipax] if show_tooltips else [])
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
        def update_fitsinfo():
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
            hdulinf = hdulinfo(G_fits_obj)
            ctxt = [['File',f'{pthparts[0]}'], *[[' ', f"{p}"] for p in pthparts[1:]],
                                    ['Inst.',f'{inst} ({tel}, {filt} filter)'],
                                    ['Obs.',f'visit {visit} of {pln} ({hem})'],
                                    ['Date',f'{dt.strftime("%d/%m/%Y")}, day {doy}'],
                                    ['Time',f'{dt.strftime("%H:%M:%S")}(ICRS)'],
                                    ['CML',f'{cml:.4f}°'],
                                    ['HDUs',f"{0} {hdulinf[0]['name']}({hdulinf[0]['type']})"], *[[" ", f"{i} {h['name']}({h['type']})"] for i,h in enumerate(hdulinf[1:],1)]]
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
        update_fitsinfo()
        def binopts():
            global G_morphex, G_fcmode, G_fcmethod, G_cvh, G_ksize

            m_ = [0,]*8
            for m in G_morphex:
                m_[m]=1
            m_ = "".join([str(i) for i in m_])
            return dict(MORPH=m_, RETR=G_fcmode, CHAIN=G_fcmethod, CVH=int(G_cvh), KSIZE=G_ksize, CONTTIME=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        #main update function
        def update_fig(cl):
            global id_pixels, G_clicked_coords, G_morphex, G_fcmode, G_fcmethod, G_cvh, G_ksize, G_path, retattrs,  show_mask, falsecolor
            for a in [ax, lax]:
                a.clear()
                a.set_facecolor(get_bg(a))
            ax.imshow(normed, cmap=cmap_cycler[falsecolor], zorder=0)
            if len (id_pixels) > 0:
                pxsc = [idpixels[0] for idpixels in id_pixels], [idpixels[1] for idpixels in id_pixels]
                ax.scatter(*pxsc, **_idpxkws)
            if len(clicked_coords) > 0:
                scpc = [c[1][0] for c in clicked_coords], [c[1][1] for c in clicked_coords]
                ax.scatter(*scpc, **_clickedkws)
            if len(clicked_coords) >1:
                lrange = [min(c[0] for c in clicked_coords), max(c[0] for c in clicked_coords)]
                mask = cv2.inRange(normed, lrange[0]*255, lrange[1]*255)
                retattrs.update({'LMIN':lrange[0], 'LMAX':lrange[1]})
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
                linfax.text(0.5,0.75, f'lrange = ({lrange[0]:.3f}, {lrange[1]:.3f})', fontsize=8, color='black', ha='center', va='center')
                if len(id_pixels) > 0: # if an id_pixel is selected, find the contour that encloses it
                    selected_contours = []
                    other_contours = []
                    for i,contour in enumerate(sortedcs):
                        if all([cv2.pointPolygonTest(contour, id_pixel, False) > 0 for id_pixel in id_pixels]):  # >0 means inside
                            selected_contours.append([i,contour])
                        else:
                            other_contours.append([i,contour])
                    #if a contour is found, convert the contour points to polar coordinates and return them
                    for (i, contour) in selected_contours:
                        ax.scatter(*contour.T,**_selectclinekws)
                        cpath = mpl.path.Path(contour.reshape(-1, 2))
                        ax.plot(*cpath.vertices.T, color=_selectclinekws['color'], lw=_selectclinekws['s'])
                        ax.text(contour[0][0][0], contour[0][0][1], f"{i}", **_selectctextkws)
                    for (i, contour) in other_contours:
                        if i <100 and len(contour)>2 and len(contour)<1000:
                            ax.scatter(*contour.T,**_otherclinekws)
                            cpath = mpl.path.Path(contour.reshape(-1, 2))
                            ax.plot(*cpath.vertices.T, color=_otherclinekws['color'], lw=_otherclinekws['s'])
                            ax.text(contour[0][0][0], contour[0][0][1], f"{i}", **_otherctextkws)
                    linfax.text(0.5, 0.5, f'Contours: {len(selected_contours)}({len(other_contours)} invalid)', fontsize=8, color='black', ha='center', va='center')
                    if len(selected_contours) >0:
                        linfax.text(0.5, 0.25, f'Area: {getareapct(cv2.contourArea(selected_contours[0][1]))}, N:{len(selected_contours[0][1])}', fontsize=10, color='black', ha='center', va='center')
                        G_path = selected_contours[0][1]
                        retattrs.update({'NUMPTS':len(selected_contours[0][1]), 'XYAREA_CTR':cv2.contourArea(selected_contours[0][1]), 'XYAREA_CTRP':getareapct(cv2.contourArea(selected_contours[0][1]))})
                    selectedhandles = [mpl.lines.Line2D([0], [0], label=f'{i}, {getareapct(cv2.contourArea(sortedcs[i]))}', **_handles['selectedc']) for i in [c[0] for c in selected_contours]]
                    otherhandles = [mpl.lines.Line2D([0], [0], label=f'{i}, {getareapct(cv2.contourArea(sortedcs[i]))}', **_handles['otherc']) for i in [c[0] for c in other_contours]]
                    lax.legend(handles=selectedhandles+otherhandles[:min(50,len(otherhandles)-1)], **_legendkws)
                else:
                    for i,contour in enumerate(sortedcs):
                            if i <100 and len(contour)>2 and len(contour)<1000:
                                ax.scatter(*contour.T,**_defclinekws)
                                cpath = mpl.path.Path(contour.reshape(-1, 2))
                                ax.plot(*cpath.vertices.T, color=_defclinekws['color'], lw=_defclinekws['s'])
                                ax.text(contour[0][0][0], contour[0][0][1], f"{i}", **_defctextkws)
                    linfax.text(0.5, 0.5, f'Contours: {len(sortedcs)}', fontsize=8, color='black', ha='center', va='center')
                    handles = [mpl.lines.Line2D([0], [0], label=f'{i}, {getareapct(cv2.contourArea(sortedcs[i]))}',**_handles['defc'] ) for i in range(len(sortedcs))][:min(50,len(sortedcs)-1)]
                    lax.legend(handles=handles, **_legendkws)
                    lax.set_facecolor(get_bg(lax))
            else:
                linfax.clear()
                linfax.text(0.5, 0.5, 'Select at least 2 or more\nLuminosity Samples', fontsize=8, color='black', ha='center', va='center')
                lax.set_facecolor(get_bg(ax))
            fig.canvas.draw()
            fig.canvas.flush_events()
        #config gui elements
        #---- RESET button ----#
        breset = mpl.widgets.Button(rbax, "Reset      "+u'\uF0E2')
        def reset(event):
            global clicked_coords, id_pixels
            clicked_coords = []
            id_pixels = []
            update_fig(None)
        breset.on_clicked(reset)
        #---- SAVE button ----#
        bsave = mpl.widgets.Button(sbax, "Save       "+u'\uEB4B')
        def save(event):
            global G_path, G_fits_obj,retattrs,G_headername
            if G_path is not None:
                pth = G_path.reshape(-1, 2)
                pth = fullxy_to_polar_arr(pth, normed, 40)
                if saveloc is not None:
                    n_fits_obj = savecontour_tofits(fits_obj=G_fits_obj, cont=pth)
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
                    nhattr = retattrs
                    nhattr |=dict()
                    nhattr |= binopts()
                    nhattr |= {m:fitsheader(G_fits_obj,m) for m in ['UDATE','YEAR','VISIT','DOY']}
                    header = fits.Header(nhattr)
                    G_fits_obj.append(getcontourhdu(pth, name=G_headername.upper(), header=header)) 
                    G_fits_obj.flush()
                    tqdm.write(f"Save Successful, contour added to fits file at index {len(G_fits_obj)-1}")
                    linfax.clear()
                    linfax.text(0.5, 0.5, "Save Successful, contour added\n to fits file at index"+str(len(G_fits_obj)-1), fontsize=8, color='black', ha='center', va='center')
                    fig.canvas.blit(linfax.bbox)
                    update_fitsinfo()
            else:
                tqdm.write("Save Failed: No contour selected to save")
        bsave.on_clicked(save)
        #---- RETR options ----#
        retrbts = mpl.widgets.RadioButtons(retrax,getflaglabels('RETR'), active=0)
        def retrfunc(label):
            global G_fcmode
            G_fcmode = _cvtrans['RETR'][label][0]
        retrbts.on_clicked(retrfunc)
        #---- MORPH options ----#
        bopts = getflaglabels('MORPH')
        acts = [True if _cvtrans['MORPH'][b][0] in G_morphex else False for b in bopts]
        morphbts = mpl.widgets.CheckButtons(morphax, bopts, acts)
        def morphfunc(lb):
            global G_morphex
            if _cvtrans['MORPH'][lb][0] in G_morphex:
                G_morphex.remove(_cvtrans['MORPH'][lb][0])
            else:
                G_morphex.append(_cvtrans['MORPH'][lb][0])
        morphbts.on_clicked(morphfunc)
        #---- CHAIN options ----#
        chainbts = mpl.widgets.RadioButtons(chainax,getflaglabels('CHAIN'), active=1)
        def chainfunc(label):
            global G_fcmethod
            G_fcmethod = _cvtrans['CHAIN'][label][0]
        chainbts.on_clicked(chainfunc)
        #---- CLOSE button ----#
        bclose = mpl.widgets.Button(clax, "Close      "+u'\uE20D', color='#ffaaaa', hovercolor='#ff6666')
        #bclose.label.set(fontsize=14)
        def close(event):
            plt.close()
        bclose.on_clicked(close)
        #---- CVH options ----#
        cvhbts = mpl.widgets.CheckButtons(cvhax, ['Convex Hull'], [G_cvh],)
        def cvhfunc(label):
            global G_cvh
            G_cvh = not G_cvh
        cvhbts.on_clicked(cvhfunc)
        #---- KSIZE options ----#
        ksizeslider = mpl.widgets.Slider(ksizeax,"", valmin=1, valmax=11, valinit=G_ksize, valstep=2, color='orange', initcolor='None')
        ksizeax.add_patch(mpl.patches.Rectangle((0,0), 1, 1, facecolor='#fff', lw=1, zorder=-1, transform=ksizeax.transAxes, edgecolor='black'))
        ksizeax.add_patch(mpl.patches.Rectangle((0,0.995), 5/12, 1.005, facecolor='#fff', lw=0, zorder=10, transform=ksizeax.transAxes))
        def ksizefunc(val):
            global G_ksize
            global clicked_coords
            ksizeax.set_facecolor('#fff')
            G_ksize = int(val)
            ksizesubax.clear()
            ksizesubax.text(0.1, 0.45, f'ksize: {G_ksize}', fontsize=8, color='black', ha='left', va='center')
            fig.canvas.blit(ksizesubax.bbox)
            update_fig(clicked_coords)
        ksizefunc(G_ksize)
        ksizeslider.on_changed(ksizefunc)
        #---- SELECTMODE options ----#
        _radioprops={'facecolor':['#000', _clickedkws['color'], _clickedkws['color'], _idpxkws['color'], _idpxkws['color']], 
                    'edgecolor':['#000', _clickedkws['color'], _clickedkws['color'], _idpxkws['color'], _idpxkws['color']],
                    'marker': ['o','o','X','o','X']}
        _labelprops = {'color': ['#000', _clickedkws['color'], _clickedkws['color'], _idpxkws['color'], _idpxkws['color']]}
        selradio =  mpl.widgets.RadioButtons(flax, ['None', 'Add Luminosity', 'Remove Luminosity', 'Add IDPX', 'Remove IDPX'], active=0, radio_props=_radioprops, label_props=_labelprops) 
        def selfunc(label):
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
        selradio.on_clicked(selfunc)
        #---- FULLSCREEN button ----#
        bfullscreen = mpl.widgets.Button(fsax, 'Fullscreen '+u'\uF50C')
        #bfullscreen.label.set(fontsize=14)  
        def fullscreen(event):
            fig.canvas.manager.full_screen_toggle()
        bfullscreen.on_clicked(fullscreen)

        #---- CLICK options ----#
        def on_click(event):
            global clicked_coords
            global id_pixels
            global G_clicktype
            if event.inaxes == ax:
                if event.xdata is not None and event.ydata is not None:
                    click_coords = (event.xdata, event.ydata)
                    # get the luminance value of the clicked pixel
                    if G_clicktype == 1:
                        clicked_coords.append([img[int(event.ydata), int(event.xdata)]/255, click_coords])
                    elif G_clicktype == -1:
                        if len(clicked_coords) > 0:
                            clicked_coords.pop(np.argmin([np.linalg.norm(np.array(c[1])-np.array(click_coords)) for c in clicked_coords]))
                    elif G_clicktype == 2:
                        id_pixels.append(click_coords)
                    elif G_clicktype == -2:
                        if len(id_pixels) > 0:
                            id_pixels.pop(np.argmin([np.linalg.norm(np.array(c)-np.array(click_coords)) for c in id_pixels]))   
            update_fig(clicked_coords)
        click_event = fig.canvas.mpl_connect('button_press_event', on_click) #noqa: F841
        #---- HOVER options ----#
        
        def on_hover(event):
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
        hover_event = fig.canvas.mpl_connect('motion_notify_event', on_hover) #noqa: F841
        #----TEXTENTRY----#
        textbox = mpl.widgets.TextBox(headernameax, 'Header Name:', initial=G_headername)
        delchars = ''.join(c for c in map(chr, range(1114111)) if not c.isalnum() and c not in [' ','_','-'])
        delmap = str.maketrans("", "", delchars)
        def on_textchange(text:str):
            global G_headername
            text = text.translate(delmap)
            G_headername = text.replace(" ","_").replace()
        textbox.on_submit(on_textchange)
        #----KEYPRESS----#
        def on_key(event):
            if event.key in _keybindings:
                action, val = _keybindings[event.key]
                if action == 'CLICKMODE':
                    selradio.set_active(getflagindex('ACTION', val))
                elif action == 'RETR':
                    retrbts.set_active(getflagindex('RETR', val))
                elif action == 'MORPH':
                    morphbts.set_active(getflagindex('MORPH', val))
                elif action == 'CHAIN':
                    chainbts.set_active(getflagindex('CHAIN', val))
                elif action == 'CVH':
                    cvhbts.set_active(0)
                elif action == 'KSIZE':
                    ksizeslider.set_val(min(max(G_ksize+(2 if val else -2),1),11))
                elif action in ['save', 'reset', 'fscreen', 'close']:
                    exec(action+'(None)')
                elif action == 'tooltip':
                    global G_show_tooltips
                    G_show_tooltips = not G_show_tooltips
                elif action == 'kill':
                    plt.close()
                    exit()
                elif action == 'mask':
                    global show_mask
                    show_mask += 1
                    if show_mask >= 7:
                        show_mask = 0
                   
                elif action == 'cmap':
                    global falsecolor
                    falsecolor+= 1
                    if falsecolor >= len(cmap_cycler):
                        falsecolor = 0
                    
                        
            elif event.key == '#':
                tt = []
                for k,v in _keybindings.items():
                    if v[0] == 'tooltip':
                        tt.append((k, 'Toggle Tooltips'))
                    elif v[0] == 'kill':
                        tt.append((k, 'Force Quit Python'))
                    elif v[0] == 'RETR':
                        tt.append((k, f'Select RETR Mode: {v[1]}'))
                    elif v[0] == 'MORPH':
                        tt.append((k, f'Toggle Morphology: {v[1]}'))
                    elif v[0] == 'CHAIN':
                        tt.append((k, f'Select Approximation Mode: {v[1]}'))
                    elif v[0] == 'CVH':
                        tt.append((k, 'Toggle Convex Hull'))
                    elif v[0] == 'KSIZE':
                        tt.append((k, f'Change Kernel Size: {"Increase" if v[1] else "Decrease"}'))
                    elif v[0] == 'save':
                        tt.append((k, 'Save Contour'))
                    elif v[0] == 'reset':
                        tt.append((k, 'Reset'))
                    elif v[0] == 'fscreen':
                        tt.append((k, 'Toggle Fullscreen'))
                    elif v[0] == 'close':
                        tt.append((k, 'Close Viewer'))
                    elif v[0] =='CLICKMODE':
                        tt.append((k, f'Select Mode: {v[1]}'))
                    else:
                        tt.append((k, str(v[0])+": "+str(v[1])))
                maxlenk = max([len(t[0]) for t in tt])
                maxlenv = max([len(t[1]) for t in tt])
                template = f'║ {{k:^{maxlenk+2}}} ║ {{action:<{maxlenv+2}}} ║'
                top =      f'╔═{"═"*(maxlenk+2) }═╦═{"═"*(maxlenv+2)      }═╗'  
                bottom =   f'╚═{"═"*(maxlenk+2) }═╩═{"═"*(maxlenv+2)      }═╝'
                mid =      f'╠═{"═"*(maxlenk+2) }═╬═{"═"*(maxlenv+2)      }═╣'
                tqdm.write(top)
                tqdm.write(template.format(k='Key', action='Action'))
                tqdm.write(mid)
                for k, action in tt:
                    tqdm.write(template.format(k=k, action=action))
                tqdm.write(bottom)
            update_fig(None)    

        key_event = fig.canvas.mpl_connect('key_press_event', on_key) #noqa: F841
        
        plt.show()
        if G_path is not None:
            pth = G_path.reshape(-1, 2)
            ret= fullxy_to_polar_arr(pth, normed, 40)
        else:
            ret= None
        return ret

#---------------------------- PathFinder (Class) ------------------------------#                               

class PathFinder:
    
    _radioprops={'facecolor':['#000', _clickedkws['color'], _clickedkws['color'], _idpxkws['color'], _idpxkws['color']], 
                 'edgecolor':['#000', _clickedkws['color'], _clickedkws['color'], _idpxkws['color'], _idpxkws['color']],
                 'marker': ['o','o','X','o','X']}
    _labelprops = {'color': ['#000', _clickedkws['color'], _clickedkws['color'], _idpxkws['color'], _idpxkws['color']]}

    def __init__(self, fits_dir: fits.HDUList,saveloc=None,show_tooltips=True, morphex=(cv2.MORPH_CLOSE,cv2.MORPH_OPEN), fcmode=cv2.RETR_EXTERNAL, fcmethod=cv2.CHAIN_APPROX_SIMPLE, cvh=False, ksize=5,**persists):
        """### *JAR:VIS* Pathfinder Class
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
        raise NotImplementedError("This class is not yet implemented")
        mpl.use('QtAgg')
        from matplotlib.font_manager import fontManager
        fontManager.addfont()
        plt.rcParams['font.family'] = 'FiraCode Nerd Font'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['toolbar'] = 'None'
        self.replace_fitsobj(fits_dir)
        self.CFG = dict(
            show_tooltips = show_tooltips,
            morphex = morphex,
            fcmode = fcmode,
            fcmethod = fcmethod,
            cvh = cvh,
            ksize = ksize,
        )
        self.samples = dict(luminosity=persists.get('clicked_coords', []), identifiers=persists.get('identifiers', []))
        self.current_contour = persists.get('current_contour', None)
        self.fig = plt.figure()
        self.artists = dict()
        self.eventconns = dict()
        self.widgets = dict()
        
        
    def update_qtapp(self, fits_dir):
        fw = [[self.fig.canvas.width(), int(720*1.5)],[self.fig.canvas.height(), 720]]
        if any(i[0] < i[1] for i in fw):
            self.fig.canvas.manager.window.setGeometry(0,0,fw[0][1],fw[1][1])
        self.fig.set_size_inches(fw[0][1]/self.fig.dpi, fw[1][1]/self.fig.dpi)
        self._qapp = self.fig.canvas.manager.window
        self._qapp.setWindowTitle(f'JAR:VIS Pathfinder ({fits_dir})')
        
        self._qapp.setWindowIcon(QtGui.QIcon(self._iconpath))
    def getareapct(self,contourarea):
        num = contourarea/self.IMAREA * 100
        if num < 0.01:
            return f'{num:.2e}%'
        return f'{num:.2f}%'    

        # generate a stripped down, grey scale image of the fits file, and make normalised imagempl.rcParams['toolbar'] = 'None'
    def replace_fitsobj(self, fits_dir):
        self.read_dir = fits_dir
        self.fits_obj = fits.open(fits_dir, mode='update')
        proc =process_fits_file(prepare_fits(self.fits_obj, fixed='LON', full=True))
        img = mk_stripped_polar(proc, cmap=cmr.neutral, ax_background='white', img_background='black')
        self.normed = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        self.IMAREA = (np.pi*(self.normed.shape[0]/2)**2)/2
        self.update_qtapp(fits_dir)


    def init_axes(self):
        gs = self.fig.add_gridspec(1,2, wspace=0.05, hspace=0, width_ratios=[8,3], left=0, right=1, top=1, bottom=0)
        # mgs: 17 rows, 12 columns, to layout the options
        mgs = gs[1].subgridspec(11,12, wspace=0, hspace=0, height_ratios=[
                                0.5,  0.5,  0.5,  0.5,      
                                1, 1,      
                                0.3, 0.1,    
                                1.5,3,3]
        ) 
        # layout the axes in the gridspecs                  
        self._gridspecs = {'plot':gs[0], 'options':mgs}
        self._description_ax = self.fig.add_subplot(mgs[0:2,:8], label='DescriptionOutput')
        self._closebutton_ax = self.fig.add_subplot(mgs[0,8:], label='CLOSE')
        self._savebutton_ax = self.fig.add_subplot(mgs[1,8:], label='SAVE')
        self._resetbutton_ax = self.fig.add_subplot(mgs[2,8:], label='RESET')
        self._frcreenbutton_ax = self.fig.add_subplot(mgs[3,8:], label='FSCRN')
        self._clickselect_ax = self.fig.add_subplot(mgs[2:4,:8], label='ACTION', facecolor=_bgs['sidebar'])
        self._retrselect_ax = self.fig.add_subplot(mgs[4,:8], label='RETR', facecolor=_bgs['sidebar'])
        self._morphselect_ax = self.fig.add_subplot(mgs[4:6,8:], label='MORPH', facecolor=_bgs['sidebar'])
        self._chainselect_ax = self.fig.add_subplot(mgs[5,:8], label='CHAIN', facecolor=_bgs['sidebar'])
        self._ksizeslider_ax = self.fig.add_subplot(mgs[7,:8], label='KSIZE', facecolor=_bgs['sidebar'])
        self._cvhselect_ax = self.fig.add_subplot(mgs[6:8,8:], label='CHAIN', facecolor=_bgs['sidebar'])
        self._ksizevalue_ax = self.fig.add_subplot(mgs[6,:8], label='KSIZE', facecolor=_bgs['sidebar'])
        self._fitfile_ax = self.fig.add_subplot(mgs[8,:], label='fitinfo', facecolor=_bgs['sidebar'])
        self._contourlist_ax = self.fig.add_subplot(mgs[-1,:], label='legend', facecolor=_bgs['legend'])
        self._ax = self.fig.add_subplot(gs[0],label='main', zorder=11)
        if self.CFG['show_tooltips']:
            self._tooltip_ax = self.fig.add_subplot(mgs[9,:], label='tooltip', facecolor='#000')
            self.artists['tootlip']=self._tooltip_ax.text(0.012, 0.98, '', fontsize=8, color='black', ha='left', va='top', wrap=True, bbox=dict(facecolor='white', alpha=0.5, lw=0), zorder=10)
        self._axeslist = [self._ax, self._contourlist_ax, self._closebutton_ax, self._description_ax,  self._retrselect_ax, self._morphselect_ax, self._chainselect_ax, self._clickselect_ax, self._savebutton_ax, self._cvhselect_ax, self._ksizeslider_ax, self._fitfile_ax, self._resetbutton_ax, self._ksizevalue_ax, self._frcreenbutton_ax]+([self._tooltip_ax] if self.CFG['show_tooltips'] else [])
        self._axgroups ={'buttons':[self._closebutton_ax, self._savebutton_ax, self._resetbutton_ax, self._frcreenbutton_ax],'textboxes':[self._description_ax, self._fitfile_ax],'options':[self._retrselect_ax, self._morphselect_ax, self._chainselect_ax, self._cvhselect_ax, self._ksizeslider_ax, self._ksizevalue_ax, self._clickselect_ax],'legends':[self._contourlist_ax]}

    def update_fitdesc(self):
        self._fitfile_ax.clear()
        visit, tel, inst, filt, pln, hem,cml,doy  = *fitsheader(self.fits_obj, 'VISIT', 'TELESCOP', 'INSTRUME', 'FILTER', 'PLANET', 'HEMISPH', 'CML', 'DOY'), 
        dt = get_datetime(self.fits_obj)
        self._fitfile_ax.text(0.04, 0.96, "FITS File Information" , fontsize=12, color='black', ha='left', va='top')
        pth = str(self.read_dir)
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
        ctxt = [['File',f'{pthparts[0]}'], *[[' ', f"{p}"] for p in pthparts[1:]],
                                ['Inst.',f'{inst} ({tel}, {filt} filter)'],
                                ['Obs.',f'visit {visit} of {pln} ({hem})'],
                                ['Date',f'{dt.strftime("%d/%m/%Y")}, day {doy}'],
                                ['Time',f'{dt.strftime("%H:%M:%S")}(ICRS)'],
                                ['CML',f'{cml:.4f}°']]
        tl=self._fitfile_ax.table(cellText=ctxt,cellLoc='right', cellColours=[[_bgs['sidebar'], _bgs['sidebar']] for i in range(len(ctxt))], colWidths=[0.14,1], colLabels=None, rowLabels=None, colColours=None, rowColours=None, colLoc='center', rowLoc='center', loc='bottom left', bbox=[0.02, 0.02, 0.96, 0.8], zorder=0)
        tl.visible_edges = ''
        tl.auto_set_font_size(False)
        tl.set_fontsize(9)
        for key, cell in tl.get_celld().items():
            cell.set_linewidth(0)
            cell.PAD =0
            if key[0]<len(pthparts) and key[1] == 1:
                cell.set_fontsize(10-len(pthparts)+1)
        self.fig.canvas.blit(self._fitfile_ax.bbox)
    def update_descriptor(self,*args, **kwargs):
        self._description_ax.clear()
        if len(args)>=2:
            x,y = args[:2]
        else:
            x,y = 0.5,0.5
        if len(args)==3:
            st = args[2]
        elif len(args)==1:
            st = args
        else:
            st=''
        kwargs.setdefault('fontsize',10)
        kwargs.setdefault('color','black')
        kwargs.setdefault('ha','center')
        kwargs.setdefault('va','center')
        self._description_ax.text(x, y, st, **kwargs)
        self.fig.canvas.blit(self._description_ax.bbox)
   
    
    def initialise_app(self):
        self._ksizeslider_ax.set_facecolor('#fff')
        for a in self._axeslist:
            a.xaxis.set_visible(False)
            a.yaxis.set_visible(False)
        self.fig.canvas.set_cursor(3)
        self.fig.set_facecolor('black')
        self.update_fitdesc(self.fits_obj)
    def update_mainax(self):
        for a in [self._ax, self._contourlist_ax, self._description_ax]:
            a.clear()
            a.set_facecolor(get_bg(a))
            self._ax.imshow(self.normed, cmap=cmr.neutral, zorder=0)
        if len (self.samples['identifiers']) > 0:
            pxsc = [idpixels[0] for idpixels in self.samples['identifiers']], [idpixels[1] for idpixels in self.samples['identifiers']]
            self._ax.scatter(*pxsc, **_idpxkws)
        if len(self.samples['luminosity']) > 0:
            scpc = [c[1][0] for c in self.samples['luminosity']], [c[1][1] for c in self.samples['luminosity']]
            self._ax.scatter(*scpc, **_clickedkws)
        if len(self.samples['luminosity']) >1:
            self.lrange = [min(c[0] for c in self.samples['luminosity']), max(c[0] for c in self.samples['luminosity'])]
            self.mask = cv2.inRange(self.normed, self.lrange[0]*255, self.lrange[1]*255)
            kernel = np.ones((self.CFG['ksize'], self.CFG['ksize']), np.uint8)
            for morph in self.CFG['morphex']:
                self.mask = cv2.morphologyEx(self.mask, morph, kernel)
            contours, hierarchy = cv2.findContours(image=self.mask, mode=self.CFG['fcmode'], method=self.CFG['fcmethod'])
            if self.CFG['cvh']:
                contours = [cv2.convexHull(cnt) for cnt in contours]
            sortedcs = sorted(contours, key=lambda x: cv2.contourArea(x))
            sortedcs.reverse()

            # update the description to show the lrange on row 1
            
            if len(self.samples['identifiers']) >0:
                selected_contours = []
                other_contours = []
                for i,contour in enumerate(sortedcs):
                    if all([cv2.pointPolygonTest(contour, id_pixel, False) > 0 for id_pixel in self.samples['identifiers']]):
                        selected_contours.append([i,contour])
                    else:
                        other_contours.append([i,contour])
                for (i, contour) in selected_contours:
                    self._ax.scatter(*contour.T,**_selectclinekws)
                    cpath = mpl.path.Path(contour.reshape(-1, 2))
                    self._ax.plot(*cpath.vertices.T, color=_selectclinekws['color'], lw=_selectclinekws['s'])
                    self._ax.text(contour[0][0][0], contour[0][0][1], f"{i}", **_selectctextkws)
                for (i, contour) in other_contours:
                    self._ax.scatter(*contour.T,**_otherclinekws)
                    cpath = mpl.path.Path(contour.reshape(-1, 2))
                    self._ax.plot(*cpath.vertices.T, color=_otherclinekws['color'], lw=_otherclinekws['s'])
                    self._ax.text(contour[0][0][0], contour[0][0][1], f"{i}", **_otherctextkws)

                # update the description output to show the number of contours on row 2

                if len(selected_contours) >0:
                   # update the description output to show the area of the selected contour on row 3
                    self.current_contour = selected_contours[0][1]
                selectedhandles = [mpl.lines.Line2D([0], [0], label=f'{i}, {self.getareapct(cv2.contourArea(sortedcs[i]))}', **_handles['selectedc']) for i in [c[0] for c in selected_contours]]
                otherhandles = [mpl.lines.Line2D([0], [0], label=f'{i}, {self.getareapct(cv2.contourArea(sortedcs[i]))}', **_handles['otherc']) for i in [c[0] for c in other_contours]]
                self._contourlist_ax.legend(handles=selectedhandles+otherhandles, **_legendkws)
            else:
                for i,contour in enumerate(sortedcs):
                    self._ax.scatter(*contour.T,**_defclinekws)
                    cpath = mpl.path.Path(contour.reshape(-1, 2))
                    self._ax.plot(*cpath.vertices.T, color=_defclinekws['color'], lw=_defclinekws['s'])
                    self._ax.text(contour[0][0][0], contour[0][0][1], f"{i}", **_defctextkws)
                self._contourlist_ax.text(0.5, 0.5, f'Contours: {len(sortedcs)}', fontsize=10, color='black', ha='center', va='center')
                handles = [mpl.lines.Line2D([0], [0], label=f'{i}, {self.getareapct(cv2.contourArea(sortedcs[i]))}',**_handles['defc'] ) for i in range(len(sortedcs))]
                self._contourlist_ax.legend(handles=handles, **_legendkws)
                self._contourlist_ax.set_facecolor(get_bg(self._contourlist_ax))
        else:
            # update the description output to show the prompt to select luminosity samples
            self._contourlist_ax.set_facecolor(get_bg(self._ax))
        self.fig.canvas.blit(self._ax.bbox)
        self.fig.canvas.blit(self._contourlist_ax.bbox)
    def _widget_reset_button(self, event):
        self.samples = dict(luminosity=[], identifiers=[])
        self.update_mainax()
    def _widget_close_button(self, event):
        plt.close()
    def _widget_fullscreen_button(self, event):
        self.fig.canvas.manager.full_screen_toggle()
    def _widget_save_button(self, event):
        if self.current_contour is not None:
            pth = self.current_contour.reshape(-1, 2)
            pth = fullxy_to_polar_arr(pth, self.normed, 40)
            if self.saveloc is not None: # if a save location is provided, save the contour to a new fits file
                n_fits_obj = savecontour_tofits(fits_obj=self.fits_obj, cont=pth)
                n_fits_obj.writeto(self.saveloc, overwrite=True)
                tqdm.write(f"Saved to {self.saveloc}, closing")
                self.update_descriptor("Saved to "+str(self.saveloc)+"\n, closing")
                time.sleep(1)
                plt.close()
                self.fits_obj.close()
                
                
            else:
                self.fits_obj.append(getcontourhdu(pth)) 
                self.fits_obj.flush()
                tqdm.write(f"Save Successful, contour added to fits file at index {len(self.fits_obj)-1}")
                self.update_descriptor("Save Successful, contour added\n to fits file at index"+str(len(self.fits_obj)-1))
        else:
            tqdm.write("Save Failed: No contour selected to save")
            self.update_descriptor("Save Failed: No contour selected to save")
    def _widget_retr_select(self, label):
        self.CFG['fcmode'] = _cvtrans['RETR'][label][0]
    def _widget_morph_select(self, lb):
        if _cvtrans['MORPH'][lb][0] in self.CFG['morphex']:
            self.CFG['morphex'].remove(_cvtrans['MORPH'][lb][0])
        else:
            self.CFG['morphex'].append(_cvtrans['MORPH'][lb][0])
    def _widget_chain_select(self, label):
        self.CFG['fcmethod'] = _cvtrans['CHAIN'][label][0]
    def _widget_cvh_select(self, label):
        self.CFG['cvh'] = not self.CFG['cvh']
    def _widget_ksize_slider(self, val):
        self._ksizeslider_ax.set_facecolor('#fff')
        self.CFG['ksize'] = int(val)
        self._ksizevalue_ax.clear()
        self._ksizevalue_ax.text(0.1, 0.45, f'ksize: {self.CFG["ksize"]}', fontsize=10, color='black', ha='left', va='center')
        self.fig.canvas.blit(self._ksizevalue_ax.bbox)
        #self.update_mainax()
    def _widget_click_select(self, label):
        if 'Luminosity' in label:
            ret = 1
        elif 'IDPX' in label:
            ret = 2
        if 'Remove' in label:
            ret = -ret
        self.G_clicktype = ret
    def _on_click(self, event):
        if event.inaxes == self._ax:
            if event.xdata is not None and event.ydata is not None:
                click_coords = (event.xdata, event.ydata)
                if self.G_clicktype == 1:
                    self.samples['luminosity'].append([self.normed[int(event.ydata), int(event.xdata)]/255, click_coords])
                elif self.G_clicktype == -1:
                    if len(self.samples['luminosity']) > 0:
                        self.samples['luminosity'].pop(np.argmin([np.linalg.norm(np.array(c[1])-np.array(click_coords)) for c in self.samples['luminosity']]))
                elif self.G_clicktype == 2:
                    self.samples['identifiers'].append(click_coords)
                elif self.G_clicktype == -2:
                    if len(self.samples['identifiers']) > 0:
                        self.samples['identifiers'].pop(np.argmin([np.linalg.norm(np.array(c)-np.array(click_coords)) for c in self.samples['identifiers']]))   
        self.update_mainax()
    def _on_move(self, event):
        inax = event.inaxes
        if inax in self._axgroups['options']+self._axgroups['buttons']:
            axlabel = inax.get_label()
            txts = [t for t in  inax.get_children() if isinstance(t, mpl.text.Text)]
            txts = [t for t in txts if t.get_text() != '']
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
            self.artists['tootlip'].set_text(f'{texti[0]}: {texti[1]}')                    
            self.fig.canvas.draw_idle()
            self.fig.canvas.blit(self._tooltip_ax.bbox)
    def _on_key(self, event):
        _keybindings = {
        '0': (self._widget_click_select,'None'),
        '1': (self._widget_click_select,'Add Luminosity'),
        '2': (self._widget_click_select,'Add IDPX'),
        '3': (self._widget_click_select,'Remove Luminosity'),
        '4': (self._widget_click_select,'Remove IDPX'),
        's':(self._widget_save_button,'none'),
        'r':(self._widget_reset_button,'none'),
        'f':(self._widget_fullscreen_button,'none'),
        'q':(self._widget_close_button,'none')
        }
        if event.key in _keybindings:
            action, val = _keybindings[event.key]
            action(val)
    def init_widgets(self):
        widgets = dict(
           RESET= mpl.widgets.Button(self._resetbutton_ax, "Reset      "+u'\uF0E2'),
           CLOSE= mpl.widgets.Button(self._closebutton_ax, "Close      "+u'\uF011'),
           SAVE= mpl.widgets.Button(self._savebutton_ax,  "Save      "+u'\uF0C7'),
           FSCREEN= mpl.widgets.Button(self._frcreenbutton_ax, "Fullscreen "+u'\uF50C'),
           RETR= mpl.widgets.RadioButtons(self._retrselect_ax, ['EXTERNAL', 'LIST', 'CCOMP', 'TREE'], active=0),
           MORPH= mpl.widgets.RadioButtons(self._morphselect_ax, *list(zip([[k,True if k in self.CFG['morphex'] else False] for k in list(_cvtrans['MORPH'].keys() - ['info'])]))),
           CHAIN= mpl.widgets.RadioButtons(self._chainselect_ax, ['NONE', 'SIMPLE', 'TC89_L1', 'TC89_KCOS'], active=1),
           CVH= mpl.widgets.RadioButtons(self._cvhselect_ax, ['Convex Hull'], active=[self.CFG['cvh']]),
           CLICK= mpl.widgets.RadioButtons(self._clickselect_ax, ['Add Luminosity', 'Remove Luminosity', 'Add IDPX', 'Remove IDPX'], active=0, radio_props=self._radioprops, label_props=self._labelprops),
           KSIZE= mpl.widgets.Slider(self._ksizeslider_ax,  valmin=1, valmax=11, valinit=G_ksize, valstep=2, color='orange', initcolor='None'))
        widgets['KSIZE'].add_patch(mpl.patches.Rectangle((0,0), 1, 1, facecolor='#fff', lw=1, zorder=-1, transform=self._ksizeslider_ax.transAxes, edgecolor='black'))
        widgets['KSIZE'].add_patch(mpl.patches.Rectangle((0,0.995), 5/12, 1.005, facecolor='#fff', lw=0, zorder=10, transform=self._ksizeslider_ax.transAxes))
        conns = dict(
            RESET= widgets['RESET'].on_clicked(self._widget_reset_button),
            CLOSE= widgets['CLOSE'].on_clicked(self._widget_close_button),
            SAVE= widgets['SAVE'].on_clicked(self._widget_save_button),
            FSCREEN= widgets['FSCREEN'].on_clicked(self._widget_fullscreen_button),
            RETR= widgets['RETR'].on_clicked(self._widget_retr_select),
            MORPH= widgets['MORPH'].on_clicked(self._widget_morph_select),
            CHAIN= widgets['CHAIN'].on_clicked(self._widget_chain_select),
            CVH= widgets['CVH'].on_clicked(self._widget_cvh_select),
            CLICK= widgets['CLICK'].on_clicked(self._widget_click_select),
            KSIZE= widgets['KSIZE'].on_changed(self._widget_ksize_slider))
        self.widgets = widgets
        self.eventconns.update(widgets=conns)
  
    def init_events(self):
        move_event = self.fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        click_event = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        key_event = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.eventconns.update(move=move_event, click=click_event, keypress=key_event)



    def run(self):
        self.init_axes()
        self.initialise_app()
        self.init_widgets()
        self.init_events()
        plt.show()
        self.fits_obj.close()
        return self.samples, self.current_contour

#------------ QuickPlot Class (for power.py optional plotting) ----------------#
class QuickPlot:
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
        


def get_pathfinderhead():
    # stdscr=curses.initscr()
    # curses.noecho()
    # curses.cbreak()
    # rows, cols = stdscr.getmaxyx()
    lines = [
        "[][][][][][][][][][][][][][][][][][][][][][][][][][]",
        "[]                                                []",
        "[]               JAR:VIS Pathfinder               []",
        "[]                                                []",
        "[]           (press # for keybindings.)           []",
        "[][][][][][][][][][][][][][][][][][][][][][][][][][]"
    ]
    for l in lines:
        tqdm.write(l)
