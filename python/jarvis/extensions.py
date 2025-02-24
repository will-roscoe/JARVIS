import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
import cv2
from .utils import fitsheader, fpath, get_datetime, prepare_fits, ensure_dir, rpath, split_path
from .polar import process_fits_file
from .cvis import mk_stripped_polar, savecontour_tofits
from astropy.io import fits
from tqdm import tqdm
import matplotlib as mpl
from PyQt6 import QtGui


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


    def __imcbar(self, ax, img, cmap='cubehelix', origin='lower', vlim=(1., 1000.), clabel='Intensity [kR]',fs=12,pad=0.05):
        im = ax.imshow(img, cmap=cmap,origin=origin, vmin=vlim[0], vmax=vlim[1])
        cbar = plt.colorbar(pad=0.05, ax=ax, mappable=im)
        cbar.ax.set_ylabel(clabel,fontsize=fs, pad=pad)

    def _plot_(self,ax, image_data, cmap='cubehelix', origin='lower', vlim=(1., 1000.)):# make a quick-look plot to check image array content:
        ax.set(title='Image array in fits file.', **self.labelpairs['degpixels'])
        self.__imcbar(ax, image_data, cmap=cmap, origin=origin, vlim=vlim)
        return ax


    def _plot_raw_limbtrimmed(self,ax, im_clean,lons=np.arange(0,1440,1),lats=np.arange(0,720,1), cmap='cubehelix',vlim=(0.,1000.)):# Quick plot check of the centred, limb-trimmed image:
        ax.set(title='Image centred, limb-trimmed.',**self.labelpairs['degpixels'])
        ax.pcolormesh(lons,lats,np.flip(np.flip(im_clean, axis=1)),cmap=cmap,
                    vmin=vlim[0],vmax=vlim[1])
        return ax

    def _plot_extracted_wcoords(self,ax:plt.Axes, image_extract,cmap='cubehelix',vlim=(0., 1000.)):
        ax.set(title='Auroral region extracted.', **self.labelpairs['deg'])
        self.__imcbar(ax, image_extract, cmap=cmap, vlim=vlim)
        ax.set_yticks(ticks=[0*4,10*4,20*4,30*4,40*4], labels=['0','10','20','30','40'])
        ax.set_xticks(ticks=[0*4,90*4,180*4,270*4,360*4], labels=['360','270','180','90','0'])
        return ax

    def _plot_polar_wregions(self,ax: plt.Axes, image_extract,cpath, rho=np.linspace(0,40,num=160), theta=np.linspace(0,2*np.pi,num=1440), fs=12, cmap='cubehelix', vlim=(1.,1000.)):#?p
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
        cm=ax.pcolormesh(theta,rho,image_extract,cmap=cmap,vmin=vlim[0],vmax=vlim[1])
        ax.plot([np.radians(360-r) for r in dr[1]], dr[0], color='red',linewidth=3.)
        # -> OVERPLOT THE ROI IN POLAR PROJECTION HERE. <-
        # Add colourbar: ---------------------------------------------------------------
        cbar = plt.colorbar(ticks=[0.,100.,500.,900.],pad=0.05, mappable=cm, ax=ax)
        cbar.ax.set_yticklabels(['0','100','500','900'])
        cbar.ax.set_ylabel('Intensity [kR]',fontsize=12)
        return ax

    def _plot_sqr_wregions(self,ax,image_extract,cpath,testlons=np.arange(0,360,0.25),testcolats=np.arange(0,40,0.25),cmap='cubehelix', vlim=(1.,1000.) ):#?sq
        # # === Plot full image array using pcolormesh so we can understand the masking
        # # process i.e. isolating pixels that fall inside the ROI =======================
        ax.set(title='ROI', xlabel='SIII longitude [deg]', ylabel='co-latitude [deg]')
        ax.set_xticks(ticks=[0,90,180,270,360], labels=['360','270','180','90','0'])
        dr = [[i[0] for i in cpath], [i[1] for i in cpath]]
        ax.pcolormesh(testlons,testcolats,image_extract,cmap=cmap,vmin=vlim[0],vmax=vlim[1])
        ax.plot([360-r for r in dr[1]], dr[0],color='red',linewidth=3.)
        return ax
    def _plot_mask(self,ax,dark_mask_2d,loc='ROI'):#?
        ax.imshow(dark_mask_2d, origin='lower')
        ax.set(title=f'{loc} mask in image space', xlabel='longitude pixels', ylabel='co-latitude pixels')
        return ax
    def _plot_brojected(self,ax,image,loc='ROI',cmap='cubehelix', origin='lower', vlim=(1., 1000.), saveto=None):#?
        ax.set(title=f'Brojected {loc} image.', **self.labelpairs['pixels'])
        self.__imcbar(ax, image, cmap=cmap, origin=origin, vlim=vlim)
        if saveto is not None:
            plt.savefig(saveto, dpi=350)
        return ax



_legendkws =dict(loc='upper right', fontsize=8, labelcolor='linecolor', frameon=False, mode='expand', ncol=3)
_flegkws = dict(loc='upper left', fontsize=8, frameon=False, mode='expand', ncol=1)
_idpxkws = dict(s=20, color='#ff0000ff', zorder=12, marker='x')
_clickedkws = dict(s=20, color='#ff5500', zorder=12, marker='x')
_selectclinekws = dict(s=0.3, color='#ff0000ff', zorder=10)
_selectctextkws = dict(fontsize=8, color=_selectclinekws['color'])
_otherclinekws = dict(s=0.2, color='#ffff0055', zorder=10)
_otherctextkws = dict(fontsize=8, color=_otherclinekws['color'])
_defclinekws = dict(s=_selectclinekws['s'], color=_selectclinekws['color'], zorder=_selectclinekws['zorder'])
_defctextkws = dict(fontsize=8, color=_defclinekws['color'])
_handles =dict(
    selectedc = dict(color=_selectclinekws['color'], lw=2),
    otherc = dict(color=_otherclinekws['color'], lw=2),
    defc = dict(color=_defclinekws['color'], lw=2),
    static = [
        mpl.lines.Line2D([0], [0], label='ID Pixel',      color=_idpxkws['color'], lw=0, marker='o', markersize=5),
        mpl.lines.Line2D([0], [0], label='Clicked Pixel', color=_clickedkws['color'], lw=0, marker='o', markersize=5),
    ]

)
_cvtrans = {
    'RETR':{
    'info':'modes to find the contours',
    'EXTERNAL': [cv2.RETR_EXTERNAL, 'retrieves only the extreme outer contours'],
    'LIST': [cv2.RETR_LIST, 'retrieves all of the contours without establishing any hierarchical relationships'],
    'CCOMP': [cv2.RETR_CCOMP, 'retrieves all of the contours and organizes them into a two-level hierarchy, where external contours are on the top level and internal contours are on the second level'],
    'TREE': [cv2.RETR_TREE, 'retrieves all of the contours and reconstructs a full hierarchy of nested contours']
    },
    'CHAIN':{
    'info':'methods to approximate the contours',
    'NONE': [cv2.CHAIN_APPROX_NONE, 'stores absolutely all the contour points. That is, any 2 subsequent points (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors, that is, max(abs(x1-x2),abs(y2-y1))==1.'],
    'SIMPLE': [cv2.CHAIN_APPROX_SIMPLE, 'compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points.'],
    'TC89_L1': [cv2.CHAIN_APPROX_TC89_L1, 'applies one of the flavors of the Teh-Chin chain approximation algorithm.'],
    'TC89_KCOS': [cv2.CHAIN_APPROX_TC89_KCOS, 'applies one of the flavors of the Teh-Chin chain approximation algorithm.']
    },
    'MORPH':{
    'info':'morphological operations to apply to the mask before finding the contours',
    'CLOSE': [cv2.MORPH_CLOSE, 'Fill small holes'],
    'OPEN': [cv2.MORPH_OPEN, 'Remove small noise'],
    'GRADIENT': [cv2.MORPH_GRADIENT, 'Difference between dilation and erosion of an image.'],
    'TOPHAT': [cv2.MORPH_TOPHAT, 'Difference between input image and Opening of the image'],
    'BLACKHAT': [cv2.MORPH_BLACKHAT, 'Difference between the closing of the input image and input image'],
    'HITMISS': [cv2.MORPH_HITMISS, 'Extracts a particular structure from the image'],
    'ERODE': [cv2.MORPH_ERODE, 'Erodes away the boundaries of foreground object'],
    'DILATE': [cv2.MORPH_DILATE, 'Increases the object area'],
    },
    'KSIZE':{
    'info':'kernel size for the morphological operations. Larger values will smooth out the contours more',
    },
    'CVH':{
    'info':'whether to use convex hulls of the contours. Reduces the complexity of the contours',
    'Convex Hull':[True, 'whether to use convex hulls of the contours. Reduces the complexity of the contours'],
    },
    'ACTION':{
    'info':'Select the action when clicking on a point in the image',
    'Add Luminosity':[1, 'Add a luminosity sample at the clicked point'],
    'Remove Luminosity':[-1, 'Remove a luminosity sample closest to the clicked point'],
    'Add IDPX':[2, 'Add an ID pixel at the clicked point'],
    'Remove IDPX':[-2, 'Remove an ID pixel closest to the clicked point'],
    'None':[0,'']
    },
    'CLOSE':{
    'info':'Close the viewer'
    },
    'SAVE':{
    'info':'Save the selected contour, either to the current fits file or to a new file if a path has been provided.'
    },
    'RESET':{
    'info':'Reset the viewer pixel selections'
    },
    'FSCRN':{
    'info':'Toggle fullscreen mode'
}}
bgs ={
    'main':'#000',
    'sidebar':'#fff',
    'legend':'#000',


}
def get_bg(ax):
    return bgs.get(ax.get_label(), bgs.get('sidebar', bgs.get('main', '#fff')))

def luminosity_viewer(fits_dir: fits.HDUList,saveloc=None,show_tooltips=True, morphex=(cv2.MORPH_CLOSE,cv2.MORPH_OPEN), fcmode=cv2.RETR_EXTERNAL, fcmethod=cv2.CHAIN_APPROX_SIMPLE, cvh=False, ksize=5, ):
    global G_fits_obj
    G_fits_obj = fits.open(fits_dir)
    from matplotlib import font_manager 
    if saveloc is None:
        saveloc = fits_dir
    ensure_dir(saveloc)
    font_manager.fontManager.addfont(fpath('python/jarvis/resources/FiraCodeNerdFont-Regular.ttf'))
    plt.rcParams['font.family'] = 'FiraCode Nerd Font'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['toolbar'] = 'None'

    # generate a stripped down, grey scale image of the fits file, and make normalised imagempl.rcParams['toolbar'] = 'None'
    proc =process_fits_file(prepare_fits(G_fits_obj, fixed='LON', full=True))
    img = mk_stripped_polar(proc, cmap=cmr.neutral, ax_background='white', img_background='black') 
    normed = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # set up global variables needed for the click event and configuration options
    global imarea
    imarea = (np.pi*(normed.shape[0]/2)**2)/2
    global clicked_coords
    clicked_coords = []
    global id_pixels
    id_pixels = []
    global G_morphex
    G_morphex = [m for m in morphex]
    global G_fcmode
    G_fcmode = fcmode
    global G_fcmethod
    G_fcmethod = fcmethod
    global G_cvh
    G_cvh = cvh
    global G_ksize
    G_ksize = ksize
    global G_clicktype
    G_clicktype=0
    global G_path
    G_path = None
    def getareapct(area):
        num = area/imarea * 100
        if num < 0.01:
            return f'{num:.2e}%'
        return f'{num:.2f}%'
    
    # set up the figure and axes
    fig = plt.figure()
    fw = [[fig.canvas.width(), int(720*1.5)],[fig.canvas.height(), 720]]
    if any(i[0] < i[1] for i in fw):
        fig.canvas.manager.window.setGeometry(0,0,fw[0][1],fw[1][1])
    fig.set_size_inches(fw[0][1]/fig.dpi, fw[1][1]/fig.dpi)
    qapp = fig.canvas.manager.window
    qapp.setWindowTitle('Luminance Viewer')
    iconpath = fpath('python/jarvis/resources/aa_asC_icon.ico')
    qapp.setWindowIcon(QtGui.QIcon(iconpath))




    # gs: 1 row, 2 columns, splitting the figure into plot and options
    gs = fig.add_gridspec(1,2, wspace=0.05, hspace=0, width_ratios=[8,3], left=0, right=1, top=1, bottom=0)
    # mgs: 17 rows, 12 columns, to layout the options
    mgs = gs[1].subgridspec(11,12, wspace=0, hspace=0, height_ratios=[
                            0.5,  0.5,  0.5,  0.5,      
                            1, 1,      
                            0.3, 0.1,    
                            1.5,3,3]
    ) 
    # layout the axes in the gridspecs                  
    
    linfax = fig.add_subplot(mgs[0:2,:8], label='infotext')
    clax = fig.add_subplot(mgs[0,8:], label='CLOSE')
    sbax = fig.add_subplot(mgs[1,8:], label='SAVE')
    rbax = fig.add_subplot(mgs[2,8:], label='RESET')
    fsax = fig.add_subplot(mgs[3,8:], label='FSCRN')
    flax = fig.add_subplot(mgs[2:4,:8], label='ACTION', facecolor=bgs['sidebar'])
    retrax = fig.add_subplot(mgs[4,:8], label='RETR', facecolor=bgs['sidebar'])
    morphax = fig.add_subplot(mgs[4:6,8:], label='MORPH', facecolor=bgs['sidebar'])
    chainax = fig.add_subplot(mgs[5,:8], label='CHAIN', facecolor=bgs['sidebar'])
    ksizeax = fig.add_subplot(mgs[7,:8], label='KSIZE', facecolor=bgs['sidebar'])
    cvhax = fig.add_subplot(mgs[6:8,8:], label='CHAIN', facecolor=bgs['sidebar'])
    ksizesubax = fig.add_subplot(mgs[6,:8], label='KSIZE', facecolor=bgs['sidebar'])
    fitinfoax = fig.add_subplot(mgs[8,:], label='fitinfo', facecolor=bgs['sidebar'])
    lax = fig.add_subplot(mgs[-1,:], label='legend', facecolor=bgs['legend'])
    ax = fig.add_subplot(gs[0],label='main', zorder=11)
    if show_tooltips:
        tooltipax = fig.add_subplot(mgs[9,:], label='tooltip', facecolor='#000')
        tooltip = tooltipax.text(0.012, 0.98, '', fontsize=8, color='black', ha='left', va='top', wrap=True, bbox=dict(facecolor='white', alpha=0.5, lw=0), zorder=10)
        
    
    axs = [ax, lax, rbax, linfax,  retrax, morphax, chainax, flax, sbax, cvhax, ksizeax, fitinfoax, clax, ksizesubax, fsax]+([tooltipax] if show_tooltips else [])
    axgroups ={'buttons':[rbax, sbax, clax, fsax],'textboxes':[linfax, fitinfoax],'options':[retrax, morphax, chainax, cvhax, ksizeax, ksizesubax, flax],'legends':[lax]}
    global lastax
    lastax = None
    linfax.text(0.5, 0.5, 'Select at least 2 or more\nLuminosity Samples', fontsize=10, color='black', ha='center', va='center')    
    # final plot initialisation
    ksizeax.set_facecolor('#fff')
    for a in axs:
        a.xaxis.set_visible(False)
        a.yaxis.set_visible(False)
    fig.canvas.setWindowTitle('Luminance Viewer')
    fig.canvas.set_cursor(3)
    fig.set_facecolor('black')
    ax.imshow(normed, cmap=cmr.neutral, zorder=0)
    visit, tel, inst, filt, pln, hem,cml,doy  = *fitsheader(G_fits_obj, 'VISIT', 'TELESCOP', 'INSTRUME', 'FILTER', 'PLANET', 'HEMISPH', 'CML', 'DOY'), 
    dt = get_datetime(G_fits_obj)
    fitinfoax.text(0.04, 0.96, "FITS File Information" , fontsize=12, color='black', ha='left', va='top')
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
    ctxt = [['File',f'{pthparts[0]}'], *[[' ', f"{p}"] for p in pthparts[1:]],
                              ['Inst.',f'{inst} ({tel}, {filt} filter)'],
                              ['Obs.',f'visit {visit} of {pln} ({hem})'],
                              ['Date',f'{dt.strftime("%d/%m/%Y")}, day {doy}'],
                              ['Time',f'{dt.strftime("%H:%M:%S")}(ICRS)'],
                              ['CML',f'{cml:.4f}°']]
    tl=fitinfoax.table(cellText=ctxt,cellLoc='right', cellColours=[[bgs['sidebar'], bgs['sidebar']] for i in range(len(ctxt))], colWidths=[0.14,1], colLabels=None, rowLabels=None, colColours=None, rowColours=None, colLoc='center', rowLoc='center', loc='bottom left', bbox=[0.02, 0.02, 0.96, 0.8], zorder=0)
    tl.visible_edges = ''
    tl.auto_set_font_size(False)
    tl.set_fontsize(9)
    for key, cell in tl.get_celld().items():
        cell.set_linewidth(0)
        cell.PAD =0
        if key[0]<len(pthparts) and key[1] == 1:
            cell.set_fontsize(10-len(pthparts)+1)
    # main update function
    def update_fig(cl):
        global id_pixels
        global G_clicked_coords
        global G_morphex
        global G_fcmode
        global G_fcmethod
        global G_cvh
        global G_ksize
        global G_path
        for a in [ax, lax]:
            a.clear()
            a.set_facecolor(get_bg(a))
        ax.imshow(normed, cmap=cmr.neutral, zorder=0)
        if len (id_pixels) > 0:
            pxsc = [idpixels[0] for idpixels in id_pixels], [idpixels[1] for idpixels in id_pixels]
            ax.scatter(*pxsc, **_idpxkws)
        if len(clicked_coords) > 0:
            scpc = [c[1][0] for c in clicked_coords], [c[1][1] for c in clicked_coords]
            ax.scatter(*scpc, **_clickedkws)
        if len(clicked_coords) >1:
            lrange = [min(c[0] for c in clicked_coords), max(c[0] for c in clicked_coords)]
            mask = cv2.inRange(normed, lrange[0]*255, lrange[1]*255)
            # smooth out the mask to remove noise
            kernel = np.ones((G_ksize, G_ksize), np.uint8)  # Small kernel to smooth edges
            for morph in G_morphex:
                mask = cv2.morphologyEx(mask, morph, kernel)
            # find the contours of the mask
            contours, hierarchy = cv2.findContours(image=mask, mode=G_fcmode, method=G_fcmethod)
            if G_cvh: # Convex hull of the contours which reduces the complexity of the contours
                contours = [cv2.convexHull(cnt) for cnt in contours]
            sortedcs = sorted(contours, key=lambda x: cv2.contourArea(x))
            sortedcs.reverse()

            linfax.clear()
            linfax.text(0.5,0.75, f'lrange = ({lrange[0]:.3f}, {lrange[1]:.3f})', fontsize=10, color='black', ha='center', va='center')
            

            if len(id_pixels) > 0: # if an id_pixel is selected, find the contour that encloses it
                selected_contours = []
                other_contours = []
                for i,contour in enumerate(sortedcs):
                    if all([cv2.pointPolygonTest(contour, id_pixel, False) > 0 for id_pixel in id_pixels]):  # >0 means inside
                        selected_contours.append([i,contour])
                    else:
                        other_contours.append([i,contour])
                # if a contour is found, convert the contour points to polar coordinates and return them
                for (i, contour) in selected_contours:
                    ax.scatter(*contour.T,**_selectclinekws)
                    cpath = mpl.path.Path(contour.reshape(-1, 2))
                    ax.plot(*cpath.vertices.T, color=_selectclinekws['color'], lw=_selectclinekws['s'])
                    ax.text(contour[0][0][0], contour[0][0][1], f"{i}", **_selectctextkws)
                for (i, contour) in other_contours:
                    ax.scatter(*contour.T,**_otherclinekws)
                    cpath = mpl.path.Path(contour.reshape(-1, 2))
                    ax.plot(*cpath.vertices.T, color=_otherclinekws['color'], lw=_otherclinekws['s'])
                    ax.text(contour[0][0][0], contour[0][0][1], f"{i}", **_otherctextkws)
                linfax.text(0.5, 0.5, f'Contours: {len(selected_contours)}({len(other_contours)} invalid)', fontsize=10, color='black', ha='center', va='center')
                if len(selected_contours) >0:
                    linfax.text(0.5, 0.25, f'Area: {getareapct(cv2.contourArea(selected_contours[0][1]))}, N:{len(selected_contours[0][1])}', fontsize=10, color='black', ha='center', va='center')
                    G_path = selected_contours[0][1]
                selectedhandles = [mpl.lines.Line2D([0], [0], label=f'{i}, {getareapct(cv2.contourArea(sortedcs[i]))}', **_handles['selectedc']) for i in [c[0] for c in selected_contours]]
                otherhandles = [mpl.lines.Line2D([0], [0], label=f'{i}, {getareapct(cv2.contourArea(sortedcs[i]))}', **_handles['otherc']) for i in [c[0] for c in other_contours]]
                lax.legend(handles=selectedhandles+otherhandles, **_legendkws)
            else:
                for i,contour in enumerate(sortedcs):
                    ax.scatter(*contour.T,**_defclinekws)
                    cpath = mpl.path.Path(contour.reshape(-1, 2))
                    ax.plot(*cpath.vertices.T, color=_defclinekws['color'], lw=_defclinekws['s'])
                    ax.text(contour[0][0][0], contour[0][0][1], f"{i}", **_defctextkws)
                linfax.text(0.5, 0.5, f'Contours: {len(sortedcs)}', fontsize=10, color='black', ha='center', va='center')
                handles = [mpl.lines.Line2D([0], [0], label=f'{i}, {getareapct(cv2.contourArea(sortedcs[i]))}',**_handles['defc'] ) for i in range(len(sortedcs))]
                lax.legend(handles=handles, **_legendkws)
                lax.set_facecolor(get_bg(lax))
        else:
            linfax.text(0.5, 0.5, 'Select at least 2 or more\nLuminosity Samples', fontsize=10, color='black', ha='center', va='center')
            lax.set_facecolor(get_bg(ax))
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        
    # config gui elements
    #---- RESET button ----#
    breset = mpl.widgets.Button(rbax, "Reset      "+u'\uF0E2')
    
    def reset(event):
        global clicked_coords
        global id_pixels
        clicked_coords = []
        id_pixels = []
        update_fig(None)
    breset.on_clicked(reset)
    #---- SAVE button ----#
    bsave = mpl.widgets.Button(sbax, "Save       "+u'\uEB4B')
  
    def save(event):
        global G_path
        global G_fits_obj
        if G_path is not None:
            G_fits_obj = savecontour_tofits(fits_obj=G_fits_obj, contour=G_path)
            if saveloc is not None:
                G_fits_obj.writeto(saveloc, overwrite=True)
                tqdm.write(f"Saved to {saveloc}, restarting viewer with new data")
                plt.close()
                luminosity_viewer(saveloc, saveloc, G_morphex, G_fcmode, G_fcmethod
                                  , G_cvh, G_ksize)
            else:
                tqdm.write("Save Failed: No save location provided or found")
        else:
            tqdm.write("Save Failed: No contour selected to save")
    bsave.on_clicked(save)
    #---- RETR options ----#
    retrbts = mpl.widgets.RadioButtons(retrax, ['EXTERNAL', 'LIST', 'CCOMP', 'TREE'], active=0)
    def retrfunc(label):
        global G_fcmode
        G_fcmode = _cvtrans['RETR'][label][0]
    retrbts.on_clicked(retrfunc)
    #---- MORPH options ----#
    bopts = list(_cvtrans['MORPH'].keys() - ['info'])
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
    chainbts = mpl.widgets.RadioButtons(chainax, ['NONE', 'SIMPLE', 'TC89_L1', 'TC89_KCOS'], active=1)
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
        ksizeax.set_facecolor('#fff')
        G_ksize = int(val)
        fig.canvas.draw_idle()
        ksizesubax.clear()
        ksizesubax.text(0.1, 0.45, f'ksize: {G_ksize}', fontsize=10, color='black', ha='left', va='center')
        fig.canvas.draw()
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
    if show_tooltips:
        def on_hover(event):
            global lastax
            inax = event.inaxes
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
        hover_event = fig.canvas.mpl_connect('motion_notify_event', on_hover) #noqa: F841
    plt.show()
    if G_path is not None:
        return G_path
    else:
        return None
    

    