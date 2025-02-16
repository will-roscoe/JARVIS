from pathlib import Path
#! defines the project root directory (as the root of the gh repo) 
GHROOT = Path(__file__).parents[2]
#! if you move this file/folder, you need to change this line to match the new location. 
#! index matches the number of folders to go up from where THIS file is located: /[2] python/[1] jarvis/[0] const.py

DATADIR = GHROOT / 'datasets'
PYDIR = GHROOT / 'python'
PKGDIR = PYDIR / 'jarvis' 
FITSINDEX = 1 #defines the default target HDU within a fits file.

# plot_defaults = dict(
# figure = dict(figsize=(7,6)),
# moonfp = dict(
#     colkey=lambda i: (('gold','IO'), ('aquamarine','EUR'), ('w','GAN'))[i],
#     text= dict(fontsize=10,alpha=0.5,path_effects=[mpl_patheffects.withStroke(linewidth=1, foreground='black')],horizontalalignment='center', verticalalignment='center', fontweight='bold'),
#     p1 = dict(color='k', linestyle='-', lw=4),
#     p2 = dict( linestyle='-', lw=2.5)),
# cml = dict(plot=dict(color='r', linestyle='--', lw=1.2), text=dict(fontsize=11, color='r', horizontalalignment='center', verticalalignment='center', fontweight='bold')),



# )


# def plot_polar(fitsobj:fits.HDUList, ax:mpl.projections.polar.PolarAxes,**kwargs)-> mpl.projections.polar.PolarAxes:

 
#     shrink = 1 if full else 0.75 # size of the colorbar
#     possub = 1.05 if full else 1.03 if not fixed_lon else 1.02 # position in the y axis of the subtitle
#     poshem = 45 if full else 135 if any([not is_south, not fixed_lon]) else -135 #position of the "N/S" marker
#     ax.set_theta_zero_location("N")    #! 
#     ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(base=10)) #! 
#     ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: '{:.0f}°'.format(x))) # set radial labels
#     ax.yaxis.set_tick_params(labelcolor='white', ) # set radial labels color #! 
#     if fixed_lon:
#         if full:
#             ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=np.pi/2))
#             if is_south:    
#                 shift_t = lambda x: x # should be 180°, 90°, 0°, 270°
#             else:           
#                 shift_t = lambda x: 2*np.pi-x # should be 0°, 90°, 180°, 270°
#             ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: '{:.0f}°'.format(np.degrees(shift_t(x))%360)))
#         else:
#             ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=np.pi/4))
#         ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=2*np.pi/36)) # 
#     else:
#         # clockticks
#         ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(base=2*np.pi/8))
#         ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(base=2*np.pi/24))
#         ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(clock_format))    
#     if full:   
#         ax.set_rlabel_position(0)   #position of the radial labels 
#     else:
#         ax.set_thetalim([np.pi/2,3*np.pi/2]) 
#     ax.set_facecolor('k') #black background #! 
#     ax.set_rlim([0,rlim]) # max colat range #! 
#     ax.tick_params(axis='both',pad=2.)    # shift position of LT labels#! 
#     radials = np.arange(0,rlim,10,dtype='int')
#     ax.set_rgrids(radials)#, color='white')#! 
    
#     # Titles
#     t_ = dict(suptitle=f'Visit {fitsobj[1].header["VISIT"]} (DOY: {fitsobj[1].header["DOY"]}/{fitsobj[1].header["YEAR"]}, {get_datetime(fitsobj)})', 
#               title=f'{"Fixed LT. " if not fixed_lon else ""}Integration time={fitsobj[1].header["EXPT"]} s. CML: {np.round(cml, decimals=1)}°')
#     if 'title' in kwargs:
#         titlekw = kwargs.pop('title')
#         if not titlekw: # if title=False, do not print any title
#             t_.update(suptitle='', title='')
#         elif isinstance(titlekw,str): #if title='foo', make the title 'foo', keep automatic subtitle
#             t_.update(suptitle=titlekw)
#         elif isinstance(titlekw,(tuple, list)): #if title=('foo','bar'), make the title 'foo' and the subtitle 'bar'
#             t_.update(suptitle=titlekw[0], title=titlekw[1])
#     plt.suptitle(t_['suptitle'], y=0.99, fontsize=14) #one of the two titles for every plot#! 
#     plt.title(t_['title'],y=possub, fontsize=12) #! 
    

#     if not fixed_lon and full: # meridian line (0°)  
#         plt.text(np.radians(cml)+np.pi, 4+rlim, '0°', color='coral', fontsize=12,horizontalalignment='center', verticalalignment='bottom', fontweight='bold') #! 
#         ax.plot([np.radians(cml)+np.pi,np.radians(cml)+np.pi],[0, 180], color='coral', 
#         path_effects=[mpl_patheffects.withStroke(linewidth=1, foreground='black')], linestyle='-.', lw=1) #
#         prime meridian (longitude 0)#! 

#     #Actual plot and colorbar (change the vmin and vmax to play with the limits
#     #of the colorbars, recommended to enhance/saturate certain features)
#     if 'ticks' in kwargs:
#         ticks = kwargs.pop('ticks')
#     elif int(fitsobj[1].header['EXPT']) < 30:
#         ticks = [10.,40.,100.,200.,400.,800.,1500.]
#     else:
#         ticks = [10.,40.,100.,200.,400.,1000.,3000.]
#     cmap = kwargs.pop('cmap') if 'cmap' in kwargs else 'viridis'
#     norm = kwargs.pop('norm') if 'norm' in kwargs else mpl.colors.LogNorm(vmin=ticks[0], vmax=ticks[-1])
#     shrink = kwargs.pop('shrink') if 'shrink' in kwargs else shrink
#     pad = kwargs.pop('pad') if 'pad' in kwargs else 0.06 #noqa: F841
    






#     rho = np.linspace(0, 180, num=int(image_data.shape[0]))
#     theta = np.linspace(0, 2 * np.pi, num=image_data.shape[1])
#     if is_south:
#         rho = rho[::-1]
#     if fixed_lon: 
#         image_centred = image_data
#     else: 
#         image_centred = np.roll(image_data,int(cml-180.)*4,axis=1) #shifting the image to have CML pointing southwards in the image
#     im_flip = np.flip(image_centred,0) # reverse the image along the longitudinal (x, theta) axis
#     corte = im_flip[:(int((image_data.shape[0])/crop)),:] # cropping image, if crop=1, nothing changes
#     # plotting cml, only for lon
#     if fixed_lon:
#         if is_south:
#             corte = np.roll(corte,180*4,axis=1)
#     plt.pcolormesh(theta,rho[:(int((image_data.shape[0])/crop))],corte,norm=norm, cmap=cmap)#!5 <- Color of the plot

#     if 'draw_cbar' not in kwargs:
#         cbarkw = True
#     else:
#         cbarkw = kwargs.pop('draw_cbar')
#     if cbarkw:
#         cbar = plt.colorbar(ticks=ticks, shrink=shrink, pad=0.06)
#         cbar.ax.set_yticklabels([str(int(i)) for i in ticks])
#         cbar.ax.set_ylabel('Intensity [kR]', rotation=270.) #! 
#     #Grids (major and minor)
#     if 'draw_grid' in kwargs:
#         draw_grid = kwargs.pop('draw_grid')
#         if not draw_grid:
#             ax.grid(False, which='both')
#     else:
#         ax.grid(True, which='major', color='w', alpha=0.7, linestyle='-') #! 
#         plt.minorticks_on()
#         ax.grid(True, which='minor', color='w', alpha=0.5, linestyle='-') #! 
#         #stronger meridional lines for the 0, 90, 180, 270 degrees:
#         for i in range(0,4):
#             ax.plot([np.radians(i*90),np.radians(i*90)],[0, 180], 'w', lw=0.9) #! 
    
#     #print which hemisphere are we in:
#     ax.text(poshem, 1.3*rlim, str(fitsheader(fitsobj, 'HEMISPH')).capitalize(), fontsize=21, color='k', 
#              horizontalalignment='center', verticalalignment='center', fontweight='bold')   
 