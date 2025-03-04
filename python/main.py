

import datetime
import os
import matplotlib as mpl
from sympy import fft
mpl.use('qtagg') # forces the use of the Qt5/6 backend, neccessary for pathfinder
from jarvis import fpath
from jarvis.extensions import pathfinder
from jarvis.cvis import generate_coadded_fits
from jarvis.utils import fits_from_glob, group_to_visit, hst_segmented_paths
from jarvis.power import powercalc
from jarvis.stats import stats, correlate #noqa: F401
from tqdm import tqdm
from astropy.io import fits
import numpy as np
import pandas as pd #noqa: F401
#norm = mpl.colors.Normalize(vmin=0, vmax=1000)

#cmap = (mpl.colors.ListedColormap(['red', 'green', 'blue', 'cyan'])
#        .with_extremes(under='black', over='white'))
#bounds = [150, 300, 450, 600, 750]
#norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            
# makes an image
#n = fpath(r'datasets\HST\v04\jup_16-140-20-48-59_0103_v04_stis_f25srf2_proj.fits')
#moind(n, 'temp',fixed='lon', full=True, regions=False,moonfp=True)#cmap=cmap,norm=norm

         
# makes an image
#n = fpath(r'datasets\HST\v04\jup_16-140-20-48-59_0103_v04_stis_f25srf2_proj.fits')
#fitsfile = fits.open(n)
#moind(fitsfile)

# makes a gif
#make_gif('datasets/HST/v04', dpi=300) #moonfp=True, remove_temp=False)

#hdu = fits.open(n)
#d = hdu[1].data
#print(d)


outfile = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt'
# NOTE: we use this condition to run the code only if it is run as a script to avoid running it when imported, this is best practice.
if __name__ == '__main__': # __name__ is a special,file-unique variable that is set to '__main__' when the script is run as a script, and set to the name of the module when imported.
    gps = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    #remove 3 (broken), 20 (southern) #group Numbe
# loop to generate the coadded fits, identify the boundary, and calculate the power
    # for i in tqdm(gps):
    #     basefitpath = fpath(f'datasets/HST/group_{i:0>2}') 
    #     fitsg = fits_from_glob(basefitpath)
    #     copath = fpath(f'datasets/HST/custom/g{i+1:0>2},v{group_to_visit(i+1):0>2}_[3,1]gaussian-coadded.fits')
    #     fit = generate_coadded_fits(fitsg, saveto=copath, gaussian=(3,1), overwrite=True,indiv=False, coadded=True)
    #     #fit.info()
    #     fit.close()
    #     pt = pathfinder(copath)
    #     fit = fits.open(copath)
    #     #fit.info()
    #     # print(*[fi for fi in f], sep='\n')
    #     #print(*[f.fileinfo(i) for i in range(len(f))], sep='\n')
    #     #f.info()
    #     try:
    #         path = np.array(fit['BOUNDARY'].data.tolist())
    #         fit.close()
    #         pbr = tqdm(total=len(fitsg), desc=f'"powercalc"(group {i} of {len(gps)})')
    #         for j,f in enumerate(fitsg):
    #             pc = powercalc(f,path)
    #             pbr.set_postfix_str(f"P={pc[0]:.3f}GW,I={pc[1]*1e8:.3f}x10⁻⁸GW/km²")
    #             pbr.update()
    #             f.close()
    #         pbr.close()
    #     except KeyError:
    #         fit.close()
    #         print(f'No boundary found for group {i}')
# loop to only generate the gaussians ----------------------------------------
#     for i in tqdm(gps):
#         basefitpath = fpath(f'datasets/HST/group_{i:0>2}') 
#         fitsg = fits_from_glob(basefitpath)
#         copath = fpath(f'datasets/HST/custom/g{i:0>2},v{group_to_visit(i):0>2}_[3,1]gaussian-coadded.fits')
#         fit = generate_coadded_fits(fitsg, saveto=copath, gaussian=(3,1), overwrite=True,indiv=False, coadded=True)
#         fit.close()
# # loop to only run pathfinder ------------------------------------------------
    # for i in tqdm(gps):
    #     copath = fpath(f'datasets/HST/custom/g{i:0>2},v{group_to_visit(i):0>2}_[3,1]gaussian-coadded.fits')
    #     pt = pathfinder(copath)
# loop to only run powercalc -------------------------------------------------
    # for i in tqdm(gps):
    #     basefitpath = fpath(f'datasets/HST/group_{i:0>2}') 
    #     fitsg = fits_from_glob(basefitpath)
    #     copath = fpath(f'datasets/HST/custom/g{i:0>2},v{group_to_visit(i):0>2}_[3,1]gaussian-coadded.fits')
    #     fit = fits.open(copath)
    #     try:
    #         path = np.array(fit['BOUNDARY'].data.tolist())
    #         fit.close()
    #         pbr = tqdm(total=len(fitsg), desc=f'"powercalc"(group {i} of {len(gps)})')
    #         for j,f in enumerate(fitsg):
    #             pc = powercalc(f,path, writeto=outfile)
    #             pbr.set_postfix_str(f"P={pc[0]:.3f}GW,I={pc[1]*1e8:.3f}x10⁻⁸GW/km²")
    #             pbr.update()
    #             f.close()
    #         pbr.close()
    #     except KeyError:
    #         fit.close()
    #         print(f'No boundary found for group {i}')
## loop to generate gaussians of segments of the fits
    fpaths = hst_segmented_paths(2,False)
    # pbar = tqdm(total=len(fpaths), desc='Generating coadded fits')
    # for g,f in fpaths.items():
    #         if any(f'g{grp:0>2}' in g for grp in gps):
    #             copath = fpath(f'datasets/HST/custom/{g}_[3,1]gaussian-coadded.fits')
    #             print(f)
    #             fit = generate_coadded_fits([fits.open(ff) for ff in f], saveto=copath, gaussian=(3,1), overwrite=True,indiv=True, coadded=True)
    #             fit.close()
    #         pbar.update()
    # pbar.close()
## loop to run pathfinder on segments of the fits
    # lrange = (0.25,0.32)

    # for g in fpaths.keys():
    #     #if int(g[1:3]) < 7 or if g=='g07sA':
    #         copath =fpath(f'datasets/HST/custom/{g}_[3,1]gaussian-coadded.fits')
    #         if os.path.isfile(copath):
    #             pt = pathfinder(copath, fixlrange=lrange)
## loop to run powercalc on segments of the fits
    for g,fs in fpaths.items():
        copath = fpath(f'datasets/HST/custom/{g}_[3,1]gaussian-coadded.fits')
        if os.path.isfile(copath):
            fit = fits.open(copath)
            try:
                cpath = np.array(fit['BOUNDARY'].data.tolist())
                
                pbr = tqdm(total=len(fs), desc=f'"powercalc"({g})')
                for f in fs:
                    ff = fits.open(f)
                    pc = powercalc(ff, cpath, writeto=outfile)
                    pbr.set_postfix_str(f"P={pc[0]:.3f}GW,I={pc[1]*1e8:.3f}x10⁻⁸GW/km²")
                    pbr.update()
                    ff.close()
                pbr.close()
            except KeyError:
                fit.close()
                print(f'No boundary found for {g}')








# statsfile = 'powers.txt'
# table = pd.read_csv(statsfile, header=0,sep=r'\s+')
# print(table.columns)
# print(table)
# print(table)
# print(stats(table['PFlux'],mean=True,median=True,std=True))

# statsfile_2 = (fpath(r'datasets\Solar_Wind.txt'))
# table_2 = pd.read_csv(statsfile_2, header=0,sep=r'\s+')
# table_2 = table_2.iloc[308:310]
# print(table_2)
# correlated = correlate(table['PFlux'], table_2['jup_sw_pdyn'])

# table.plot(x='Time', y='PFlux', title='Power Flux vs Time at Jupiter', xlabel='Time [h:m:s]', ylabel='PFlux [W/m^2]')
# plt.title("Variation in power flux")
# plt.xlabel("Time [h:m:s]")
# plt.ylabel("Power Flux [W/m^2]")
# plt.show()






filepath =fpath( "data/...")