

import datetime
from math import e
import os
import matplotlib as mpl
from regex import F, P
from sympy import fft
from glob import glob
mpl.use('qtagg') # forces the use of the Qt5/6 backend, neccessary for pathfinder
from jarvis import fpath
from jarvis.extensions import pathfinder, extract_conf
from jarvis.cvis import generate_coadded_fits, generate_rollings
from jarvis.utils import ensure_dir, fits_from_glob, group_to_visit, hst_fpath_dict, hst_fpath_segdict, fitsheader, get_datetime, visit_to_group,rpath
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



# NOTE: we use this condition to run the code only if it is run as a script to avoid running it when imported, this is best practice.
if __name__ == '__main__': # __name__ is a special,file-unique variable that is set to '__main__' when the script is run as a script, and set to the name of the module when imported.
    gps = [1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    #remove 3 (broken), 20 (southern) #group Numbe
# loop to generate the coadded fits, identify the boundary, and calculate the power
    #fpaths = hst_fpath_segdict(2,False)
    fpaths = hst_fpath_dict(byvisit=True) 
    fsegs = hst_fpath_segdict(2,False)
    favgs = dict()
    avgsdir = fpath('datasets/HST/custom/rollavgs')
    coaddir = fpath('datasets/HST/custom/coadds')

    #! First generate coadds in each segment
    # copaths = []
    # ensure_dir(coaddir)
    # for visit,fitpaths in fsegs.items():
    #     path = coaddir+"/"+visit +".fits"
    #     fobj = generate_coadded_fits([fits.open(fp) for fp in fitpaths], saveto=path, kernel_params=(3,1), overwrite=True,indiv=False, coadded=True)
    #     fobj.close()
    #     copaths.append([visit, path])
    # #! then run pathfinder on each segment.
    # for [visit,copath] in copaths:
    #     ret = pathfinder(copath, steps=False)
    #     if ret[0]:
    #         print(f"Pathfinder succeeded on {copath=}")
    #     else:
    #         print(f"Pathfinder failed on {copath=}")
    #         copaths.remove(copath) 
    # #! generate rolling averages of the fits, in their respective groups, and then save and split them into their respective segments
    # pbar = tqdm(total=len(fpaths), desc='Generating coadded fits')
    avgpaths = dict()
    for k,f in fpaths.items():
    #     # from fsegs identify the "visit" key that contains the fits obj
    #     # then turn into the correct dir 
    #     fitsobjs = generate_rollings([fits.open(ff) for ff in f], kernel_params=(3,1),indiv=True, coadded=True)
        
        for i in range(len(f)):
    #         fobj = fitsobjs[i]
            initpath = f[i]
            visit = None
            for key, value in fsegs.items():
                if initpath in value:
                    visit = key
                    break
            avgpath = avgsdir+f"/{visit}/"+initpath.split("/")[-1].split("\\")[-1] 
            if visit in avgpaths:
                avgpaths[visit] = avgpaths[visit] + [avgpath]
            else:
                avgpaths[visit] = [avgpath]
    #             ensure_dir(avgsdir+f"/{visit}")
    #         fobj.writeto(avgpath, overwrite=True)
    #         fobj.close()
    #     pbar.update()
    # pbar.close()
    #! using each pathed coadded fit, run pathfinder with conf, in silent mode, and then run powercalc. 
    failed = []
    pbarr = tqdm(total=len(avgpaths.keys()), desc="powergen")
    outfile = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.txt'
    open(outfile, 'w').close()
    for visit,fitspaths in avgpaths.items():
                copath = coaddir+"/"+visit +".fits"
            
            
                fit =fits.open(copath)
                if 'BOUNDARY' in fit:
                    conf = extract_conf(fit['BOUNDARY'].header,ignore=('LUMXY'))
                    fit.close()
                    pbr = tqdm(total=len(fitspaths), desc=visit)
                    for j,f in enumerate(fitspaths):
                        pf = pathfinder(f, fixlrange=(max(conf['LMIN']-0.2,0.05),conf['LMAX']), steps=False,show=False,conf=conf)
                        path = pf[1]
                        if pf[0]:
                            pc = powercalc(fits.open(f),path, writeto=outfile)
                            pbr.set_postfix_str(f"P={pc[0]:.3f}GW,I={pc[1]*1e8:.3f}x10⁻⁸GW/km²")
                        else:
                            failed.append([f,copath,path]) # this is a list of paths that failed pathfinder.
                        pbr.update()
                        fit.close()
                    pbr.close()
                else:
                    fit.close()
                    tqdm.write(f'No boundary found for {visit=!s}')
                pbarr.update()
    pbarr.close()
    for file in failed:
        print(f"failed on: {rpath(file[0])} using {rpath(file[1])} config: {file[2]}")
    print(f"{len(failed)=}")
        # except KeyError:
        #    fit.close()
        #    print(f'No boundary found for {visit=}')
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

    # pbar = tqdm(total=len(fpaths), desc='Generating coadded fits')
    # for k,f in fpaths.items():
    #     fitsobjs = generate_rollings([fits.open(ff) for ff in f], kernel_params=(3,1),indiv=True, coadded=True)
    #     ensure_dir(savedir+f'/{k}')
    #     for i,ff in enumerate(fitsobjs):
    #         ff.writeto(fpath(savedir+f'/{k}/{get_datetime(ff).strftime("%Y_%m_%dT%H_%M_%S")}window3_.fits'), overwrite=True)
    #     pbar.update()
    # pbar.close()
## loop to run pathfinder on segments of the fits
    # lrange = (0.25,0.32)

    # for g in fpaths.keys():
    #     #if int(g[1:3]) < 7 or if g=='g07sA':
    #         copath =fpath(f'datasets/HST/custom/{g}_[3,1]gaussian-coadded.fits')
    #         if os.path.isfile(copath):
    #             pt = pathfinder(copath, fixlrange=lrange)
## loop to run powercalc on segments of the fits
    # for g,fs in fpaths.items():
    #     copath = fpath(f'datasets/HST/custom/{g}_[3,1]gaussian-coadded.fits')
    #     if os.path.isfile(copath):
    #         fit = fits.open(copath)
    #         try:
    #             cpath = np.array(fit['BOUNDARY'].data.tolist())
                
    #             pbr = tqdm(total=len(fs), desc=f'"powercalc"({g})')
    #             for f in fs:
    #                 ff = fits.open(f)
    #                 pc = powercalc(ff, cpath, writeto=outfile)
    #                 pbr.set_postfix_str(f"P={pc[0]:.3f}GW,I={pc[1]*1e8:.3f}x10⁻⁸GW/km²")
    #                 pbr.update()
    #                 ff.close()
    #             pbr.close()
    #         except KeyError:
    #             fit.close()
    #             print(f'No boundary found for {g}')








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