

import matplotlib as mpl
mpl.use('qtagg')
from jarvis import fpath
from jarvis.extensions import pathfinder
from jarvis.cvis import gaussian_coadded_fits
from jarvis.utils import fits_from_glob, group_to_visit
from jarvis.power import powercalc
from tqdm import tqdm
from astropy.io import fits

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


if __name__ == '__main__':
    # # script to generate the coadded fits
    groups = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    basefitpath = [fpath(f'datasets/HST/group_{i:0>2}') for i in groups]
    nfits =[]
    _mpbar = tqdm(total=len(basefitpath), desc='Generating coadded fits')
    for i,fp in enumerate(basefitpath):
        _mpbar.set_description(f'Generating coadded fits for group {i+1:0>2}')
        fitsg = fits_from_glob(fp)
        p = fpath(f'datasets/HST/custom/g{i+1:0>2},v{group_to_visit(i+1):0>2}_[3,1]gaussian-coadded.fits')
        fit = gaussian_coadded_fits(fitsg, saveto=p, gaussian=(3,1), overwrite=True,indiv=False, coadded=True)
        _mpbar.update(1)
        nfits.append(p)
    _mpbar.close()
    # script to generate the contours 
   
    paths =[]
    for fit in nfits:
        pt=pathfinder(fit)
        #paths.append(pathfinder(fit))
        f = fits.open(fit, 'append')
        f.info()
        print(*[fi for fi in f], sep='\n')
        print(*[f.fileinfo(i) for i in range(len(f))], sep='\n')
        powercalc(f,pt)
    #print(paths)














    # paths = {}
    # fitspaths = [f'datasets/HST/custom/v{i:0>2}_coadded_gaussian[3_1].fits' for i in range(1, 21)]
    # print(fitspaths)
    # p = fpath(r'datasets\HST\custom\v04_coadded_gaussian[3_1].fits')

    #pathfinder(p)
    # contours = pathtest()
    # savecontourpoints(contours, fpath(r"datasets/HST/custom/v04_coadded_gaussian[3_1].fits"))
