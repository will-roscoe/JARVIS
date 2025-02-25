

import matplotlib as mpl
mpl.use('qtagg')
from jarvis import fpath
from jarvis.extensions import pathfinder
from jarvis.cvis import gaussian_coadded_fits
from jarvis.utils import fits_from_glob, group_to_visit
from jarvis.power import powercalc
from tqdm import tqdm
from astropy.io import fits
import numpy as np

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
    #remove 3 (broken), 20 (southern) #group Number
    for i in tqdm([1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]):
        basefitpath = fpath(f'datasets/HST/group_{i:0>2}') 
        fitsg = fits_from_glob(basefitpath)
        copath = fpath(f'datasets/HST/custom/g{i+1:0>2},v{group_to_visit(i+1):0>2}_[3,1]gaussian-coadded.fits')
        fit = gaussian_coadded_fits(fitsg, saveto=copath, gaussian=(3,1), overwrite=True,indiv=False, coadded=True)
        fit.info()
        fit.close()
        pt = pathfinder(copath)
        fit = fits.open(copath)
        fit.info()
        
        # print(*[fi for fi in f], sep='\n')
        #print(*[f.fileinfo(i) for i in range(len(f))], sep='\n')
        #f.info()
        
        
        fit.close()
        try:
            path = np.array(fit['BOUNDARY'].data.tolist())
            for f in fitsg:
                pc = powercalc(f,path)
                f.close()
        except IndexError:
            print(f'No boundary found for group {i}')

           

   





