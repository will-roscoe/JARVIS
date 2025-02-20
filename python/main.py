

from jarvis.utils import fpath
# from jarvis import make_gif, moind, fpath, gradmap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import glob
from jarvis.power import powercalc

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


if __name__ == "__main__":
    powercalc()
    # contours = pathtest()
    # savecontourpoints(contours, fpath(r"datasets/HST/custom/v04_coadded_gaussian[3_1].fits"))