


from jarvis import fpath
from jarvis.extensions import pathfinder
from jarvis.cvis import gaussian_coadded_fits
from jarvis.utils import fits_from_glob, group_to_visit
<<<<<<< HEAD
from jarvis.stats import stats, correlate
from jarvis.power import powercalc
=======
>>>>>>> parent of 218edd5 (test)
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
<<<<<<< HEAD
    groups = [1,2,4] #remove 3 (broken), 20 (southern)
=======
    groups = range(1,21)
>>>>>>> parent of 218edd5 (test)
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
<<<<<<< HEAD
        pt=pathfinder(fit)
        #paths.append(pathfinder(fit))
=======
        paths.append(pathfinder(fit))
>>>>>>> parent of 218edd5 (test)
        f = fits.open(fit, 'append')
        f.info()
        print(*[fi for fi in f], sep='\n')
        print(*[f.fileinfo(i) for i in range(len(f))], sep='\n')
<<<<<<< HEAD
        powercalc(f,pt)
=======
    print(paths)

>>>>>>> parent of 218edd5 (test)


<<<<<<< HEAD
print(stats(table['PFlux'],mean=True,median=True,std=True))

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
=======
>>>>>>> parent of 218edd5 (test)










    # paths = {}
    # fitspaths = [f'datasets/HST/custom/v{i:0>2}_coadded_gaussian[3_1].fits' for i in range(1, 21)]
    # print(fitspaths)
    # p = fpath(r'datasets\HST\custom\v04_coadded_gaussian[3_1].fits')

    #pathfinder(p)
    # contours = pathtest()
    # savecontourpoints(contours, fpath(r"datasets/HST/custom/v04_coadded_gaussian[3_1].fits"))
