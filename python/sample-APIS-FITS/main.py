from astropy.io import fits

#NOTE: Location of the FITS file (relative to active dir)
FPATH = r"datasets\sample-APIS-FITS\oez022t14_proc.fits"

#NOTE: HDUList object: structure containing data array and some header information
hdulist = fits.open(FPATH)

print(hdulist.info())