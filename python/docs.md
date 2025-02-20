# JARVIS Documentation

## Submodules

### `jarvis.utils`

- **`ensure_dir(file_path)`**: 
  Ensures that the directory for the given file path exists, creating it if necessary.
  
  **Parameters:**
    - `file_path` (str): The path of the directory to ensure exists.

- **`fpath(x)`**: 
  Joins the given path with the project root directory.
  
  **Parameters:**
    - `x` (str): The path to join with the project root directory.
  
  **Returns:**
    - str: The joined path.

- **`rpath(x)`**: 
  Returns the relative path from the project root directory.
  
  **Parameters:**
    - `x` (str): The path to convert to a relative path.
  
  **Returns:**
    - str: The relative path.

- **`basename(x, ext=False)`**: 
  Returns the base name of the file path, optionally including the extension.
  
  **Parameters:**
    - `x` (str): The file path.
    - `ext` (bool, optional): Whether to include the extension. Default is False.
  
  **Returns:**
    - str: The base name of the file path.

- **`fits_from_glob(fits_dir, suffix='/*.fits', recursive=True, sort=True, names=False)`**: 
  Returns a list of FITS objects from a directory.
  
  **Parameters:**
    - `fits_dir` (str): The directory to search for FITS files.
    - `suffix` (str, optional): The suffix to match files. Default is '/*.fits'.
    - `recursive` (bool, optional): Whether to search directories recursively. Default is True.
    - `sort` (bool, optional): Whether to sort the files. Default is True.
    - `names` (bool, optional): Whether to return file names. Default is False.
  
  **Returns:**
    - List[fits.HDUList]: A list of FITS objects.
    - List[str]: A list of file names if `names` is True.

- **`fitsheader(fits_object, *args, ind=FITSINDEX, cust=True)`**: 
  Returns the header value of a FITS object.
  
  **Parameters:**
    - `fits_object` (fits.HDUList): The FITS object.
    - `*args` (str): List of header keys to return.
    - `ind` (int, optional): Index of the FITS object in the HDUList. Default is FITSINDEX.
    - `cust` (bool, optional): If True, returns custom values for 'south', 'fixed_lon', 'fixed'. Default is True.
  
  **Returns:**
    - Union[str, List[str]]: The header value(s).

- **`fits_from_parent(original_fits, new_data=None, **kwargs)`**: 
  Returns a new FITS object with the same header as the original, but with new data and/or header values.
  
  **Parameters:**
    - `original_fits` (fits.HDUList): The original FITS object.
    - `new_data` (np.ndarray, optional): The new data array. Default is None.
    - `**kwargs` (dict): Additional header values to update.
  
  **Returns:**
    - fits.HDUList: The new FITS object.

- **`get_datetime(fits_object)`**: 
  Returns a datetime object from the FITS header.
  
  **Parameters:**
    - `fits_object` (fits.HDUList): The FITS object.
  
  **Returns:**
    - datetime: The datetime object.

- **`prepare_fits(fits_obj, regions=False, moonfp=False, fixed='lon', rlim=40, full=True, crop=1, **kwargs)`**: 
  Returns a FITS object with specified header values for processing.
  
  **Parameters:**
    - `fits_obj` (fits.HDUList): The FITS object.
    - `regions` (bool, optional): Whether to include regions. Default is False.
    - `moonfp` (bool, optional): Whether to include moon footprints. Default is False.
    - `fixed` (str, optional): The fixed parameter. Default is 'lon'.
    - `rlim` (int, optional): The radial limit. Default is 40.
    - `full` (bool, optional): Whether to include the full image. Default is True.
    - `crop` (float, optional): The crop factor. Default is 1.
    - `**kwargs` (dict): Additional header values to update.
  
  **Returns:**
    - fits.HDUList: The prepared FITS object.

- **`make_filename(fits_obj)`**: 
  Returns a filename based on the FITS header values.
  
  **Parameters:**
    - `fits_obj` (fits.HDUList): The FITS object.
  
  **Returns:**
    - str: The generated filename.

- **`update_history(fits_object, *args)`**: 
  Updates the HISTORY field of the FITS header with the current time.
  
  **Parameters:**
    - `fits_object` (fits.HDUList): The FITS object.
    - `*args` (str): Additional history entries.
  
  **Returns:**
    - None

- **`debug_fitsheader(fits_obj)`**: 
  Prints the header of the FITS object.
  
  **Parameters:**
    - `fits_obj` (fits.HDUList): The FITS object.
  
  **Returns:**
    - None

- **`debug_fitsdata(fits_obj)`**: 
  Prints the data of the FITS object.
  
  **Parameters:**
    - `fits_obj` (fits.HDUList): The FITS object.
  
  **Returns:**
    - None

- **`mcolor_to_lum(*colors)`**: 
  Converts colors to luminance values.
  
  **Parameters:**
    - `*colors` (str): List of colors to convert.
  
  **Returns:**
    - Union[int, List[int]]: The luminance value(s).

- **`clock_format(x_rads, pos=None)`**: 
  Converts radians to clock format.
  
  **Parameters:**
    - `x_rads` (float): The angle in radians.
    - `pos` (int, optional): The position. Default is None.
  
  **Returns:**
    - str: The clock format.

- **`Jfits`**: 
  Class for handling FITS files with various methods for updating and writing data.
  
  **Attributes:**
    - `ind` (int): The index of the FITS object in the HDUList.
    - `loc` (str): The location of the FITS file.
    - `hdul` (fits.HDUList): The FITS object.
  
  **Methods:**
    - `data`: Returns the data array.
    - `header`: Returns the header.
    - `update(data=None, **kwargs)`: Updates the FITS object with new data and/or header values.
    - `writeto(path)`: Writes the FITS object to a file.
    - `close()`: Closes the FITS object.
    - `data_apply(func, *args, **kwargs)`: Applies a function to the data array.
    - `apply(func, *args, **kwargs)`: Applies a function to the FITS object.

- **`group_to_visit(*args)`**: 
  Converts group numbers to visit numbers.
  
  **Parameters:**
    - `*args` (int): List of group numbers.
  
  **Returns:**
    - Union[int, List[int]]: The visit number(s).

- **`visit_to_group(*args)`**: 
  Converts visit numbers to group numbers.
  
  **Parameters:**
    - `*args` (int): List of visit numbers.
  
  **Returns:**
    - Union[int, List[int]]: The group number(s).

- **`fitsdir(sortby='visit', full=False)`**: 
  Returns a list of FITS file paths sorted by visit number.
  
  **Parameters:**
    - `sortby` (str, optional): The sorting parameter. Default is 'visit'.
    - `full` (bool, optional): Whether to return the full file paths. Default is False.
  
  **Returns:**
    - List[str]: The list of FITS file paths.

### `jarvis.transforms`

- **`fullxy_to_polar(x, y, img, rlim=40)`**: 
  Transforms image coordinates to polar coordinates.
  
  **Parameters:**
    - `x` (float): The x-coordinate.
    - `y` (float): The y-coordinate.
    - `img` (np.ndarray): The image array.
    - `rlim` (int, optional): The radial limit. Default is 40.
  
  **Returns:**
    - tuple: The polar coordinates (colatitude, longitude).

- **`fullxy_to_polar_arr(xys, img, rlim=40)`**: 
  Transforms a list of image coordinates to polar coordinates.
  
  **Parameters:**
    - `xys` (List[tuple]): List of (x, y) coordinates.
    - `img` (np.ndarray): The image array.
    - `rlim` (int, optional): The radial limit. Default is 40.
  
  **Returns:**
    - np.ndarray: The array of polar coordinates.

- **`polar_to_fullxy(colat, lon, img, rlim)`**: 
  Transforms polar coordinates to image coordinates.
  
  **Parameters:**
    - `colat` (float): The colatitude.
    - `lon` (float): The longitude.
    - `img` (np.ndarray): The image array.
    - `rlim` (int): The radial limit.
  
  **Returns:**
    - np.ndarray: The image coordinates.

- **`gaussian_blur(input_arr, radius, amount, boundary='wrap', mode='same')`**: 
  Applies a Gaussian blur to the input array.
  
  **Parameters:**
    - `input_arr` (np.ndarray): The input array.
    - `radius` (int): The radius of the Gaussian kernel.
    - `amount` (float): The amount of blur.
    - `boundary` (str, optional): The boundary condition. Default is 'wrap'.
    - `mode` (str, optional): The convolution mode. Default is 'same'.
  
  **Returns:**
    - np.ndarray: The blurred array.

- **`gradmap(input_arr, kernel2d, boundary='wrap', mode='same')`**: 
  Returns the gradient of the input array using the kernel2d convolution kernel.
  
  **Parameters:**
    - `input_arr` (np.ndarray): The input array.
    - `kernel2d` (np.ndarray): The 2D convolution kernel.
    - `boundary` (str, optional): The boundary condition. Default is 'wrap'.
    - `mode` (str, optional): The convolution mode. Default is 'same'.
  
  **Returns:**
    - np.ndarray: The gradient map.

- **`dropthreshold(input_arr, threshold)`**: 
  Sets values below the threshold to 0 in the input array.
  
  **Parameters:**
    - `input_arr` (np.ndarray): The input array.
    - `threshold` (float): The threshold value.
  
  **Returns:**
    - np.ndarray: The thresholded array.

- **`coadd(input_arrs, weights=None)`**: 
  Coadds a list of arrays with optional weights.
  
  **Parameters:**
    - `input_arrs` (List[np.ndarray]): The list of input arrays.
    - `weights` (List[float], optional): The list of weights. Default is None.
  
  **Returns:**
    - np.ndarray: The coadded array.

- **`normalize(input_arr)`**: 
  Normalizes the input array to the range [0, 1].
  
  **Parameters:**
    - `input_arr` (np.ndarray): The input array.
  
  **Returns:**
    - np.ndarray: The normalized array.

- **`adaptive_coadd(input_fits, eff_ang=90)`**: 
  Coadds FITS objects adaptively based on their CML positions.
  
  **Parameters:**
    - `input_fits` (List[fits.HDUList]): The list of FITS objects.
    - `eff_ang` (int, optional): The effective angle. Default is 90.
  
  **Returns:**
    - fits.HDUList: The coadded FITS object.

- **`align_cmls(input_fits, primary_index)`**: 
  Aligns a list of FITS objects to a primary FITS object by rolling the images to match the CML positions.
  
  **Parameters:**
    - `input_fits` (List[fits.HDUList]): The list of FITS objects.
    - `primary_index` (int): The index of the primary FITS object.
  
  **Returns:**
    - List[fits.HDUList]: The list of aligned FITS objects.

### `jarvis.time_conversions`

- **`et2datetime(ettimes)`**: 
  Converts SPICE/ET time to Python datetime.
  
  **Parameters:**
    - `ettimes` (Union[float, List[float]]): The SPICE/ET time(s).
  
  **Returns:**
    - Union[datetime, List[datetime]]: The Python datetime object(s).

- **`datetime2et(pytimes)`**: 
  Converts Python datetime to SPICE/ET time.
  
  **Parameters:**
    - `pytimes` (Union[datetime, List[datetime]]): The Python datetime object(s).
  
  **Returns:**
    - Union[float, List[float]]: The SPICE/ET time(s).

- **`str2et(strs, fmt)`**: 
  Converts string to ET time.
  
  **Parameters:**
    - `strs` (Union[str, List[str]]): The string(s) to convert.
    - `fmt` (str): The datetime format.
  
  **Returns:**
    - Union[float, List[float]]: The ET time(s).

- **`et2str(et, fmt)`**: 
  Converts ET time to string.
  
  **Parameters:**
    - `et` (Union[float, List[float]]): The ET time(s).
    - `fmt` (str): The datetime format.
  
  **Returns:**
    - Union[str, List[str]]: The formatted string(s).

- **`et2doy2004(ettimes)`**: 
  Converts ET time to days after 2004-01-01.
  
  **Parameters:**
    - `ettimes` (Union[float, List[float]]): The ET time(s).
  
  **Returns:**
    - Union[float, List[float]]: The days after 2004-01-01.

- **`doy20042et(doy2004)`**: 
  Converts days after 2004-01-01 to ET time.
  
  **Parameters:**
    - `doy2004` (Union[float, List[float]]): The days after 2004-01-01.
  
  **Returns:**
    - Union[float, List[float]]: The ET time(s).

- **`datetime2doy2004(pytimes)`**: 
  Converts Python datetime to days after 2004-01-01.
  
  **Parameters:**
    - `pytimes` (Union[datetime, List[datetime]]): The Python datetime object(s).
  
  **Returns:**
    - Union[float, List[float]]: The days after 2004-01-01.

- **`doy20042datetime(doy2004)`**: 
  Converts days after 2004-01-01 to Python datetime.
  
  **Parameters:**
    - `doy2004` (Union[float, List[float]]): The days after 2004-01-01.
  
  **Returns:**
    - Union[datetime, List[datetime]]: The Python datetime object(s).

### `jarvis.stats`

- **`(module)`**: 
  Placeholder for statistical functions for analysis.

### `jarvis.reading_mfp`

- **`moonfploc(iolon, eulon, galon)`**: 
  Calculates the expected latitude and longitude of the four Galilean moons based on the Hess 2011 table and FITS file headers.
  
  **Parameters:**
    - `iolon` (float): The longitude of Io.
    - `eulon` (float): The longitude of Europa.
    - `galon` (float): The longitude of Ganymede.
  
  **Returns:**
    - tuple: The expected latitude and longitude of the four Galilean moons.

### `jarvis.power`

- **`powercalc(fits_obj, pat)`**: 
  Calculates the UV emission power from HST STIS FITS images, defining spatial regions and computing emission power.
  
  **Parameters:**
    - `fits_obj` (fits.HDUList): The FITS object.
    - `pat` (str): The pattern to match.
  
  **Returns:**
    - None

### `jarvis.polar`

- **`process_fits_file(fitsobj)`**: 
  Processes a FITS file for polar plotting.
  
  **Parameters:**
    - `fitsobj` (fits.HDUList): The FITS object.
  
  **Returns:**
    - fits.HDUList: The processed FITS object.

- **`plot_polar(fitsobj, ax, **kwargs)`**: 
  Plots a polar projection of the FITS object data.
  
  **Parameters:**
    - `fitsobj` (fits.HDUList): The FITS object.
    - `ax` (mpl.projections.polar.PolarAxes): The axis to plot on.
    - `**kwargs` (dict): Additional keyword arguments for customization.
  
  **Returns:**
    - mpl.projections.polar.PolarAxes: The axis with the plot.

- **`plot_moonfp(fitsobj, ax)`**: 
  Plots the footprints of moons on a polar plot.
  
  **Parameters:**
    - `fitsobj` (fits.HDUList): The FITS object.
    - `ax` (mpl.projections.polar.PolarAxes): The axis to plot on.
