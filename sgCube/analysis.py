from __future__ import division
from __future__ import print_function
from builtins import range
import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma, mad_std
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
from photutils import deblend_sources, detect_sources, detect_threshold  #, source_properties
from spectral_cube import Projection
from radio_beam import Beam
from sgSpec import rms_spectrum, median_spectrum, Gauss_Hermite
from scipy.optimize import curve_fit

def get_segmentation(data, snr_thrsh=3., npixels=5, kernel=None, deblend=False,
                     detect_threshold_param={}, gaussian2DParams={}):
    """
    Generate a mask for the image based on the image segmentation.

    Parameters
    ----------
    data : 2D array
        The image (moment 0 map) to generate the mask.
    snr_thrsh : float, default: 3.
        The signal-to-noise ratio per pixel above the background for which
        to consider a pixel as possibly being part of a source.
    npixels : float, default: 5.
        The number of connected pixels, each greater than threshold, that an
        object must have to be detected. npixels must be a positive integer.
    kernel : array-like (2D) or Kernel2D, optional
        The 2D array of the kernel used to filter the image before thresholding.
        Filtering the image will smooth the noise and maximize detectability of
        objects with a shape similar to the kernel.
    detect_threshold_param : dict, optional
        The parameters of detect_threshold(), except data and snr.
    gaussian2DParams : dict, optional
        The parameters to generate a 2D Gaussian kernel.
        FWHM : float, default: 2.
            The FWHM of the 2D Gaussian kernel.
        x_size : float, default: 3.
            The size in x axis of the kernel.
        y_size : float, default: 3.
            The size in y axis of the kernel.

    Returns
    -------
    segm : 2D array
        The image segmentation.

    Notes
    -----
    None.
    """
    #-> Determine the detection threshold for each pixel.
    threshold = detect_threshold(data, nsigma=snr_thrsh, **detect_threshold_param)
    #-> If the kernel is not specified, we use a Gaussian kernel.
    if kernel is None:
        nFWHM  = gaussian2DParams.get("FWHM", 2.0)
        x_size = gaussian2DParams.get("x_size", 3.0)
        y_size = gaussian2DParams.get("y_size", 3.0)
        sigma = nFWHM * gaussian_fwhm_to_sigma #Convert FWHM to sigma
        kernel = Gaussian2DKernel(sigma, x_size=x_size, y_size=y_size)
        kernel.normalize()
    #-> Generate the image segmentation.
    segm = detect_sources(data, threshold, npixels=npixels, kernel=kernel)
    if deblend:
        segm = deblend_sources(data, segm, npixels=npixels, kernel=kernel)
    return segm

def Mask_Segmentation(data, snr_thrsh=2., wcs=None, source_position=None, segkws={}):
    """
    Generate the mask using image segmentation to identify the source.

    Parameters
    ----------
    data : 2D array
        The image (moment 0 map) to generate the mask.
    snr_thrsh : float, default: 2.
        The threshold of the signal-to-noise ratio.
    wcs : wcs class (optional)
        The wcs of the image.
    source_position : SkyCoord (optional)
        The SkyCoord of the source.  If not provided, all the segments are masked.
    segkws : dict, default: {}
        The key words for the image segmentation function.

    Returns
    -------
    mask : 2D array
        The mask array derived from the m0 map.
    """
    segm = get_segmentation(data, snr_thrsh=snr_thrsh, **segkws)
    if source_position is None:
        mask = segm.data != 0
    else:
        assert not wcs is None # The wcs is necessary to convert the source position
        if wcs.naxis == 2:
            ra_pix, dec_pix = wcs.wcs_world2pix([[source_position.ra.deg, source_position.dec.deg]], 1)[0]
        elif wcs.naxis == 3:
            ra_pix, dec_pix, _ = wcs.wcs_world2pix([[source_position.ra.deg, source_position.dec.deg, 0]], 1)[0]
        else:
            raise RuntimeError('Cannot incorporate the wcs dimensions!')

        label = segm.data[int(ra_pix), int(dec_pix)]
        if label == 0:
            mask = np.zeros_like(segm.data, dtype=bool)
        else:
            mask = segm.data == label
    return mask

def mask_ellipse_single(pos, ellipse_center, ellipse_sa, ellipse_pa):
    """
    Generate a mask with a single ellipse.

    Parameters
    ----------
    pos : tuple
        The tuple of two meshgrids (X, Y).
    ellipse_center : tuple
        The coordinate of the center of the ellipse (x, y), units: pixel.
    ellipse_sa : tuple
        The semi-major axis and semi-minor axis of the ellipse, (a, b), units:
        pixel.
    ellipse_pa : float
        The position angle of the ellipse, units: radian.

    Returns
    -------
    mask : array
        The mask array with the same shape as both the meshgrid in pos.

    Notes
    -----
    None.
    """
    pi = np.pi
    cos_angle = np.cos(pi - ellipse_pa)
    sin_angle = np.sin(pi - ellipse_pa)
    xc = pos[0] - ellipse_center[0]
    yc = pos[1] - ellipse_center[1]
    xct = xc * cos_angle - yc * sin_angle
    yct = xc * sin_angle + yc * cos_angle
    rad_cc = np.sqrt((xct / ellipse_sa[0])**2. + (yct / ellipse_sa[1])**2.)
    mask = (rad_cc - 1.) <= 1e-10
    return mask

def Mask_Ellipse(props, image, growth=1.):
    """
    Generate a mask for the image based on the input source properties.

    Parameters
    ----------
    props : list
        The list of source properties obtained from the function source_properties().
    image : array
        The image data.
    growth : float
        The factor to increase the size of the mask from the source property.

    Returns
    -------
    mask : array
        The mask array with the same shape of the input image.

    Notes
    -----
    None.
    """
    ny, nx = image.shape
    meshX, meshY = np.meshgrid(np.arange(nx), np.arange(ny))
    mask = np.zeros_like(image, dtype=bool)
    for prop in props:
        center = (prop.xcentroid.value, prop.ycentroid.value)
        a = prop.semimajor_axis_sigma.value * growth
        b = prop.semiminor_axis_sigma.value * growth
        theta = prop.orientation.value
        mask_add = mask_ellipse_single([meshX, meshY], center, [a, b], theta)
        mask = np.logical_or(mask, mask_add)
    return mask

def Mask_Image_Ellipse(data, snr_thrsh=2., wcs=None, growth=4., source_position=None, segkws={}):
    """
    Generate the mask using image segmentation to identify the source.

    Parameters
    ----------
    data : 2D array
        The image (moment 0 map) to generate the mask.
    wcs : wcs class
        The wcs of the image.
    snr_thrsh : float, default: 2.
        The threshold of the signal-to-noise ratio.
    source_position : SkyCoord (optional)
        The SkyCoord of the source.  If not provided, all the segments are masked.
    segkws : dict, default: {}
        The key words for the image segmentation function.

    Returns
    -------
    mask : 2D array
        The mask array derived from the data map.
    """
    segm = get_segmentation(data, snr_thrsh=snr_thrsh, **segkws)
    if not source_position is None:
        assert not wcs is None # The wcs is necessary to convert the source position
        ra_pix, dec_pix = wcs.wcs_world2pix([[source_position.ra.deg, source_position.dec.deg]], 1)[0]
        label = segm.data[int(ra_pix), int(dec_pix)]
        segm.data[segm.data != label] = 0
    #-> Generate the properties of the detected sources.
    props = source_properties(data, segm)
    mask = Mask_Ellipse(props, data, growth)
    return mask

def Mask_Fix_Ellipse(shape, center, semi_axes, pa):
    """
    Generate a mask with given position and size.

    Parameters
    ----------
    shape : tuple (ny, nx)
        The shape of the spatial dimension.
    center : tuple (x, y)
        The center of the ellipse, units: pixel.
    semi_axes : tuple (a, b)
        The semimajor axis and semiminor axis, units: pixel.
    pa : float
        The position angle of the ellipse, units: radian.

    Returns
    -------
    mask : 2D array
        The mask with a random elliptical area of True.
    """
    ny, nx = shape
    x = np.arange(nx)
    y = np.arange(ny)
    xx, yy = np.meshgrid(x, y)
    mask = mask_ellipse_single((xx, yy), center, semi_axes, pa)
    return mask

def Mask_Random_Ellipse(shape, semi_axes, pa):
    """
    Generate a random mask with with random position and given size.

    Parameters
    ----------
    shape : tuple (ny, nx)
        The shape of the spatial dimension.
    semi_axes : tuple (a, b)
        The semimajor axis and semiminor axis, units: pixel.
    pa : float
        The position angle of the ellipse, units: radian.

    Returns
    -------
    mask : 2D array
        The mask with a random elliptical area of True.
    """
    ny, nx = shape
    center = random_pos_pix([ny, nx])
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    mask = mask_ellipse_single((xx, yy), center, semi_axes, pa)
    return mask

def Mask_Cube(mask_2d, cube):
    """
    Generate the mask of the spectral data cube from a 2D mask.  It just repeats
    the 2D mask along the spectral (0th) axis.

    Parameters
    ----------
    mask_2d : numpy 2D array
        The 2D mask.
    cube : SpectralCube
        The spectral cube.

    Returns
    -------
    mask : numpy 3D array
        The mask of the spectral cube
    """
    mask = np.repeat(mask_2d[..., np.newaxis], cube.shape[0], axis=2)
    mask = np.moveaxis(mask, 2, 0)
    return mask

def SkyRMS_pixel(data, mask=None, verbose=False):
    """
    Calculate the RMS of the pixels excluded from the masked region.
    """
    if mask is None:
        mask = np.zeros_like(data, dtype=bool)
    else:
        assert data.shape == mask.shape
    #rms = np.std(data[~mask])
    rms = mad_std(data[~mask])

    if verbose:
        print("The shape of the data is: {0}".format(data.shape))
        print("There are {0} pixels used!".format(np.sum(~mask)))
    return rms

def CubeRMS_pixel(cube, mask=None, nsample=10, channel_list=None):
    """
    Calculate the pixel rms of the individual channels of the data cube.  The
    function calculate the rms of the relevant channels and return the median of
    the sample rms's.

    Parameters
    ----------
    cube : SpectralCube
        The data cube.
    mask : 2D bool array
        The mask to exclude the pixels when calculating the rms of the data slice.
    nsample : float, default: 10
        The number of sampled data slice.
    channel_list : list of int (optional)
        The list of channel indices.  If given, the data slice with be selected
        from this list.

    Returns
    -------
    rms : Quantity
        The rms of the data cube calculated as the median of the rms of the sampled
        slices.  The unit follows the data cube.
    """
    if channel_list is None:
        channel_list = np.arange(cube.shape[0])
    rmsList = []
    sample_list = np.random.choice(channel_list, nsample)
    for i in sample_list:
        rmsList.append(SkyRMS_pixel(cube[i, :, :].value, mask))
    rms = np.median(rmsList) * cube.unit
    return rms

def beam_size(beam):
    """
    Calculate the beam size.

    Parameters
    ----------
    beam : Beam class
        The beam information which can be directly obtained from spectral_cube.

    Returns
    -------
    bsz : Quantity
        The beam size.
    """
    bma = beam.major
    bmi = beam.minor
    bsz = np.pi * bma * bmi / 4. / np.log(2.)
    return bsz

def beam2pix(header):
    """
    Calculate the beam to pixel conversion ratio, which is used to convert
    between Jy/beam and Jy.

    Parameters
    ----------
    header : fits header
        The header of the data.

    Returns
    -------
    b2p : float
        The ratio of the beam size to the pixel size.
    """
    bmaj = header["BMAJ"]
    bmin = header["BMIN"]
    bsiz = np.pi * bmaj * bmin / 4. / np.log(2.)
    pixs = np.abs(header["CDELT1"] * header["CDELT2"])
    b2p  = bsiz / pixs
    return b2p

def random_pos_pix(shape):
    """
    Propose a random position in terms of pixel coordinate.
    """
    x1 = np.random.randint(shape[0])
    x2 = np.random.randint(shape[1])
    return (x1, x2)

def sum_mask(data, mask):
    """
    Sum the value of the masked region.
    """
    return np.sum(data[mask])

def Photometry_Mask(data, mask, rms_iteration=20, iteration_max=2000,
                    tolerance=0.95, verbose=False, show_sample=False, mask_bkg=None):
    """
    Measure the flux and uncertainty of the masked region.

    Parameters
    ----------
    data : 2D array
        The image (moment 0 map) to generate the mask.
    mask : 2D array
        The mask array derived from the data map.  The region to measure is masked as True.
    rms_iteration : float, default: 20
        The number of sample to calculate the rms of the sky flux.
    iteration_max : float, default: 2000
        The maximum number of iteration.
    tolerance : float, default: 0.95
        Drop the sky sampling if the number useful pixels is below the tolerance level.
    verbose : bool, default: False
        Print more information if True.
    show_sample : bool, default: False
        Provide the map of sampled pixels.
    mask_bkg (optional) : 2D array
        The mask of the background source.  True for sky pixel, False for
        contaminated pixel.

    Returns
    -------
    flux : float
        The flux of the source of the masked pixels.
    rms : float
        The std of the sampled sky flux with similar number of pixels as the source.
    samp_pattern : 2D array (optional)
        The map of sampled pixels.
    """
    ny, nx = data.shape
    assert mask.shape == (ny, nx)
    if mask_bkg is None:
        mask_bkg = ~mask
    else:
        assert mask_bkg.shape == (ny, nx)
    #-> Sum the flux of the source
    flux = sum_mask(data, mask)
    if show_sample:
        samp_pattern = np.zeros_like(data)
        samp_pattern[mask] = 2
    #-> Calculate the RMS of the sky flux
    #--> The number of pixels should be the same as that of the source.
    npix_src = np.sum(mask)
    #--> The radius of the circular aperture to measure the sky flux.
    r_mask_sky = np.sqrt(npix_src / np.pi) + 0.1 # Add 0.1 to avoid the digital problem
    if verbose:
        print("The source has {0} pixels and the radius is {1:.2f}!".format(npix_src, r_mask_sky))
    #--> Sample the sky many times
    skyList = []
    counter = 0
    for loop in range(iteration_max):
        if counter >= rms_iteration:
            break
        mask_sky_org = Mask_Random_Ellipse((ny, nx), (r_mask_sky, r_mask_sky), 0) # Generate a circular mask
        mask_sky = mask_bkg & mask_sky_org # Throw away the pixels of the source
        npix_use = np.float(np.sum(mask_sky))
        if (npix_use / npix_src) > tolerance: # If there are enough useful pixels, we take the sampling
            flux_sky = sum_mask(data, mask_sky)
            if np.isnan(flux_sky):
                if verbose:
                    print("*The sampled flux ({0}) is nan!".format(pos))
                continue
            else:
                skyList.append(flux_sky)
            counter += 1
            if show_sample:
                samp_pattern[mask_sky] = 1
        elif verbose:
            print("*The sampled pixels ({0}) are {1}/{2}.".format(pos, npix_use, np.sum(mask_sky_org)))
    if len(skyList) < rms_iteration:
        raise RuntimeWarning("The sky sampling is not enough!")
    unct = np.std(skyList)
    if show_sample:
        return flux, unct, samp_pattern
    else:
        return flux, unct

def Spectrum_Mask(cube, mask):
    """
    Extract the spectrum of the data cube from the masked region.

    Parameters
    ----------
    cube : SpectralCube
        The data cube to extract the spectrum.
    mask : array_like
        The masked pixels are used to extract the spectrum.  If mask is 2D, it
        is applied for all the channels.  If mask is 3D, the masks of all the
        channels should be provided.

    Returns
    -------
    spc_x : 1D array
        The spectral x axis.  It could be frequency or velocity depending on the
        unit of cube.
    spc_f : 1D array
        The flux of the spectrum, units following cube.
    """
    if len(mask.shape) == 2:
        mask = Mask_Cube(mask, cube)
    elif len(mask.shape) == 3:
        assert cube.shape == mask.shape
    else:
        raise ValueError("The shape of the mask ({0}) is not correct!".format(mask.shape))
    cube_msk = cube.with_mask(mask)
    spc_f = cube_msk.sum(axis=(1,2))
    spc_x = cube_msk.spectral_axis
    return (spc_x, spc_f)

def Spectrum_Random(cube, nspec, semi_axes, pa, mask=None, tolerance=0.95,
                    maxiters=200):
    """
    Extract the spectra randomly from the data cube.
    """
    nz, ny, nx = cube.shape
    if mask is None:
        mask = np.ones([ny, nx], dtype=bool)
    else:
        assert (len(mask.shape) == 2) & (mask.shape[0] == ny) & (mask.shape[1] == nx)
    spcList = []
    for loop in range(maxiters):
        if len(spcList) >= nspec:
            break
        mask_apt = Mask_Random_Ellipse(shape=(ny, nx), semi_axes=semi_axes, pa=pa)
        mask_spc = mask & mask_apt
        if np.sum(mask_spc) / np.sum(mask_apt) < tolerance:
            continue
        spcList.append(Spectrum_Mask(cube, mask_spc))
    if len(spcList) < nspec:
        print("Reach the maximum iterations ({0}) but cannot get enough spectra.".format(maxiters))
    return spcList

def ReadMap(filename):
    """
    Read the 2D map  from fits file.

    Parameteres
    -----------
    filename : string
        The fits file name.

    Returns
    -------
    mom : Projection class
        The moment map, units: following the fits file.
    """
    hdulist = fits.open(filename)
    header = hdulist[0].header
    data = np.squeeze(hdulist[0].data)
    beam = Beam(header["BMAJ"]*u.deg, header["BMIN"]*u.deg, header["BPA"]*u.deg)
    mom = Projection(data, wcs=WCS(header, naxis=2), beam=beam, unit=header["BUNIT"])
    return mom

def GaussHermite(cube, mask, line_velrange=None, fit_velrange=None, p0_dict={},
                 use_mommaps=None, verbose=False):
    """
    Calculate the map of Gauss-Hermite velocity fields.

    Parameters
    ----------
    cube : SpectralCube
        The data cube to extract the spectrum.
    mask : array_like
        The masked pixels are used to perform the spectral fitting.
    line_velrange : list (optional)
        The start and end velocity of the spectral line, used to find the line-free
        channels to estimate the noise and baseline.
    fit_velrange : list (optional)
        The velocity range of the spectral used in the fitting.
    p0_dict : dict
        The dict of the initial guess of "a", "b", "c", "h3", and "h4".
    use_mommaps : list (optional)
        The list of moment 1 and moment 2 maps.  If provided, the initial guesses
        of "b" and "c" will be from moment 1 and 2 maps, unless nan is provided.
    verbose : bool
        Print auxiliary information, if True.

    Returns
    -------
    mapDict : dict
        The dict of the fitting results.  The maps of "a", "b", "c", "h3", and
        "h4" are provided.
    """
    #-> Prepare the fitting
    if line_velrange is None:
        line_velrange = [-300, 300]
    if fit_velrange is None:
        fit_velrange = [-500, 500]
    nspc, nrow, ncol = cube.shape
    wave = cube.spectral_axis.value
    mapList = ["a", "b", "c", "h3", "h4"]
    mapDict = {
        "a": np.zeros([nrow, ncol]),
        "b": np.zeros([nrow, ncol]),
        "c": np.zeros([nrow, ncol]),
        "h3": np.zeros([nrow, ncol]),
        "h4": np.zeros([nrow, ncol]),
    }
    p0a = p0_dict.get("p0a", None) # amplitude
    p0b = p0_dict.get("p0b", 0)  # velocity
    p0c = p0_dict.get("p0c", 50) # sigma
    p03 = p0_dict.get("p0h3", 0) # h3
    p04 = p0_dict.get("p0h4", 0) # h4
    p0z = p0_dict.get("p0z", None) # zero point
    if not p0z is None:
        mapDict["z"] = np.zeros([nrow, ncol])
    if use_mommaps is None:
        flag_mom = False
    else:
        m1, m2 = use_mommaps
        flag_mom = True
    for loop_r in range(nrow):
        for loop_c in range(ncol):
            if mask[loop_r, loop_c]:
                #-> Get the data ready
                spec = cube[:, loop_r, loop_c].value
                rms  = rms_spectrum(wave, spec, flag=line_velrange)
                if p0z is None:
                    p0z = median_spectrum(wave, spec, flag=line_velrange)
                unct = np.ones(nspc) * rms
                fltr = (wave > fit_velrange[0]) & (wave < fit_velrange[1])
                x = wave[fltr]
                y = (spec - p0z)[fltr]
                e = unct[fltr]
                #-> Get the initial guess
                if p0a is None:
                    p00 = np.max(y)
                else:
                    p00 = p0a
                if flag_mom:
                    p01 = m1[loop_r, loop_c].value
                    p02 = m2[loop_r, loop_c].value
                else:
                    p01 = p0b
                    p02 = p0c
                if np.isnan(p01):
                    p01 = p0b
                if np.isnan(p02):
                    p02 = p0c
                p_init = [p00, p01, p02, p03]
                try:
                    popt, pcov = curve_fit(Gauss_Hermite, x, y, p0=p_init, sigma=e)
                except:
                    if verbose:
                        print("1st step: Cannot fit at [{0}, {1}]".format(loop_r, loop_c))
                    for loop_k in range(5): # Provide the nan for spexals failed to fit
                        kw = mapList[loop_k]
                        mapDict[kw][loop_r, loop_c] = np.nan
                    continue
                p_init = [popt[0], popt[1], popt[2], popt[3], p04]
                try:
                    popt, pcov = curve_fit(Gauss_Hermite, x, y, p0=p_init, sigma=e)
                except:
                    if verbose:
                        print("2st step: Cannot fit at [{0}, {1}]".format(loop_r, loop_c))
                    for loop_k in range(5): # Provide the nan for spexals failed to fit
                        kw = mapList[loop_k]
                        mapDict[kw][loop_r, loop_c] = np.nan
                    continue
                for loop_k in range(5): # Fill the calculated values
                    kw = mapList[loop_k]
                    mapDict[kw][loop_r, loop_c] = popt[loop_k]
            else:
                for loop_k in range(5): # Provide the nan for spexals not fitted.
                    kw = mapList[loop_k]
                    mapDict[kw][loop_r, loop_c] = np.nan
    return mapDict
