from __future__ import division
from __future__ import print_function
from builtins import range
from astropy.table import hstack, Column
import numpy as np
from photutils import aperture_photometry, CircularAperture, CircularAnnulus

__all__ = ["Circular_Aperture", "Circular_Annulus", "Aperture_Photometry",
           "RingPosRandom", "RingPosRandom", "Sky_RMS", "CurveOfGrowth"]

def Circular_Aperture(image, x, y, r, coord="pix"):
    """
    Setup the aperture.
    """
    pixel_size = image.pixel_size
    if coord == "wcs":
        assert not pixel_size is None
        srcPstXY = image.wcs_world2pix([x], [y], 1)
        x_pix = srcPstXY[0][0]
        y_pix = srcPstXY[1][0]
        r_pix = r / pixel_size
    elif coord == "pix":
        x_pix = x
        y_pix = y
        r_pix = r
    aperture = CircularAperture((x_pix, y_pix), r=r_pix)
    return aperture

def Circular_Annulus(image, x, y, r_in, r_out, coord="pix"):
    """
    Setup the annulus.
    """
    pixel_size = image.pixel_size
    if coord == "wcs":
        assert not pixel_size is None
        srcPstXY = image.wcs_world2pix([x], [y], 1)
        x_pix = srcPstXY[0][0]
        y_pix = srcPstXY[1][0]
        r_in_pix = r_in / pixel_size
        r_out_pix = r_out / pixel_size
    elif coord == "pix":
        x_pix = x
        y_pix = y
        r_in_pix = r_in
        r_out_pix = r_out
    annulus = CircularAnnulus((x_pix, y_pix), r_in=r_in_pix, r_out=r_out_pix)
    return annulus

def Aperture_Photometry(image, aperture, annulus, mask=True):
    """
    Perform aperture photometry. Only measure the flux of the unmasked pixels
    within the aperture.

    Parameters
    ----------
    image : Image object
        The image data.
    aperture : Aperture object
        The aperture to extract the flux of the source.
    annulus : Aperture object
        The aperture to extract the flux of the sky.
    mask : bool or 2D array, default: True
        If True, use the mask of the input image; if False, do
        not use mask. Otherwise, the mask could be specified as
        an 2D array.

    Returns
    -------
    apResults : dict
        flux : float
            The flux of the source, extracted from the aperture.
        coverage : float
            The fraction of the total pixels in the aperture to calculate
            the flux.
        phot_table : Table
            The information of the aperture photometry.

    Notes
    -----
    None.
    """
    data = image.get_data()
    if mask is True:
        mask = image.get_Mask()
    elif mask is False:
        mask = None
    else:
        assert mask.shape == data.shape
    #-> Measure the flux of the source and sky
    rawflux_table = aperture_photometry(data, aperture, mask=mask)
    bkgflux_table = aperture_photometry(data, annulus, mask=mask)
    phot_table = hstack([rawflux_table, bkgflux_table], table_names=['raw', 'bkg'])
    #-> Calculate the unmasked area
    aperturesMaskedArea = aperture_photometry(mask.astype(int), aperture)['aperture_sum']
    annulusMaskArea = aperture_photometry(mask.astype(int), annulus)['aperture_sum']
    apertureArea = aperture.area - aperturesMaskedArea
    annulusArea = annulus.area - annulusMaskArea
    #-> Calculate the flux of the background within the aperture
    bkg_mean = phot_table['aperture_sum_bkg'] / annulusArea
    bkg_sum = bkg_mean * apertureArea
    phot_table.add_columns([Column(apertureArea, name='aperture_pixels'),
                            Column(annulusArea, name='annulus_pixels')])
    final_sum = phot_table['aperture_sum_raw'] - bkg_sum
    phot_table['aperture_sum_sub'] = final_sum
    flux_final = phot_table['aperture_sum_sub'][0]
    #-> The ratio of effective area and aperture area. It reflects how
    # reliable the measurement is.
    coverage   = apertureArea[0] / aperture.area
    apResults = {
        "flux": flux_final,
        "coverage": coverage,
        "aperture": aperture,
        "annulus": annulus,
        "phot_table": phot_table,
    }
    return apResults

def RingPosRandom(nSmpl, xBegin, yBegin, xSize, ySize):
    '''
    This function is to provide the positions randomly in a region.

    Parameters
    ----------
    nSmpl : int
    The number of position to sample the sky.
    centx : float
    The pixel x coordinate of the center.
    centy : float
    The pixel y coordinate of the center.
    ringRad : float
    The pixel radius of the ring.

    Returns
    -------
    pst : a list of tuple (x, y)
        The x and y are the coord.

    Notes
    -----
    None.
    '''
    pst = []
    xPost = xBegin + xSize * np.random.random(nSmpl)
    yPost = yBegin + ySize * np.random.random(nSmpl)
    for loop in range(nSmpl):
        pst.append( (xPost[loop], yPost[loop]) )
    return pst

def Sky_RMS(image, ap_radius, nsample, cv_thrsh=0.99, ntry_max=1000, QuietMode=True):
    '''
    This function do aperture photometry randomly on the sky to estimate the sky rms.

    Parameters
    ----------
    image : Image object
        The image data.
    ap_radius : list of float
        The list of radius [r1, r2, r3]: r1 is the aperture radius, r2 is the
        inner radius of the sky annulus and r3 is the outer radius of the sky
        annulus. The unit is arcsec.
    nsample : int
        The number of samples.
    cv_thrsh : float, default: 0.99
        The threshold of coverage to adopt a measurement.
    ntry_max : int, default: 1000
        The max number to try the aperture photometry measurements.

    Returns
    -------
    rmsResults : dict
        sky_rms : float
            The standard deviation of the sky sample.
        fluxList : float array
            The record of the effective measurements of the sky.
        aperList : list
            The list of apertures used.

    Notes
    -----
    None.
    '''
    data = image.get_data()
    pixel_size = image.pixel_size
    r_ap, r_in, r_out = np.array(ap_radius) / pixel_size # Convert to the pixel unit.
    #-> Measure the aperture flux of the sky around the source
    xStart = r_out #The annulus to measure the sky should be left.
    yStart = r_out
    xSize  = data.shape[1] - 2. * xStart
    ySize  = data.shape[0] - 2. * yStart
    fluxList = []
    aperList = []
    counter = 0
    while( (len(fluxList) < nsample) & (counter < ntry_max)):
        centX, centY = RingPosRandom(1, xStart, yStart, xSize, ySize)[0]
        positionRMS = [(centX, centY)]
        #apResult = AperturePhotometry_Pixel(image, positionRMS, apRadius=apRadius,
        #                                    mask=mask, QuietMode=True)
        aperture = Circular_Aperture(image, centX, centY, r_ap, "pix")
        annulus  = Circular_Annulus(image, centX, centY, r_in, r_out, "pix")
        apResult = Aperture_Photometry(image, aperture, annulus, mask=True)
        flux = apResult['flux']
        coverage = apResult['coverage']
        if coverage >= cv_thrsh:
            fluxList.append(flux)
            aperList.append(aperture)
        counter += 1
    fluxList = np.array(fluxList)
    if len(fluxList) > 0:
        sky_rms = np.std(fluxList)
    else:
        sky_rms = np.nan
    if not QuietMode:
        if counter == ntry_max:
            print("[Sky_RMS]: Try the maximum times ({0})!".format(ntry_max))
        print("[Sky_RMS]: The sky RMS is: {0:.3f}".format(sky_rms))
    rmsResults = {
        "sky_rms": sky_rms,
        "fluxList": fluxList,
        "aperList": aperList
    }
    return rmsResults

def CurveOfGrowth(image, pos, r_range, r_ann, nsample=10, step="log",
      coord="pix", mask=True):
    """
    Measure the curve of growth of the source.

    Parameters
    ----------
    image : Image object
        The image data.
    pos : list
        The position of the source, [x, y] with units of pixel coordinate or wcs,
        which should be consistent with "coord".
    r_range : list
        The range of aperture radius, [r_min, r_max].
    r_ann : list
        The inner and outer radius of the sky annulus.
    nsample : int
        The number of the aperture radii to sample the curve of growth.
    step : str
        The step to sample the aperture radii, "log" or "linear".
    coord : str
        The units of the position and radius, "pix" or "wcs".
    mask : bool
        The mask for aperture photometry.

    Returns
    -------
    cogResults : dict
        flux_list : array
            The flux measured directly from the data.
        r_ap_list : array
            The aperture radius.
        apert_list: list
            The list of aperture.

    Notes
    -----
    None.
    """
    x, y = pos
    r_ann_in, r_ann_out = r_ann
    if step == "log":
        stepFunc = np.logspace
        r_min, r_max = np.log10(r_range)
    elif step == "linear":
        stepFunc = np.linspace
        r_min, r_max = r_range
    r_apList = stepFunc(r_min, r_max, nsample)
    if r_apList[-1] > r_ann_in:
        raise ValueError("[CurveOfGrowth]: The aperture radius is larger than the sky annulus!")
    annulus = Circular_Annulus(image, x, y, r_ann_in, r_ann_out, coord=coord)
    fluxList = []
    aptList  = []
    for r_ap in r_apList:
        aperture = Circular_Aperture(image, x, y, r_ap, coord=coord)
        apResult = Aperture_Photometry(image, aperture, annulus, mask)
        fluxList.append(apResult["flux"])
        aptList.append(aperture)
    fluxList = np.array(fluxList)
    cogResults = {
        "flux_list": fluxList,
        "r_ap_list": r_apList,
        "apert_list": aptList,
        "annulus": annulus
    }
    return cogResults
