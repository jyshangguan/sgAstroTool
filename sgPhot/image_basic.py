from __future__ import division
from __future__ import print_function
from builtins import range
from builtins import object
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.table import Table, Column
from photutils.utils import make_random_cmap
from astropy.visualization import AsinhStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
#from photutils import aperture_photometry, CircularAperture, CircularAnnulus
from photutils import DAOStarFinder, deblend_sources, detect_sources, detect_threshold
from photutils import EllipticalAperture, source_properties
import warnings

__all__ = ["Background_Fit_Polynomial", "Segmentation_Remove_Circle",
           "Props_Remove_Circle", "Mask_Ellipse_Single", "Mask_Ellipse",
           "check_InsideCircle", "check_InsideEllipse", "Props_Refiner",
           "Image", "image_align_wcs"]

def Background_Fit_Polynomial(image, order=3, mask=None):
    """
    Fit the background with a polynomial function.

    Parameters
    ----------
    image : array_like
        The 2D image data.
    order : float
        The order of the polynomial used to fit the background.
    mask : array_like
        The 2D mask of the image.

    Returns
    -------
    background : array_like
        The background image of the input data.

    Notes
    -----
    None.
    """
    ny, nx = image.shape
    meshX, meshY = np.meshgrid(np.arange(nx), np.arange(ny))
    if not mask is None:
        assert image.shape == mask.shape
        fltr_good_pixel = np.logical_not(mask.flatten())
        X_flat = meshX.flatten()[fltr_good_pixel]
        Y_flat = meshY.flatten()[fltr_good_pixel]
        d_flat = image.flatten()[fltr_good_pixel]
    else:
        X_flat = meshX.flatten()
        Y_flat = meshY.flatten()
        d_flat = image.flatten()
    p_init = models.Polynomial2D(degree=order)
    fit_p = fitting.LevMarLSQFitter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = fit_p(p_init, X_flat, Y_flat, d_flat)
    background = p(meshX, meshY)
    return background

def Segmentation_Remove_Circle(image, segment_img, circle):
    """
    Remove the sources on the image segmentation within a given circular area.

    Parameters
    ----------
    image : array_like
        The 2D image data.
    segment_img : array_like
        The image segmentation.
    circle : tuple
        The parameter of the circular area, (x, y, r), center position and the radius.

    Returns
    -------
    segment_img : array_like
        The image segmentation with the labels in the circular area removed.

    Notes
    -----
    None.
    """
    props = source_properties(image, segment_img)
    tbl = props.to_table()
    x, y, r = circle
    dist = np.sqrt((tbl["xcentroid"].value - x)**2 + (tbl["ycentroid"].value - y)**2)
    fltr_dist = dist < r
    label_to_remove = tbl[fltr_dist]["id"].data
    segment_img.remove_labels(labels=label_to_remove)
    return segment_img

def Props_Remove_Circle(props, circle):
    """
    Remove the props from the input circular region.

    Parameters
    ----------
    props : list
        The list of source properties obtained from the function source_properties().
    circle : tuple
        The parameter of the circular area, (x, y, r), center position and the radius.

    Returns
    -------
    props_new : list
        The list of props with the circular region ignored.

    Notes
    -----
    Mainly used to ignore the target when generating the mask.
    """
    props_new = []
    center = (circle[0], circle[1])
    radius = circle[2]
    for prop in props:
        pos = (prop.xcentroid.value, prop.ycentroid.value)
        if check_InsideCircle(pos, center, radius):
            pass
        else:
            props_new.append(prop)
    return props_new

def Mask_Ellipse_Single(pos, ellipse_center, ellipse_sa, ellipse_pa):
    """
    Generate a mask with a single ellipse.

    Parameters
    ----------
    pos : list
        The list of two meshgrids [X, Y].
    ellipse_center : list
        The coordinate of the center of the ellipse [x, y].
    ellipse_sa : list
        The semi-major axis and semi-minor axis of the ellipse, (a, b).
    ellipse_pa : float
        The position angle of the ellipse.

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
    rad_cc = (xct / ellipse_sa[0])**2. + (yct / ellipse_sa[1])**2.
    mask = rad_cc <= 1
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
    mask = np.zeros_like(image)
    for prop in props:
        center = (prop.xcentroid.value, prop.ycentroid.value)
        a = prop.semimajor_axis_sigma.value * growth
        b = prop.semiminor_axis_sigma.value * growth
        theta = prop.orientation.value
        mask_add = Mask_Ellipse_Single([meshX, meshY], center, [a, b], theta)
        mask = np.logical_or(mask, mask_add)
    return mask

def check_InsideEllipse(pos, ellipse_center, ellipse_sa, ellipse_pa):
    """
    Check whether the position is inside an ellipse.

    Parameters
    ----------
    pos : list
        The list of two meshgrids [X, Y].
    ellipse_center : list
        The coordinate of the center of the ellipse [x, y].
    ellipse_sa : list
        The semi-major axis and semi-minor axis of the ellipse, (a, b).
    ellipse_pa : float
        The position angle of the ellipse.

    Returns
    -------
    flag : bool or array_like
        The flag whether the position is inside the ellipse.

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
    rad_cc = (xct / ellipse_sa[0])**2. + (yct / ellipse_sa[1])**2.
    flag = rad_cc <= 1
    return flag

def check_InsideCircle(pos, center, radius):
    """
    Check whether the position is within a circle.

    Parameters
    ----------
    pos : list
        The list of two meshgrids [X, Y].
    center : list
        The coordinate of the center of the ellipse [x, y].
    radius : float
        The radius of the circle.

    Returns
    -------
    flag : float
        The flag whether the position is inside the circle.

    Notes
    -----
    None.
    """
    xc = pos[0] - center[0]
    yc = pos[1] - center[1]
    rad_cc = np.sqrt(xc**2 + yc**2)
    flag = rad_cc < radius
    return flag

def Props_Refiner(image, props_l, props_h, growth, snr_thrsh, npixels_add=20,
      detail=False, QuietMode=False):
    """
    Use the props from high-threshold segmentation to refine that from low-threshold
    segmentation.

    Parameters
    ----------
    image : Image object
        The image data.
    props_l : list
        The list of source properties with lower threshold.
    props_h : list
        The list of source properties with higher threshold.
    growth : float
        The factor to scale the mask.
    snr_thrsh : float
        The lower threshold.
    npixels_add : int, default: 20
        The npixels used to generate the additional props. It should be large in
        order to avoid the unphysical detections.
    detail : bool
        If True, plot the detail of the props.
    QuietMode : bool
        If True, raise warnings.

    Returns
    -------
    props_new : list
        The list of new source properties.

    Notes
    -----
    The function is only for get_Mask_Smart().
    """
    data = image.get_data()
    sky_median = image.sky_median
    sky_std = image.sky_std
    if (sky_median is None) or (sky_std is None):
        if not QuietMode:
            warnings.warn("[Props_Refiner]: The sky median and std is calculated with sigma clip!")
        sky_mean, sky_median, sky_std = sigma_clipped_stats(data, sigma=3.0, iters=5)
    assert len(props_h) > 0
    assert len(props_l) > 0
    #-> Check whether the patches in props_l are mixing the small patches in props_h.
    cntrX_max = []
    cntrY_max = []
    #--> Generate the list of the positions of the props_h
    for prop in props_h:
        cntrX_max.append(prop.xcentroid.value)
        cntrY_max.append(prop.ycentroid.value)
    cntrX_max = np.array(cntrX_max)
    cntrY_max = np.array(cntrY_max)
    pos = [cntrX_max, cntrY_max]
    props_new = []
    #idxAddList = []
    idxDelList = []
    #--> Replace the props_l if there are more than 1 prop in props_h overlapping.
    for prop in props_l:
        center = (prop.xcentroid.value, prop.ycentroid.value)
        a = prop.semimajor_axis_sigma.value * growth
        b = prop.semiminor_axis_sigma.value * growth
        theta = prop.orientation.value
        insideFlag = check_InsideEllipse(pos, center, (a, b), theta)
        nInside = np.sum(insideFlag)
        if nInside == 1:
            idxDelList.append(np.where(insideFlag == True)[0][0])
            props_new.append(prop)
        elif nInside == 0:
            props_new.append(prop)
        else:
            pass
    #-> Remove the overlapping props
    idxAddList = np.arange(len(props_h))
    idxDelList = np.unique(idxDelList)
    idxAddList = list(set(idxAddList) - set(idxDelList))
    for idx in idxAddList:
        props_new.append(props_h[idx])
    #-> Make up some small patches.
    #--> Mask the bright source.
    mask = Mask_Ellipse(props_new, data, growth)
    npixel = np.sum(mask)
    data[mask] = sky_median + sky_std * randn(npixel)
    image_replace = Image(data, image.header, image.pixel_size)
    #--> Look for the fainter sources.
    segm_add = image_replace.get_Segmentation(snr_thrsh=snr_thrsh, npixels=npixels_add)
    props_add = source_properties(data, segm_add)
    for prop in props_add:
        props_new.append(prop)
    if detail:
        #-> Plot for debug
        print("snr_thrsh: {0}".format(snr_thrsh))
        vmin = sky_median - 2 * sky_std
        vmax = sky_median + 30 * sky_std
        ax = image.plot_Image(cmap="Greys", origin='lower', vmin=vmin, vmax=vmax)
        for prop in props_h:
            position = (prop.xcentroid.value, prop.ycentroid.value)
            a = prop.semimajor_axis_sigma.value * growth
            b = prop.semiminor_axis_sigma.value * growth
            theta = prop.orientation.value
            aperture = EllipticalAperture(position, a, b, theta=theta)
            aperture.plot(color='red', lw=1., alpha=0.5, ax=ax)
        for prop in props_l:
            position = (prop.xcentroid.value, prop.ycentroid.value)
            a = prop.semimajor_axis_sigma.value * growth
            b = prop.semiminor_axis_sigma.value * growth
            theta = prop.orientation.value
            aperture = EllipticalAperture(position, a, b, theta=theta)
            aperture.plot(color='green', linestyle="--", lw=1., alpha=0.5, ax=ax)
        for prop in props_new:
            position = (prop.xcentroid.value, prop.ycentroid.value)
            a = prop.semimajor_axis_sigma.value * growth
            b = prop.semiminor_axis_sigma.value * growth
            theta = prop.orientation.value
            aperture = EllipticalAperture(position, a, b, theta=theta)
            aperture.plot(color='blue', linestyle="-.", lw=1., alpha=0.5, ax=ax)
        for prop in props_add:
            position = (prop.xcentroid.value, prop.ycentroid.value)
            a = prop.semimajor_axis_sigma.value * growth
            b = prop.semiminor_axis_sigma.value * growth
            theta = prop.orientation.value
            aperture = EllipticalAperture(position, a, b, theta=theta)
            aperture.plot(color='cyan', linestyle=":", lw=1., alpha=0.5, ax=ax)
        plt.show()
    return props_new


class Image(object):
    """
    The class of an image.

    Parameters
    ----------
    data : numpy 2D array
        The 2D image data.
    hearder : astropy fits header
        The header of the image data.
    pixel_size : float, default: 1.
        The physical scale of the pixel, units: arcsec.
    mag_zero_point : float (optional)
        The zero point of the image.
    mag2flux_density : float (optional)
        The conversion factor from magnitude to flux density.
    psf_FWHM_arcsec : float (optional)
        The FWHM of the PSF of the image, units: arcsec.
    """
    def __init__(self, data, header, pixel_size=1, mag_zero_point=None,
                 mag2flux_density=None, psf_FWHM_arcsec=None):
        self.__data_org = data
        self.__data = data
        self.header = header
        self.wcs = WCS(header)
        self.pixel_size = pixel_size
        self.magzp = mag_zero_point
        self.mag2fd = mag2flux_density
        self.psf_FWHM_arcsec = psf_FWHM_arcsec
        self.flag_sky_subtraction = False
        self.background = None
        self.mask = None
        self.mask_growth = None
        self.segmentation = None
        self.mask_properties = None
        self.sky_std = None
        self.sky_median = None
        self.sky_mean = None
        self.source_table = None

    def get_Background(self, *args, **kwargs):
        """
        Measure the background with 2D polynomial function.

        Parameters
        ---------
        order : float
            The order of the polynomial used to fit the background.
        mask : array_like
            The 2D mask of the image.

        Returns
        -------
        background : array_like
            The background image of the input data.

        Notes
        -----
        None.
        """
        self.background = Background_Fit_Polynomial(self.__data, *args, **kwargs)
        return self.background

    def wcs_world2pix(self, *args, **kwargs):
        """
        Convert the wcs to pixel coordinates.
        """
        assert not self.wcs is None
        return self.wcs.wcs_world2pix(*args, **kwargs)

    def wcs_pix2world(self, *args, **kwargs):
        """
        Convert the pixel to wcs coordinates.
        """
        assert not self.wcs is None
        return self.wcs.wcs_pix2world(*args, **kwargs)

    def get_Mask(self):
        """
        Get the mask of the image.
        """
        return self.mask.copy()

    def update_Mask(self, mask_tuple):
        """
        Update the information of the mask.

        Parameters
        ----------
        mask_tuple : tuple
            mask : (2D) array_like
                The mask of the image.
            growth : float
                The scale to enlarge the mask ellipticals.
            props : list
                The list of source properties of the mask.

        Returns
        -------
        None

        Notes
        -----
        None.
        """
        mask, growth, props = mask_tuple
        assert self.__data.shape == mask.shape
        self.mask = mask
        self.mask_growth = growth
        self.mask_properties = props
        return None

    def sky_Subtraction(self, mask=True, redo=False, *args, **kwargs):
        """
        Measure the background and subtract it. It will only do once unless
        redo is specified.

        Parameters
        ----------
        redo : bool
            The toggle to force to do the sky subtraction.
        order : float
            The order of the polynomial used to fit the background.
        mask : array_like
            The 2D mask of the image.

        Returns
        -------
        background : array_like
            The background image of the input data.

        Notes
        -----
        None.
        """
        if mask is True:
            mask = self.mask
            if mask is None:
                warnings.warn("[sky_Subtraction]: The mask of the image is not available.")
        elif mask is False:
            mask = None
        else:
            pass # The mask is specified by user.
        #-> If redo is True, force to subtract the sky.
        if redo:
            self.flag_sky_subtraction = False
        #-> Be cautious if the sky is already subtracted.
        if self.flag_sky_subtraction:
            raise Warning("The sky is already subtracted before!")
        else:
            self.flag_sky_subtraction = True
            self.get_Background(mask=mask, *args, **kwargs)
            self.__data -= self.background
        return None

    def get_Segmentation(self, snr_thrsh=3., npixels=5, kernel=None, deblend=False,
          detect_threshold_param={}, gaussian2DParams={}, update_segmation=True):
        """
        Generate a mask for the image based on the image segmentation.

        Parameters
        ----------
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
        data = self.__data
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
        segm = detect_sources(data, threshold, npixels=npixels, filter_kernel=kernel)
        if deblend:
            segm = deblend_sources(data, segm, npixels=npixels, filter_kernel=kernel)
        if update_segmation:
            self.segmentation = segm
        return segm

    def get_Mask_Segmentation(self, growth=5., ignore_circ_list=[], mask_nan=True,
          update_mask=True):
        """
        Generate a mask for the image based on the image segmentation.

        Parameters
        ----------
        growth : float, default: 5.
            The factor to scale the size of the mask based on the source elliptical
            aperture.
        ignore_circ_list : list, optional
            The list of (x, y, r), tuple of position and radius of the circular regions
            where the segmentation will be removed.
        mask_nan : bool, default: True
            If True, mask the nan in the image.

        Returns
        -------
        mask : 2D array
            The generated mask.
        growth : float
            The scale to enlarge the mask ellipticals.
        props : list
            The list of source properties of the mask.

        Notes
        -----
        None.
        """
        data = self.__data
        segm = self.segmentation
        assert not segm is None
        segm = segm.copy() #Do not want to modify the segmentation.
        #-> Remove the labels in the regions need to be ignored.
        for circ in ignore_circ_list:
            segm = Segmentation_Remove_Circle(data, segm, circ)
        #-> Generate the properties of the detected sources.
        props = source_properties(data, segm)
        #-> Generate the mask.
        mask = Mask_Ellipse(props, data, growth)
        if mask_nan:
            mask = np.logical_or(mask, np.isnan(data))
        if update_mask:
            self.mask = mask
            self.mask_growth = growth
            self.mask_properties = props
        return (mask, growth, props)

    def get_Mask_Smart(self, growth=5., snr_thrsh=(3., 30.), nlevels=10, step="log",
          ignore_circ_list=[], mask_nan=True, npixels_add=20, update_mask=True,
          detail=False):
        """
        Get the deblended mask. The function calculate the segmentation at
        different levels of S/N threshold in order to deblend the source
        detection and generate the mask.

        Parameters
        ----------
        growth : float, default: 5
            The scale to enlarge the mask ellipticals.
        snr_thrsh : tuple
            The min and max of the threshold of segmentation, with the unit of
            sky sigma.
        nlevels : int, default: 10
            The number of levels to calculate the segmentations.
        step : str, default: "log"
            The type of spacing used in generating the snr_thrsh list, "log" or
            "linear".
        ignore_circ_list : list, optional
            The list of (x, y, r), tuple of position and radius of the circular
            regions where the mask is not generated.
        mask_nan : bool
            If True, mask the nan.
        npixels_add : int, default: 20
            The npixels used to generate the additional props. It should be
            large in order to avoid the unphysical detections.
        update_image : bool
            If True, update the mask of the image.
        detail : bool
            If True, plot the props of each level.

        Returns
        -------
        mask : (2D) array_like
            The mask of the image.
        growth : float
            The scale to enlarge the mask ellipticals.
        props : list
            The list of source properties of the mask.

        Notes
        -----
        The function is tested with ALLWISE images.
        """
        data = self.__data
        if step == "log":
            stepFunc = np.logspace
            snr_min, snr_max = np.log10(snr_thrsh)
        elif step == "linear":
            stepFunc = np.linspace
            snr_min, snr_max = snr_thrsh
        snr_thrshList = stepFunc(snr_min, snr_max, nlevels)
        propsList = []
        for snr in snr_thrshList:
            segm = self.get_Segmentation(snr_thrsh=snr, update_segmation=False)
            propsList.append(source_properties(data, segm))
        props = propsList[nlevels - 1]
        for loop in range(nlevels - 1):
            loop_l = nlevels - loop - 2
            props_l = propsList[loop_l]
            snr_thrsh_l = snr_thrshList[loop_l]
            if len(props) == 0:
                props = props_l
                continue
            props = Props_Refiner(self, props_l, props, growth=growth,
                      snr_thrsh=snr_thrsh_l, npixels_add=npixels_add, detail=detail)
        #-> Remove the labels in the regions need to be ignored.
        for circ in ignore_circ_list:
            props = Props_Remove_Circle(props, circ)
        #-> Generate the mask.
        mask = Mask_Ellipse(props, data, growth)
        if mask_nan:
            mask = np.logical_or(mask, np.isnan(data))
        if update_mask:
            self.mask = mask
            self.mask_growth = growth
            self.mask_properties = props
        return (mask, growth, props)

    def sky_Median(self):
        """
        Calculate the median of the sky.
        """
        if self.mask is None:
            warnings.warn("No source mask is applied!")
            data_sky = self.__data
        else:
            data_sky = self.__data[np.logical_not(self.mask)]
        self.sky_median = np.median(data_sky)
        return self.sky_median

    def sky_STD(self):
        """
        Calculate the standard deviation of the sky.
        """
        if self.mask is None:
            warnings.warn("No source mask is applied!")
            data_sky = self.__data
        else:
            data_sky = self.__data[np.logical_not(self.mask)]
        self.sky_std = np.std(data_sky)
        return self.sky_std

    def sky_Mean(self):
        """
        Calculate the mean of the sky.
        """
        if self.mask is None:
            warnings.warn("No source mask is applied!")
            data_sky = self.__data
        else:
            data_sky = self.__data[np.logical_not(self.mask)]
        self.sky_mean = np.mean(data_sky)
        return self.sky_mean

    def get_Sky_Statistics(self):
        """
        Get the mean, median, and standard deviation of the sky.
        """
        if self.sky_mean is None:
            self.sky_mean = self.sky_Mean()
        if self.sky_median is None:
            self.sky_median = self.sky_Median()
        if self.sky_std is None:
            self.sky_std = self.sky_STD()
        return (self.sky_mean, self.sky_median, self.sky_std)

    def plot_Image(self, ax=None, vmin=None, vmax=None, stretch=None,
          w_mask=False, w_source=False, mask_lw=1.5, mask_alpha=0.5,
          src_alpha=0.5, src_radius=3, src_color="y", **kwargs):
        """
        Plot the image.

        Parameters
        ----------
        ax : Figure axis
            The axis handle of the figure.
        vmin : float
            The minimum scale of the image.
        vmax : float
            The maximum scale of the image.
        stretch : stretch object
            The stretch used to normalize the image color scale.
        w_mask : bool
            If True, plot the elliptical mask on the image.
        w_source : bool
            If True, plot the detected sources on the image.
        mask_lw : float
            The line width of the mask.
        mask_alpha : float
            The transparency of the mask.
        **kwargs : float
            The parameters of imshow() except the image and norm.

        Returns
        -------
        ax : Figure axis
            The handle of the image axis.

        Notes
        -----
        None.
        """
        if ax is None:
            fig = plt.figure(figsize=(7, 7))
            ax  = plt.gca()
        sky_median = self.sky_median
        sky_std    = self.sky_std
        assert not sky_median is None
        assert not sky_std is None
        if vmin is None:
            vmin = sky_median - 2 * sky_std
        if vmax is None:
            vmax = sky_median + 5 * sky_std
        if stretch is None:
            stretch = AsinhStretch
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch())
        ax.imshow(self.__data, norm=norm, **kwargs)
        if w_mask:
            assert not self.mask is None
            growth = self.mask_growth
            for prop in self.mask_properties:
                position = (prop.xcentroid.value, prop.ycentroid.value)
                a = prop.semimajor_axis_sigma.value * growth
                b = prop.semiminor_axis_sigma.value * growth
                theta = prop.orientation.value
                aperture = EllipticalAperture(position, a, b, theta=theta)
                aperture.plot(color='blue', lw=mask_lw, alpha=mask_alpha, ax=ax)
        if w_source:
            tbl = self.source_table
            assert not tbl is None
            xlist = tbl["xcentroid"]
            ylist = tbl["ycentroid"]
            for loop in range(len(tbl)):
                cir = plt.Circle((xlist[loop], ylist[loop]), radius=src_radius,
                                 color=src_color, fill=False, alpha=src_alpha)
                ax.add_patch(cir)
        return ax

    def plot_Mask(self, ax=None, **kwargs):
        """
        Plot the mask.

        Parameters
        ----------
        ax : Figure Axis
            The handle of the input figure axis.
        **kwargs : other parameters for imshow.

        Returns
        -------
        ax : Figure Axis
            The output handle of the figure axis.

        Notes
        -----
        None.
        """
        if ax is None:
            fig = plt.figure(figsize=(7, 7))
            ax  = plt.gca()
        assert not self.mask is None
        ax.imshow(self.mask, **kwargs)
        return ax

    def plot_Segmentation(self, ax=None):
        """
        Plot the image segmentation.

        Parameters
        ----------
        ax : Figure Axis
            The handle of the input figure axis.

        Returns
        -------
        ax : Figure Axis
            The output handle of the figure axis.

        Notes
        -----
        None.
        """
        segm = self.segmentation
        assert not segm is None
        if ax is None:
            ax  = plt.gca()
        rand_cmap = make_random_cmap(segm.max + 1, random_state=12345)
        ax.imshow(segm, origin='lower', cmap=rand_cmap)
        return ax

    def get_data(self):
        """
        Get the image data.
        """
        return self.__data.copy()

    def get_data_org(self):
        """
        Get the original image data.
        """
        return self.__data_org.copy()

    def get_SourceTable_Segmentation(self, *args, **kwargs):
        """
        Get the source table from the segmentation.

        Parameters
        ----------
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
        source_table : Table
            The table of detected source.

        Notes
        -----
        None.
        """
        segm = self.get_Segmentation(*args, **kwargs)
        #-> Generate the properties of the detected sources.
        props = source_properties(self.__data, segm)
        tbl = props.to_table()
        self.source_table = Table([tbl["id"], tbl["xcentroid"], tbl["ycentroid"]],
                                  names=["id", "xcentroid", "ycentroid"])
        return self.source_table

    def get_SourceTable_MaskProps(self):
        """
        Get the source table from the mask properties.

        Returns
        -------
        source_table : Table
            The table of detected source.

        Notes
        -----
        None.
        """
        props = self.mask_properties
        tbl = props.to_table()
        self.source_table = Table([tbl["id"], tbl["xcentroid"], tbl["ycentroid"]],
                                  names=["id", "xcentroid", "ycentroid"])
        return self.source_table

    def get_SourceTable_DAOfind(self, fwhm, threshold):
        """
        Get the source table from the DAOStarFinder.

        Parameters
        ----------
        fwhm : float
            The fwhm of the PSF, unit: arcsec.
        threshold : float
            The SNR threshold, unit: sky_std.

        Returns
        -------
        source_table : Table
            The table of detected source.

        Notes
        -----
        None.
        """
        fwhm_pix = fwhm / self.pixel_size
        sky_std = self.sky_std
        assert not sky_std is None
        daofind = DAOStarFinder(fwhm=fwhm_pix, threshold=threshold * sky_std)
        tbl = daofind(self.__data)
        self.source_table = Table([tbl["id"], tbl["xcentroid"], tbl["ycentroid"]],
                                 names=["id", "xcentroid", "ycentroid"])
        return self.source_table

    def find_Source(self, ra, dec, radius):
        """
        Get the source list from the specified position and radius.

        Parameters
        ----------
        ra : (1D) array_like or float
            The ra in degree.
        dec : (1D) array_like or float
            The dec in degree.
        radius : float
            The radius to match, unit: arcsec.

        Returns
        -------
        tblList : list
            The list of tables of matched sources. Sorted from the nearest to
            the farthest objects.

        Notes
        -----
        None.
        """
        ra  = np.atleast_1d(ra)
        dec = np.atleast_1d(dec)
        tbl = self.source_table
        assert len(ra) == len(dec)
        assert not tbl is None
        assert not self.wcs is None
        assert not self.pixel_size is None
        #-> Convert from wcs to pixel coordinates.
        rad_pix = radius / self.pixel_size
        srcPstXY = self.wcs.wcs_world2pix(ra, dec, 1)
        #-> Go through the position list to find all the matched sources.
        tblList = []
        for loop in range(len(ra)):
            srcX = srcPstXY[0][loop]
            srcY = srcPstXY[1][loop]
            dist = np.sqrt((tbl["xcentroid"] - srcX)**2 + (tbl["ycentroid"] - srcY)**2)
            fltr_dist = dist < rad_pix
            tbl_matched = tbl[fltr_dist]
            tbl_matched.add_column(Column(dist[fltr_dist], name="dist_pix", unit="pix"))
            tbl_matched.sort("dist_pix")
            tblList.append(tbl_matched)
        return tblList

    def write_to(self, filename, only_image=False, **kwargs):
        """
        Save the image and other informations.

        Parameters
        ----------
        filename : str
            The filename to save the image.
        only_image : bool
            If True, only save the processed image.
        **kwargs : other parameters of hduList.writeto()

        Returns
        -------
        None

        Notes
        -----
        None.
        """
        Hdr_pri = self.header.copy()
        Hdr_pri["skymean"] = self.sky_mean
        Hdr_pri["skymedian"] = self.sky_median
        Hdr_pri["skystd"] = self.sky_std
        HDU_pri = fits.PrimaryHDU(self.__data, header=Hdr_pri)
        hduList = fits.HDUList([HDU_pri])
        if only_image:
            hduList.writeto(filename, **kwargs)
            return 1
        nlayer = 2
        if not self.mask is None:
            HDU_msk = fits.ImageHDU(self.mask.astype(int))
            hduList.append(HDU_msk)
            Hdr_pri["History"] = "The mask is generated in layer {0}.".format(nlayer)
            nlayer += 1
        if self.flag_sky_subtraction:
            HDU_sky = fits.ImageHDU(self.background)
            hduList.append(HDU_sky)
            Hdr_pri["History"] = "The image is sky subtracted."
            Hdr_pri["History"] = "The background is in layer {0}.".format(nlayer)
            nlayer += 1
        HDU_org = fits.ImageHDU(self.__data_org)
        hduList.append(HDU_org)
        Hdr_pri["History"] = "The original image is in layer {0}.".format(nlayer)
        hduList.writeto(filename, **kwargs)
        return 0

def image_align_wcs(hdu):
    """
    Align the image according to the WCS so that x axis is RA and y axis is DEC.

    Note
    ----
    There seems some discussion that the simple rotation is not very accurate,
    however, that is likely higher order deviation.
    """
    from reproject import reproject_interp
    hdr = hdu.header
    img = hdu.data
    hdr_rot = hdr.copy()
    theta = hdr_rot.get("ORIENTAT", None)
    if theta is None:
        hdr_rot["CDELT1"] = -np.sqrt(hdr_rot["CD1_1"]**2 + hdr_rot["CD2_1"]**2)
        hdr_rot["CDELT2"] = np.sqrt(hdr_rot["CD2_2"]**2 + hdr_rot["CD1_2"]**2)
    else:
        theta = theta * np.pi / 180.
        hdr_rot["CDELT1"] = hdr_rot["CD1_1"] * np.cos(theta) - hdr_rot["CD2_1"] * np.sin(theta)
        hdr_rot["CDELT2"] = hdr_rot["CD2_2"] * np.cos(theta) + hdr_rot["CD1_2"] * np.sin(theta)
    hdr_rot["ORIENTAT"] = 0
    hdr_rot.remove("CD1_1")
    hdr_rot.remove("CD1_2")
    hdr_rot.remove("CD2_1")
    hdr_rot.remove("CD2_2")
    img_rot, footprint = reproject_interp(hdu, hdr_rot)
    return img_rot, hdr_rot
