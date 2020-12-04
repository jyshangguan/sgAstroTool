import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.isophote import (EllipseGeometry, Ellipse, Isophote, IsophoteList)
from photutils.isophote.sample import EllipseSample, CentralEllipseSample
from photutils.isophote.fitter import CentralEllipseFitter
from astropy import units as u
from astropy.visualization import (MinMaxInterval, LinearStretch, SqrtStretch,
                                   LogStretch, AsinhStretch, ImageNormalize)

__all__ = ['convert_output_string2float', 'get_model_from_header',
           'fit_ellipse', 'fit_isophote', 'wcs_pixel_scale', 'plot_image',
           'convert_unit_hst2astropy', 'image_moment', 'flux2mag']

def plot_image(image, wcs=None, stretch='asinh', units='arcsec', vmin=None,
               vmax=None, a=None, ax=None, plain=False, **kwargs):
    '''
    Plot an image.

    Parameters
    ----------
    image : 2D array
        The image to be ploted.
    wcs (optional) : Astropy.wcs
        WCS of the image.
    stretch : string (default: 'asinh')
        Choice of stretch: asinh, linear, sqrt, log.
    units : string (default: 'arcsec')
        Units of pixel scale.
    vmin (optional) : float
        Minimal value of imshow.
    vmax (optional) : float
        Maximal value of imshow.
    a (optional) : float
        Scale factor of some stretch function.
    ax (optional) : matplotlib.Axis
        Axis to plot the image.
    plain : bool (default: False)
        If False, tune the image.
    **kwargs : Additional parameters goes into plt.imshow()

    Returns
    -------
    ax : matplotlib.Axis
        Axis to plot the image.
    '''
    if wcs is None:
        extent = None
        units = 'pixel'
    else:
        cdelt1, cdelt2 = wcs_pixel_scale(wcs)
        nrow, ncol = image.shape
        x_len = ncol * cdelt1.to(units).value
        y_len = nrow * cdelt2.to(units).value
        extent = (-x_len/2, x_len/2, -y_len/2, y_len/2)
    stretchDict = {
        'linear': LinearStretch,
        'sqrt': SqrtStretch,
        'log': LogStretch,
        'asinh': AsinhStretch,
    }
    if a is None:
        stretch_use = stretchDict[stretch]()
    else:
        stretch_use = stretchDict[stretch](a=a)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch_use)

    if ax is None:
        plt.figure(figsize=(7, 7))
        ax = plt.gca()

    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'Greys'
    if 'origin' not in kwargs:
        kwargs['origin'] = 'lower'
    if 'norm' not in kwargs:
        kwargs['norm'] = norm
    if 'extent' not in kwargs:
        kwargs['extent'] = extent
    ax.imshow(image, **kwargs)

    if plain is False:
        ax.minorticks_on()
        ax.set_aspect('equal', adjustable='box')
    return ax


def convert_unit_hst2astropy(unit):
    '''
    Convert the units from HST standard to astropy standard.
    '''
    unitDict = {
        'ELECTRONS': 'electron',
    }
    return unitDict[unit]


def convert_output_string2float(s):
    '''
    Convert the parameter of the header to float.

    Parameters
    ----------
    s : string
        Header information in string.

    Returns
    -------
    v : float
        Best-fit value.
    ve : float
        Fit uncertainty.
    vf : string
        Flag: free, contrained, or fixed.
    '''
    if s is None:
        v = None
        ve = None
        f = None
    elif '{' in s:
        v = float(s[1:-1])
        ve = None
        f = 'contrained'
    elif '[' in s:
        v = float(s[1:-1])
        ve = None
        f = 'fixed'
    elif '+/-' in s:
        v_list = s.split('+/-')
        v = float(v_list[0])
        ve = float(v_list[1])
        f = 'free'
    else:
        raise ValueError('Input cannot be understood: {0}'.format(s))
    return v, ve, f


def get_model_from_header(header, parlist=None, ncomponents_max=100):
    '''
    Get model information from the header.

    Parameters
    ----------
    header : FITS header
        The header of model extensions, which contains best-fit parameters.
    parlist (optional) : list
        List of parameter names.
    ncomponents_max : int
        Maximum number of parameters.

    Returns
    -------
    info_dict : dict
        Model information. Number of components and details of each components.
          *_e is the uncertainty.
          *_f is the flag.
    '''
    if parlist is None:
        parlist = ['XC', 'YC', 'MAG', 'RE', 'N', 'AR', 'PA']

    counter = 1
    info_dict = {}
    while True:
        comp_name = header.get('COMP_{0}'.format(counter), None)
        if comp_name is None:
            break
        info_dict['COMP_{0}'.format(counter)] = {'{0}_MODEL'.format(counter): comp_name}

        for loop, par in enumerate(parlist):
            vc, vce, vcf = convert_output_string2float(header.get('{0}_{1}'.format(counter, par), None))
            info_dict['COMP_{0}'.format(counter)]['{0}_{1}'.format(counter, par)] = vc
            info_dict['COMP_{0}'.format(counter)]['{0}_{1}_e'.format(counter, par)] = vce
            info_dict['COMP_{0}'.format(counter)]['{0}_{1}_f'.format(counter, par)] = vcf
        counter += 1
        if counter > ncomponents_max:
            break
    info_dict['N_components'] = counter - 1
    return info_dict


def fit_ellipse(image, x0, y0, sma, eps=0, pa=0, **kwargs):
    '''
    Fit the elliptical isophotal profile.

    Parameters
    ----------
    image : 2D array-like
        Image data to perform isophotal analysis.
    x0, y0 : float
        The center pixel coordinate of the ellipse.
    sma : float
        The semimajor axis of the ellipse in pixels.
    eps : ellipticity
        The ellipticity of the ellipse.
    pa : float
        The position angle (in radians) of the semimajor axis in
        relation to the postive x axis of the image array (rotating
        towards the positive y axis). Position angles are defined in the
        range :math:`0 < PA <= \\pi`. Avoid using as starting position
        angle of 0., since the fit algorithm may not work properly. When
        the ellipses are such that position angles are near either
        extreme of the range, noise can make the solution jump back and
        forth between successive isophotes, by amounts close to 180
        degrees.
    **kwargs : Additional parameters feed to ellipse.fit_image().

    Returns
    -------
    isolist : IsophoteList instance
        A list-like object of Isophote instances, sorted by increasing
        semimajor axis length.
    '''
    geometry = EllipseGeometry(x0=x0, y0=y0, sma=sma, eps=eps, pa=pa)
    ellipse = Ellipse(image, geometry)
    isolist = ellipse.fit_image(**kwargs)
    return isolist


def fit_isophote(image, isolist):
    '''
    Fit isophotes according to the input isolist.

    Parameters
    ----------
    image : 2D array-like
        Image data to perform isophotal analysis.
    isolist : IsophoteList
        Input isophotes.

    Returns
    -------
    isolist_out : IsophoteList
        New measurements.
    '''
    isolist_out = []
    for iso in isolist:
        if iso.sma > 0:
            sample = EllipseSample(image, iso.sma, geometry=iso.sample.geometry)
            sample.update(iso.sample.geometry.fix)
            isolist_out.append(Isophote(sample, 0, True, 4))
        else:
            sample = CentralEllipseSample(image, 0.0, geometry=iso.sample.geometry)
            fitter = CentralEllipseFitter(sample)
            isolist_out.append(fitter.fit())
    isolist_out = IsophoteList(isolist_out)
    return isolist_out


def fit_isophote_deprecated(image, isolist):
    '''
    Fit according to the input isolist.
    '''
    isolist_out = []
    for iso in isolist:
        if np.isclose(iso.sma, 0):
            continue
        iso.sample.image = image
        iso.sample.update(iso.sample.geometry.fix)
        code = (5 if iso.stop_code < 0 else iso.stop_code)
        isolist_out.append(Isophote(iso.sample, iso.niter, iso.valid, code))
    isolist_out = IsophoteList(isolist_out)
    return isolist_out


def wcs_pixel_scale(wcs):
    '''
    Get the pixel scale from wcs.

    Parameters
    ----------
    wcs : astropy.wcs
    '''
    cdmat = wcs.pixel_scale_matrix
    cdelt1 = np.sqrt(cdmat[0, 0]**2 + cdmat[1, 0]**2) * u.degree
    cdelt2 = np.sqrt(cdmat[0, 1]**2 + cdmat[1, 1]**2) * u.degree
    return cdelt1, cdelt2


def image_moment(image, nx, ny):
    '''
    Calculate the moment of an image.
    '''
    xx, yy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    mom = np.sum(xx**nx * yy**ny * image) / np.sum(image)
    return mom


def flux2mag(flux, flux_err=None, zeromag=0):
    '''
    Convert flux to magnitude.
    '''
    mag = -2.5 * np.log10(flux) + zeromag
    if flux_err is None:
        mag_err = None
    else:
        mag_err = 2.5 * np.log10(np.e) * flux_err / flux
    return mag, mag_err
