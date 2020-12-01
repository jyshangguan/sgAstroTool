import numpy as np
from astropy.io import fits
from photutils.isophote import EllipseGeometry
from photutils.isophote import Ellipse

__all__ = ['convert_output_string2float', 'get_model_from_header',
           'fit_isophote']

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


def fit_isophote(image, x0, y0, sma, eps=0, pa=0, **kwargs):
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
    kwargs : Additional parameters feed to ellipse.fit_image().

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
