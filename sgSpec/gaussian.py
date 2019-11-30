from __future__ import division
import numpy as np
#from scipy.optimize import curve_fit
#from scipy.optimize import minimize, brentq

__all__ = ["Gaussian", "Double_Gaussian", "Gauss_Hermite", "Gaussian_DoublePeak"]

def Gaussian(x, a, b, c):
    """
    A Gaussian function.
    """
    y = a * np.exp(-0.5 * (x - b)**2 / c**2)
    return y

def Double_Gaussian(x, a1, b1, c1, a2, b2, c2):
    """
    A double Gaussian profile.

    Parameters
    ----------
    x : array like
        The variable of the data.
    a1, b1, c1 : float
        The amplitude, mean and standard deviation of the first Gaussian component.
    a2, b2, c2 : float
        The amplitude, mean and standard deviation of the first Gaussian component.
    """
    y = Gaussian(x, a1, b1, c1) + Gaussian(x, a2, b2, c2)
    return y

def Gauss_Hermite(x, a, b, c, h3=0., h4=0., z=0):
    """
    The Gauss-Hermite polynomial function.

    Parameters
    ----------
    x : array
        The variable.
    a : float
        The amplitude.
    b : float
        The mean.
    c : float
        The sigma.
    h3 : float
        The h3 parameter.
    h4 : float
        The h4 parameter.
    z : float
        The zero point.

    Returns
    -------
    res : array
        The Gauss-Hermite profile.
    """
    y = (x - b) / c
    cmp_h3 = 2. * y**3 - 3. * y
    cmp_h4 = 4. * y**4 - 12. * y**2 + 3
    exp = -0.5 * y**2 * (1 + h3 / 3**0.5 * cmp_h3 + h4 / 24**0.5 * cmp_h4)
    res = a * np.exp(exp) + z
    return res

def Gaussian_DoublePeak(x, ag, ac, v0, sigma, w):
    """
    The Gaussian Double Peak function, Eq. (A2) of Tiley et al. (2016MNRAS.461.3494T).

    Parameters
    ----------
    x : 1D array
        The variable of the function.  It should be monochromatically increasing.
    ag : float
        The peak flux of the two half-Gaussians.  Require ag > 0.
    ac : float
        The flux at the central velocity.  Require ac > 0.
    v0 : float
        The center of the profile.
    sigma : float
        The standard deviation of the half-Gaussian profile.  Require sigma > 0.
    w : float
        The half-width of the central parabola.  Require w > 0.

    Returns
    -------
    y : 1D array
        The Gaussian Double Peak profile.
    """
    assert (ag > 0) & (ac > 0) & (sigma > 0) & (w > 0)
    x = np.atleast_1d(x)
    #-> Left
    vc_l = v0 - w
    fltr_l = x < (vc_l)
    x_l = x[fltr_l]
    y_l = Gaussian(x_l, ag, vc_l, sigma)
    #-> Right
    vc_r = v0 + w
    fltr_r = x > vc_r
    x_r = x[fltr_r]
    y_r = Gaussian(x_r, ag, vc_r, sigma)
    #-> Center
    a = (ag - ac) / w**2.
    fltr_c = (x >= vc_l) & (x <= vc_r)
    x_c = x[fltr_c]
    y_c = ac + a * (x_c - v0)**2.
    y = np.concatenate([y_l, y_c, y_r])
    return y
