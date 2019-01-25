import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize, brentq


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

def Fit_Double_Gaussian(x, y, **curve_fit_kws):
    """
    Fit the data with a double-peaked Gaussian profile.

    Parameters
    ----------
    x : array like
        The variable of the data.
    y : array like
        The dependant variable of the data.
    **curve_fit_kws : (optional)
        Additional parameters for curve_fit.

    Returns
    -------
    popt : array
        The best-fit parameters.
    perr : array
        The errors of the best-fit parameters.
    """
    popt, pcov = curve_fit(Double_Gaussian, x, y, **curve_fit_kws)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

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
