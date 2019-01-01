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
    """
    y = Gaussian(x, a1, b1, c1) + Gaussian(x, a2, b2, c2)
    return y

def Fit_Double_Gaussian(x, y, **curve_fit_kws):
    """
    Fit the data with a double-peaked Gaussian profile.
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

#def lineParameters_DG(fit_result, perc=20, tol=0.01, resolution=10, verbose=False):
#    """
#    Convert the best-fit results to the line parameters.
#
#    Parameters
#    ----------
#    fit_result: tuple
#        x : 1D array
#            The x axis of the spectrum.
#        resTb : Table
#            The result table of the busyfit function.
#    perc : float, default: 20
#        The percent of the line peak at which we calculate the line width and the line center.
#    tol : float, (0, 1), default: 0.01
#        The tolerance level for our calculations and sanity checks.
#    resolution : float, default: 10
#        The number of times to enhance the resolution of the model spectrum comparing to the data.
#    verbose : bool, default: False
#        If true, raise the warnings.
#
#    Returns
#    -------
#    parDict : dict
#        The physical parameters of the emission line.
#        cPerc : tuple, (cPerc1, cPerc2)
#            Left and right channels of the line profile at perc% of the peak flux.
#        xPerc : tuple, (xPerc1, xPerc2)
#            Left and right x-axis values of the line profile at perc% of the peak flux.
#        Cc : float
#            Line central channel calculated at perc% of the peak flux.
#        Xc : float
#            The center of x-axis value calculated at perc% of the peak flux.
#        Wline : float
#            The full width at perc% peak flux.
#        Fpeak : float
#            Line peak flux.
#        Fperc : float
#            Fpeak * perc / 100.
#        cPeak : tuple, (cp1, cp2)
#            The channels of the left and right peaks of the line profile.
#        xPeak : tuple, (xp1, xp2)
#            The x-axis values of the left and right peaks of the line profile.
#    """
#    x, popt = fit_result
#    x = np.linspace(x[0], x[-1], len(x)*resolution)
#    fprf = Double_Gaussian(x, *popt)
#    #-> Find the location of the peaks
#    yfunc = lambda x, p: -1. * Double_Gaussian(x, *p)
#    xp10 = popt[1]
#    try:
#        xp1 = minimize(yfunc, xp10, args=(popt,)).x[0]
#    except:
#        raise RuntimeError("Fail to locate the left peak!")
#    yp1 = -1. * yfunc(xp1, popt)
#    xp20 = popt[4]
#    try:
#        xp2 = minimize(yfunc, xp20, args=(popt,)).x[0]
#    except:
#        raise RuntimeError("Fail to locate the left peak!")
#    yp2 = -1. * yfunc(xp2, popt)
#    #-> Calculate the peak flux
#    xpList = [xp1, xp2]
#    ypList = [yp1, yp2]
#    xmax   = xpList[np.argmax(ypList)]
#    ymax   = np.max(ypList)
#    #-> Find the left and right perc%-maximum channels
#    yperc = ymax * perc / 100.
#    yfunc = lambda x, p: Double_Gaussian(x, *p) -  yperc
#    #--> Left channel
#    xfit1 = x[0]
#    xfit2 = xp1
#    #---> Try to avoid the error when the boundary is very sharp.
#    if np.sign(yfunc(xfit1, popt) * yfunc(xfit2, popt)) == 1:
#        xfit2 = xfit2 * (1. + tol)
#    xPerc1, fit_flag = brentq(yfunc, xfit1, xfit2, args=(popt,), full_output=True)
#    if not fit_flag.converged:
#        raise RuntimeError("Fail to locate the left {0:.2f}% maxima!".format(perc))
#    #--> Right channel
#    xfit1 = xp2
#    xfit2 = x[-1]
#    #---> Try to avoid the error when the boundary is very sharp.
#    if np.sign(yfunc(xfit1, popt) * yfunc(xfit2, popt)) == 1:
#        xfit1 = xfit1 * (1. - tol)
#    xPerc2, fit_flag = brentq(yfunc, xfit1, xfit2, args=(popt,), full_output=True)
#    if not fit_flag.converged:
#        raise RuntimeError("Fail to locate the right {0:.2f}% maxima!".format(perc))
#    xcPerc = (xPerc1 + xPerc2) / 2. # Center of the x axis based on perc% maxima
#    wPerc  = np.abs(xPerc2 - xPerc1) # Full width at perc% maxima
#    parDict = {
#        "xPerc": (xPerc1, xPerc2),
#        "Xc"   : xcPerc,
#        "Wline": wPerc,
#        "Fpeak": ymax,
#        "Fperc": yperc,
#        "xPeak": (xp1, xp2)
#    }
#    return parDict
