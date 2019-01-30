import numpy as np
from scipy.optimize import minimize, brentq

__all__ = ["lineParameters", "linewidth_nky"]

def lineParameters(func, fit_result, xp0_list, perc=20, tol=0.01, resolution=10,
                   verbose=False):
    """
    Convert the best-fit results to the line parameters.

    Parameters
    ----------
    func : function
        The model function.
    fit_result: tuple
        x : 1D array
            The x axis of the spectrum.
        popt : Table
            The best fit parameters of the function.
    xp0_list : list
        The list of initial guess of the left and right peak of the line profile,
        order sensitive, i.e., [xp0_left, xp0_right].
    perc : float, default: 20
        The percent of the line peak at which we calculate the line width and the
        line center.
    tol : float, (0, 1), default: 0.01
        The tolerance level for our calculations and sanity checks.
    resolution : float, default: 10
        The number of times to enhance the resolution of the model spectrum comparing
        to the data.
    verbose : bool, default: False
        If true, raise the warnings.

    Returns
    -------
    parDict : dict
        The physical parameters of the emission line.
        cPerc : tuple, (cPerc1, cPerc2)
            Left and right channels of the line profile at perc% of the peak flux.
        xPerc : tuple, (xPerc1, xPerc2)
            Left and right x-axis values of the line profile at perc% of the peak flux.
        Cc : float
            Line central channel calculated at perc% of the peak flux.
        Xc : float
            The center of x-axis value calculated at perc% of the peak flux.
        Wline : float
            The full width at perc% peak flux.
        Fpeak : float
            Line peak flux.
        Fperc : float
            Fpeak * perc / 100.
        cPeak : tuple, (cp1, cp2)
            The channels of the left and right peaks of the line profile.
        xPeak : tuple, (xp1, xp2)
            The x-axis values of the left and right peaks of the line profile.
    """
    x, popt = fit_result
    x = np.linspace(x[0], x[-1], len(x)*resolution)
    fprf = func(x, *popt)
    #-> Find the location of the peaks
    yfunc = lambda x, p: -1. * func(x, *p)
    xp10 = xp0_list[0] #popt[1]
    try:
        xp1 = minimize(yfunc, xp10, args=(popt,)).x[0]
    except:
        raise RuntimeError("Fail to locate the left peak!")
    yp1 = -1. * yfunc(xp1, popt)
    xp20 = xp0_list[1] #popt[4]
    try:
        xp2 = minimize(yfunc, xp20, args=(popt,)).x[0]
    except:
        raise RuntimeError("Fail to locate the left peak!")
    yp2 = -1. * yfunc(xp2, popt)
    #-> Calculate the peak flux
    assert xp1 <= xp2
    xpList = [xp1, xp2]
    ypList = [yp1, yp2]
    xmax   = xpList[np.argmax(ypList)]
    ymax   = np.max(ypList)
    #-> Find the left and right perc%-maximum channels
    deltX = np.abs(x[1] - x[0])
    yperc = ymax * perc / 100.
    yfunc = lambda x, p: func(x, *p) -  yperc
    #--> Left channel
    xfit1 = x[0]
    xfit2 = xp1
    #---> Try to avoid the error when the boundary is very sharp.
    while (np.sign(yfunc(xfit1, popt) * yfunc(xfit2, popt)) == 1) & (xfit2 <= xp2):
        xfit2 += deltX
    xPerc1, fit_flag = brentq(yfunc, xfit1, xfit2, args=(popt,), full_output=True)
    if not fit_flag.converged:
        raise RuntimeError("Fail to locate the left {0:.2f}% maxima!".format(perc))
    #--> Right channel
    xfit1 = xp2
    xfit2 = x[-1]
    #---> Try to avoid the error when the boundary is very sharp.
    while (np.sign(yfunc(xfit1, popt) * yfunc(xfit2, popt)) == 1) & (xfit1 >= xp1):
        xfit1 -= deltX
    xPerc2, fit_flag = brentq(yfunc, xfit1, xfit2, args=(popt,), full_output=True)
    if not fit_flag.converged:
        raise RuntimeError("Fail to locate the right {0:.2f}% maxima!".format(perc))
    xcPerc = (xPerc1 + xPerc2) / 2. # Center of the x axis based on perc% maxima
    wPerc  = np.abs(xPerc2 - xPerc1) # Full width at perc% maxima
    parDict = {
        "xPerc": (xPerc1, xPerc2),
        "Xc"   : xcPerc,
        "Wline": wPerc,
        "Fpeak": ymax,
        "Fperc": yperc,
        "xPeak": (xp1, xp2)
    }
    return parDict

from HI_CoGv5 import Spectra
def linewidth_nky(velocity, flux, resolution, scale=5, get_error=False, nmc=100,
                  nidx=None, **der_cg_kws):
    """
    Calculate the linewidth using Niankun's Curve of Growth method.

    Parameters
    ----------
    velocity : 1D array
        The velocity of the spectrum.
    flux : 1D array
        The flux of the spectrum.
    resolution : float
        The spectral resolution.  The units should be the same as velocity.
    scale : int
        The number of continuous channels to define signal detection.
    get_error : bool
        Calculate the error if True.
    nmc : int
        The number of Monte Carlo steps to calculate the error.
    nidx : int
        Number of consecutive points to recognize the start and end points of
        the emission line.

    Returns
    -------
    result : dict
        The linewidth results.  The errors are included if get_error is True.
    """
    sp = Spectra(velocity, flux, resolution)
    sigma, f_mean = sp.get_sigma()
    Vc, start_channel, end_channel = sp.get_startEndVc(resolution, scale, nidx=nidx)
    result = sp.GrowthCurve(res=resolution, nidx=nidx, **der_cg_kws)
    if get_error:
        error = sp.GrowthCurveE(res=resolution, n_times=nmc, nidx=nidx, **der_cg_kws)
        for kw in error.keys():
            result[kw] = error[kw]
    return result
