import os
import subprocess
import numpy as np
from astropy.table import Table
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.optimize import minimize, brentq
from scipy.stats import mode
import matplotlib.pyplot as plt

__all__ = ["busyfit", "rms_spectrum", "median_spectrum", "busyProfile", "busyFitProfile",
           "busyResTb2List", "lineParameters_BF", "busyFitPlot", "lineParPlot"]

def busyfit(x, y, rms=None, p_init=None, relax=False, niter=None, verbose=False,
            channel=None, busyPath="/Users/shangguan/Softwares/BusyFit/"):
    """
    This is a wrapper of BusyFit.

    Parameters
    ----------
    x : 1D array
        The x axis of the spectrum.
    y : 1D array
        The y axis of the spectrum.
    rms : float (optional)
        The rms of the spectrum at the line-free channels.
    p_init : list (optional)
        The list of initial guess of the parameters, length 8.
    relax : bool, default: False
        If True, do not check for negative or shifted polynomial component.
    niter : int (optional)
        Apply Monte-Carlo method to calculate uncertainties for observational
        parameters, using 'niter' iterations.  Default is to use a much faster
        error propagation method.
    verbose : bool, default: False
        If true, print additional information and do not clear the temporary files.
    channel : list (optional)
        Define channel range to be fitted (zero-based). Default is all channels.
        Set to 0 for first/last channel, respectively.
    busyPath : string, default: ""
        The path of the busyfit function.

    Returns
    -------
    x : 1D array
        The x axis of the spectrum, filtered out the nan values.
    resTb : Table
        The table of fitting results.

    Notes
    -----
    None.
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    fltr_nan = np.isnan(x) | np.isnan(y)
    #-> Filter out the nan.
    if np.sum(fltr_nan) > 0:
        if verbose:
            print("There are nan in the values ignored!")
        fltr_nonnan = np.logical_not(fltr_nan)
        x = x[fltr_nonnan]
        y = y[fltr_nonnan]
    data = np.transpose( np.array([x, y]) )
    #-> Remove the existing files
    bsf_data = "tmp_bsf_data.dat"
    if os.path.isfile(bsf_data):
        os.remove(bsf_data)
    if os.path.isfile("busyfit_history.txt"):
        os.remove("busyfit_history.txt")
    if os.path.isfile("busyfit_output_fit.txt"):
        os.remove("busyfit_output_fit.txt")
    if os.path.isfile("busyfit_output_spectrum.txt"):
        os.remove("busyfit_output_spectrum.txt")
    #-> Run the fitting code
    np.savetxt(bsf_data, data, delimiter=" ", fmt=["%.4e", "%.4e"])
    cmdList = ["{0}busyfit -c 1 2 {1}".format(busyPath, bsf_data)]
    if not rms is None:
        cmdList.append("-n {0}".format(rms))
    if not p_init is None:
        if len(p_init) != 8:
            raise ValueError("The p_init should contain 8 parameters!")
        pList = []
        for p in p_init:
            pList.append("{0}".format(p))
        cmdList.append("-p {0}".format(" ".join(pList)))
    if relax:
        cmdList.append("-relax")
    if not niter is None:
        cmdList.append("-u {0}".format(niter))
    if verbose:
        cmdList.append("-v")
    if not channel is None:
        cmdList.append("-w {0} {1}".format(channel[0], channel[1]))
    command = " ".join(cmdList)
    if verbose:
        print(command)
    out = subprocess.call(command, shell=True)
    #-> Obtain the fitting results
    try:
        resTb = Table.read("busyfit_history.txt", format="ascii")
    except:
        raise RuntimeError("The fitting result is not found!")
    parList = ["Filename", "Success", "Nchan", "dof", "chi2", "chi2nu", "rms",
               "A", "dA", "B1", "dB1", "B2", "dB2", "C", "dC", "XE0", "dXE0",
               "XP0", "dXP0", "W", "dW", "N", "dN", "X", "dX", "W50", "dW50",
               "W20", "dW20", "Fpeak", "dFpeak", "Fint", "dFint"]
    for loop in range(len(parList)):
        pn = parList[loop]
        resTb.rename_column("col{0}".format(loop+1), pn)
    #-->If the fit does not start from the beginning, we need to shift the model
    # center by hand...
    if not channel is None:
        resTb["XE0"] += channel[0]
        if not resTb["XP0"] == 0:
            resTb["XP0"] += channel[0]
    #-> Clean the results
    if not verbose:
        os.remove(bsf_data)
        os.remove("busyfit_history.txt")
        os.remove("busyfit_output_fit.txt")
        os.remove("busyfit_output_spectrum.txt")
    return (x, resTb)

def rms_spectrum(x, y, flag=None, verbose=False):
    """
    Calculate the rms of the spectrum at the line-free region.

    Parameters
    ----------
    x : 1D array
        The x axis of the spectrum.
    y : 1D array
        The y axis of the spectrum.
    flag : [float, float], optional
        Ignore the data between x1 and x2.
    verbose : bool, default: False
        Print more information if True.
    Returns
    -------
    rms : float
        The rms of the spectrum.

    Notes
    -----
    None.
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    fltr_nan = np.isnan(x) | np.isnan(y)
    #-> Filter out the nan.
    if np.sum(fltr_nan) > 0:
        if verbose:
            print("There are nan in the values ignored!")
        fltr_nonnan = np.logical_not(fltr_nan)
        x = x[fltr_nonnan]
        y = y[fltr_nonnan]
    if not flag is None:
        fltr = (x < flag[0]) | (x > flag[1])
        y    = y[fltr]
    rms = np.std(y)
    return rms

def median_spectrum(x, y, flag=None, verbose=False):
    """
    Calculate the median of the spectrum at the line-free region.

    Parameters
    ----------
    x : 1D array
        The x axis of the spectrum.
    y : 1D array
        The y axis of the spectrum.
    flag : [float, float], optional
        Ignore the data between x1 and x2.
    verbose : bool, default: False
        Print more information if True.
    Returns
    -------
    med : float
        The median of the spectrum.

    Notes
    -----
    None.
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    fltr_nan = np.isnan(x) | np.isnan(y)
    #-> Filter out the nan.
    if np.sum(fltr_nan) > 0:
        if verbose:
            print("There are nan in the values ignored!")
        fltr_nonnan = np.logical_not(fltr_nan)
        x = x[fltr_nonnan]
        y = y[fltr_nonnan]
    if not flag is None:
        fltr = (x < flag[0]) | (x > flag[1])
        y    = y[fltr]
    med = np.median(y)
    return med

def busyProfile(x, a, b1, b2, c, xe, xp, w, n):
    """
    The busy profile.
    """
    bsf = 0.25 * a * (erf(b1 * (w + x - xe)) + 1) * (erf(b2 *(w - x + xe)) + 1) * \
          (c * np.abs(x - xp)**n + 1)
    return bsf

def busyFitProfile(fit_result, resolution=1):
    """
    Calculate the busy profile based on the busyfit result table.

    Parameters
    ----------
    fit_result: tuple
        x : 1D array
            The x axis of the spectrum.
        resTb : Table
            The result table of the busyfit function.

    Returns
    -------
    x : 1D array
        The x axis of the spectrum.
    bsf : 1D array
        The y axis of the spectrum.

    Notes
    -----
    None.
    """
    x, resTb = fit_result
    c = np.linspace(0, len(x)-1, len(x)*resolution)
    x = interp1d(np.arange(len(x)), x)(c)
    pars = busyResTb2List(resTb)
    bsf = busyProfile(c, *pars)
    return x, bsf

def busyResTb2List(resTb):
    """
    Convert the busyfit result table into a list that can be directly used by
    busyProfile().

    Parameters
    ----------
    resTb : Table
        The table of fitting results.

    Returns
    -------
    parList : list
        The list of parameters that can be directly used by busyProfile(x, *parList).
    """
    a  = resTb["A"][0]
    b1 = resTb["B1"][0]
    b2 = resTb["B2"][0]
    c  = resTb["C"][0]
    xe = resTb["XE0"][0]
    xp = resTb["XP0"][0]
    w  = resTb["W"][0]
    n  = resTb["N"][0]
    parList = [a, b1, b2, c, xe, xp, w, n]
    return parList

def lineParameters_BF(fit_result, perc=20, tol=0.01, resolution=10, verbose=False):
    """
    Convert the best-fit results to the line parameters.

    Parameters
    ----------
    fit_result: tuple
        x : 1D array
            The x axis of the spectrum.
        resTb : Table
            The result table of the busyfit function.
    perc : float, default: 20
        The percent of the line peak at which we calculate the line width and the line center.
    tol : float, (0, 1), default: 0.01
        The tolerance level for our calculations and sanity checks.
    resolution : float, default: 10
        The number of times to enhance the resolution of the model spectrum comparing to the data.
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
    x, resTb = fit_result
    c = np.linspace(0., len(x)-1., len(x)*resolution) # Make the indices of the channels
    x, bprf = busyFitProfile(fit_result, resolution)
    xfunc = interp1d(c, x) # Map from the channel to the x axis
    pars = busyResTb2List(resTb)
    #-> Find the location of the peaks
    yfunc = lambda x, p: -1. * busyProfile(x, *p)
    ce0 = np.int(resTb["XE0"][0] * resolution)
    cp10 = np.argmax(bprf[0:ce0+2]) / np.float(resolution)
    try:
        cp1 = minimize(yfunc, cp10, args=(pars,)).x[0]
    except:
        raise RuntimeError("Fail to locate the left peak!")
    yp1 = -1. * yfunc(cp1, pars)
    cp20 = (ce0 + np.argmax(bprf[ce0-1:])) / np.float(resolution)
    try:
        cp2 = minimize(yfunc, cp20, args=(pars,)).x[0]
    except:
        raise RuntimeError("Fail to locate the left peak!")
    yp2 = -1. * yfunc(cp2, pars)
    #-> Calculate the peak flux
    cpList = [cp1, cp2]
    ypList = [yp1, yp2]
    cmax   = cpList[np.argmax(ypList)]
    ymax   = np.max(ypList)
    #-> Find the left and right perc%-maximum channels
    yperc = ymax * perc / 100.
    yfunc = lambda x, p: busyProfile(x, *p) -  yperc
    #--> Left channel
    cfit1 = c[0]
    cfit2 = cp1
    #---> Try to avoid the error when the boundary is very sharp.
    if np.sign(yfunc(cfit1, pars) * yfunc(cfit2, pars)) == 1:
        cfit2 = cfit2 * (1. + tol)
    cPerc1, fit_flag = brentq(yfunc, cfit1, cfit2, args=(pars,), full_output=True)
    if not fit_flag.converged:
        raise RuntimeError("Fail to locate the left {0:.2f}% maxima!".format(perc))
    #--> Right channel
    cfit1 = cp2
    cfit2 = c[-1]
    #---> Try to avoid the error when the boundary is very sharp.
    if np.sign(yfunc(cfit1, pars) * yfunc(cfit2, pars)) == 1:
        cfit1 = cfit1 * (1. - tol)
    cPerc2, fit_flag = brentq(yfunc, cfit1, cfit2, args=(pars,), full_output=True)
    if not fit_flag.converged:
        raise RuntimeError("Fail to locate the right {0:.2f}% maxima!".format(perc))
    xp1    = xfunc(cp1)
    xp2    = xfunc(cp2)
    xPerc1 = xfunc(cPerc1)
    xPerc2 = xfunc(cPerc2)
    ccPerc = (cPerc1 + cPerc2) / 2. # Central channel
    xcPerc = (xPerc1 + xPerc2) / 2. # Center of the x axis based on perc% maxima
    wPerc  = np.abs(xPerc2 - xPerc1) # Full width at perc% maxima
    parDict = {
        "cPerc": (cPerc1, cPerc2),
        "xPerc": (xPerc1, xPerc2),
        "Cc"   : ccPerc,
        "Xc"   : xcPerc,
        "Wline": wPerc,
        "Fpeak": ymax,
        "Fperc": yperc,
        "cPeak": (cp1, cp2),
        "xPeak": (xp1, xp2)
    }
    return parDict

def busyFitPlot(fit_result, plot_w50=True, plot_w20=True, FigAx=None, yoffset=0., **kwargs):
    """
    Plot the results of busyfit()

    Parameters
    ----------
    fit_result: tuple
        x : 1D array
            The x axis of the spectrum.
        resTb : Table
            The result table of the busyfit function.
    plot_w50 : bool, default: True
        If True, plot the verticle lines indicating the W50 in green.
    plot_w20 : bool, default: True
        If True, plot the verticle lines indicating the W20 in blue.
    FigAx : tuple (optional)
        The tuple of (fig, ax) of the figure.
    yoffset : float
        The offset of the zeropoint of the y axis.
    **kwargs : dict
        The keywords for plotting the model spectrum.

    Returns
    -------
    FigAx : tuple
        The tuple of (fig, ax) of the figure.
    """
    if FigAx is None:
        fig = plt.figure(figsize=(8, 4))
        ax  = plt.gca()
    else:
        fig, ax = FigAx
    x, resTb = fit_result
    xc   = resTb["X"][0]
    W50  = resTb["W50"][0]
    W20  = resTb["W20"][0]
    x501 = xc - W50 / 2.
    x502 = xc + W50 / 2.
    x201 = xc - W20 / 2.
    x202 = xc + W20 / 2.
    fp   = resTb["Fpeak"][0]
    fp50 = fp * 0.5
    fp20 = fp * 0.2
    x, y = busyFitProfile(fit_result, resolution=10)
    ax.plot(x, y+yoffset, **kwargs)
    ax.axvline(x=xc, ls="-", color="k", label="Center")
    if plot_w50:
        ax.axvline(x=x501, ls="--", color="g", label="W50")
        ax.axvline(x=x502, ls="--", color="g")
        ax.axhline(y=fp50, ls="--", color="g")
    if plot_w20:
        ax.axvline(x=x201, ls="-.", color="b", label="W20")
        ax.axvline(x=x202, ls="-.", color="b")
        ax.axhline(y=fp20, ls="-.", color="b")
    return (fig, ax)

def lineParPlot(parDict, FigAx=None, **kwargs):
    """
    Plot the results of lineParameters().

    Parameters
    ----------
    parDict : dict
        The relevant parameters:
        xPerc : tuple, (xPerc1, xPerc2)
            Left and right x-axis values of the line profile at perc% of the peak flux.
        Xc : float
            The center of x-axis value calculated at perc% of the peak flux.
        Fperc : float
            Fpeak * perc / 100.
    FigAx : tuple (optional)
        The tuple of (fig, ax) of the figure.
    **kwargs : dict
        The keywords for the plotting.

    Returns
    -------
    FigAx : tuple
        The tuple of (fig, ax) of the figure.
    """
    if FigAx is None:
        fig = plt.figure(figsize=(8, 4))
        ax  = plt.gca()
    else:
        fig, ax = FigAx
    x1, x2 = parDict["xPerc"]
    xc = parDict["Xc"]
    yperc = parDict["Fperc"]
    ax.axvline(x=x1, **kwargs)
    kwargs["label"] = None
    ax.axvline(x=x2, **kwargs)
    ax.axhline(y=yperc, **kwargs)
    kwargs["ls"] = "-"
    ax.axvline(x=xc, **kwargs)
    return (fig, ax)
