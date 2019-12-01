# This package provides some functions using the Monte Carlo method.
#
from __future__ import print_function
from builtins import range
import numpy as np

__all__ = ["dataDiscretePerturber", "dataContinuePerturber", "dataBinPerturb"]

def dataDiscretePerturber(parData, lparData, uparData, parList, QuietMode=True):
    """
    Perturb the data according to the lower and upper limit. The parameter space is
    assumed to be discrete.

    Parameters
    ----------
    parData : 1D array
        The value of the data.
    lparData : 1D array
        The lower limit of the data.
    uparData : 1D array
        The upper limit of the data.
    parList : 1D array
        All the values of the parameter space, from small to large.
    QuietMode (optional): bool
        Do not print if True.

    Returns
    -------
    parPert : 1D array
        The perturbed data.

    Notes
    -----
    None.
    """
    parData  = np.atleast_1d(parData)
    lparData = np.atleast_1d(lparData)
    uparData = np.atleast_1d(uparData)
    npar = len(parData)
    parPert  = []
    for loop in range(npar):
        par  = parData[loop]
        upar = uparData[loop]
        lpar = lparData[loop]
        if np.sum(np.isnan([par, upar, lpar])) > 0:
            raise ValueError("There is nan in the input data list!")
        fltr_par = (parList<=upar) & (parList>=lpar)
        if np.sum(fltr_par) == 1:
            idx = np.argmax(fltr_par)
            lidx = np.max([0, idx - 1])
            uidx = np.min([len(parList)-1, idx+1])
            lpar = parList[lidx]
            upar = parList[uidx]
            if not QuietMode:
                print("You need to enlarge the error bar!")
                print(np.argmax(fltr_par), lpar, par, upar)
            fltr_par = (parList<=upar) & (parList>=lpar)
        par_sampled = np.random.choice(parList[fltr_par], 1)[0]
        parPert.append(par_sampled)
    parPert = np.array(parPert)
    return parPert

def dataContinuePerturber(parData, lparData, uparData, parMask=None, QuietMode=True):
    """
    Perturb the data according to the lower and upper limit. The parameter space is
    assumed to be continuum.

    Parameters
    ----------
    parData : 1D array
        The value of the data.
    lparData : 1D array
        The lower limit of the data.
    uparData : 1D array
        The upper limit of the data.
    parMask : 1D array
        The array of mask, with the same length as the parData. 1 to ignore the
        data, 0 to perturb the data.
    QuietMode (optional): bool
        Do not print if True.

    Returns
    -------
    parPert : 1D array
        The perturbed data.

    Notes
    -----
    None.
    """
    parData  = np.atleast_1d(parData)
    lparData = np.atleast_1d(lparData)
    uparData = np.atleast_1d(uparData)
    npar = len(parData)
    if parMask is None:
        parMask = np.zeros(npar)
    else:
        parMask = np.atleast_1d(parMask)
    parPert  = []
    for loop in range(npar):
        par  = parData[loop]
        upar = uparData[loop]
        lpar = lparData[loop]
        mask = parMask[loop]
        #-> If the data is mask, use the original value.
        if mask:
            parPert.append(par)
            continue
        if(lpar > upar):
            raise ValueError("lpar ({0}) > upar ({1})!".format(lpar, upar))
        par_sampled = (upar - lpar) * np.random.randn() + par
        parPert.append(par_sampled)
        if not QuietMode:
            print("{0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}".format(lpar, par, upar, par_sampled))
    parPert = np.array(parPert)
    return parPert

def dataBinPerturb(parData, bin_edge, parList=None, npert=100, lparData=None,
                   uparData=None, unctPerc=34., QuietMode=True):
    """
    Bin the data and estimate the uncertainty of the value in each bin by
    perturb the data according to the uncertainty of the data.

    Parameters
    ----------
    parData : 1D array
        The value of the data.
    bin_edge : 1D array
        The number of the bins or the edge of the bins.
    parList (optional): 1D array
        All the discrete values of the parameter space, from small to large. If
        not provided, the parameter spaces are considered continuum.
    lparData (optional): 1D array
        The lower limit of the data.
    uparData (optional): 1D array
        The upper limit of the data.
    unctPerc (optional): float
        The uncertainty that reflecting 50+-unctPerc error bars. Default: 34 for
        the normal standard deviation.
    QuietMode (optional): bool
        Do not print if True.

    Returns
    -------
    binMEDList : 1D array
        The perturbed median of each bin.
    binUNCList: 2D array
        The list of +-unctPerc error bars of each bin.
    parMEDList : 1D array
        The perturbed median of each parameter.
    parUNCList: 2D array
        The list of +-unctPerc error bars of each parameter.

    Notes
    -----
    None.
    """
    if parList is None:
        assert not lparData is None
        assert not uparData is None
        pertFunction = dataContinuePerturber
        pert_args    = [parData, lparData, uparData, None, QuietMode]
    else:
        pertFunction = dataDiscretePerturber
        if lparData is None:
            lparData = parData
        if uparData is None:
            uparData = parData
        pert_args    = [parData, lparData, uparData, parList, QuietMode]
    parPertList = []
    binPertList = []
    for loop in range(npert):
        parPert = pertFunction(*pert_args)
        parPertList.append(parPert)
        bin_par, edge_par = np.histogram(parPert, bins=bin_edge)
        binPertList.append(bin_par)
    parPertList = np.array(parPertList)
    binPertList = np.array(binPertList)
    parMEDList = []
    parUNCList = []
    binMEDList = []
    binUNCList = []
    npar = len(parData)
    for loop in range(npar):
        parSample = parPertList[:, loop]
        pl, pm, ph = np.percentile(parSample, [50-unctPerc, 50, 50+unctPerc])
        parMEDList.append(pm)
        parUNCList.append([pm - pl, ph - pm])
    nbin = len(bin_edge) - 1
    for loop in range(nbin):
        binSample = binPertList[:, loop]
        pl, pm, ph = np.percentile(binSample, [50-unctPerc, 50, 50+unctPerc])
        binMEDList.append(pm)
        binUNCList.append([pm - pl, ph - pm])
    parMEDList = np.array(parMEDList)
    parUNCList = np.array(parUNCList)
    binMEDList = np.array(binMEDList)
    binUNCList = np.array(binUNCList)
    return binMEDList, binUNCList, parMEDList, parUNCList
