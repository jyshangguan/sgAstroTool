import numpy as np

__all__ = ["median_spectrum", "rms_spectrum"]

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
