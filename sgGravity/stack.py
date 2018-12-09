import numpy as np
from scipy import signal
from scipy.ndimage.filters import gaussian_filter1d

def stack_data(data, wave, sigma=20, deg=1, mode="div"):
    """
    Stack the visibility amplitude data.
    """
    assert len(data.shape) == 3
    nt, nb, nch = data.shape
    dconv = gaussian_filter1d(data, sigma, axis=2)
    if mode == "div":
        dflat = data / dconv
    elif mode == "sub":
        dflat = data - dconv
    else:
        raise ValueError("The mode ({0}) is not recognized!".format(mode))
    rms   = np.zeros_like(data)
    for i in range(nt):
        for j in range(nb):
            if mode == "div":
                df = flatten_fit_poly_div(wave, dflat[i,j,:], deg)
            elif mode == "sub":
                df = flatten_fit_poly_sub(wave, dflat[i, j, :], deg)
            dflat[i,j,:] = df
            rms[i,j,:] = np.std(df)
    wt = rms**-2
    dstck, wtsum = np.ma.average(dflat, axis=0, weights=wt, returned=True)
    estck = 1. / np.sqrt(wtsum)
    return dstck, estck

def convolve_data(data, sigma):
    """
    Use Gaussian convolution to smooth the data.
    """
    assert len(data.shape) == 3
    data_conv = gaussian_filter1d(data, sigma, axis=2)
    return data_conv

def weighted_avg_and_std(values, weights, axis=None):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    ndim = len(values.shape)
    fltr_nan = np.isnan(weights)
    weights[fltr_nan] = 0
    average = np.average(values, weights=weights, axis=axis)
    average = np.expand_dims(average, ndim)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights, axis=axis)
    return (average, np.sqrt(variance))

def flatten_conv_gauss_sub(data, sigma=None, verbose=False):
    """
    Remove the large scale structure of the data by subtracting
    the convolved data.
    """
    len_data = len(data)
    if sigma is None:
        sigma = len_data / 20.
    win = signal.gaussian(len_data, sigma)
    data_conv = signal.convolve(data, win, mode='same') / sum(win)
    data_flat = data - data_conv
    if verbose:
        return data_flat, win, data_conv
    else:
        return data_flat

def flatten_conv_gauss_div(data, sigma=None, verbose=False):
    """
    Remove the large scale structure of the data by dividing
    the convolved data.
    """
    len_data = len(data)
    if sigma is None:
        sigma = len_data / 20.
    win = signal.gaussian(len_data, sigma)
    data_conv = signal.convolve(data, win, mode='same') / sum(win)
    data_flat = data / data_conv
    if verbose:
        return data_flat, win, data_conv
    else:
        return data_flat

def flatten_fit_poly_sub(x, y, deg=3, fltr=None, return_pars=False, verbose=False, **kwargs):
    """
    Remove the large scale structure of the data by subtracting
    the fitted polynomial function.
    """
    if fltr is None:
        x_fit = x
        y_fit = y
    else:
        x_fit = x[fltr]
        y_fit = y[fltr]
    z = np.polyfit(x_fit, y_fit, deg, **kwargs)
    y_poly  = np.polyval(z, x)
    y_flat = y - y_poly
    if verbose:
        return y_flat, y_poly
    elif return_pars:
        return z
    else:
        return y_flat

def flatten_fit_poly_div(x, y, deg=3, fltr=None, return_pars=False, verbose=False, **kwargs):
    """
    Remove the large scale structure of the data by dividing
    the fitted polynomial function.
    """
    if fltr is None:
        x_fit = x
        y_fit = y
    else:
        x_fit = x[fltr]
        y_fit = y[fltr]
    z = np.polyfit(x_fit, y_fit, deg, **kwargs)
    y_poly  = np.polyval(z, x)
    y_flat = y / y_poly
    if verbose:
        return y_flat, y_poly
    elif return_pars:
        return z
    else:
        return y_flat
