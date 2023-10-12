import numpy as np

def major2Re_expo(major):
    '''
    Convert major FWHP to Re for an exponential disk.

    Parameters
    ----------
    major : float
        FWHM of the major axis.

    Returns
    -------
    Re : float
        The effective radius or half light radius.
    '''
    Re = 1.678 / (2 * np.log(2)) * major
    return Re

def major2Re_gaus(major):
    '''
    Convert major FWHP to Re for an Gaussian disk.
    
    Parameters
    ----------
    major : float
        FWHM of the major axis.

    Returns
    -------
    Re : float
        The effective radius or half light radius.
    '''
    Re = 0.5 * major
    return Re