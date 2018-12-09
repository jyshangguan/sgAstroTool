import numpy as np

__all__ = ["GDR_magdis12", "metal_kewley08_pp04"]

def GDR_magdis12(Zo, eZo=None):
    """
    The gas-to-dust ratio calculated from the metallicity. The
    equation comes from Magdis et al., ApJ, 760, 6, 2012. The 
    scatter of the relation is 0.15 dex.

    Parameters
    ----------
    Zo : float
        The metallicity in the form 12 + log(O/H).
    eZo : (optional) float
        The uncertainty of the metallicity.

    Returns
    -------
    gdr : float
        The log scale gas-to-dust ratio of the galaxy.

    Notes
    -----
    None.
    """
    gdr = 10.54-0.99*Zo
    if eZo is None:
        return gdr
    else:
        egdr = np.sqrt((0.99*eZo)**2. + 0.15**2.)
        return gdr, egdr

def metal_kewley08_pp04(logM):
    """
    This function calculate the 12+log(O/H) from the stellar mass. 
    The equation come from Kewley & Ellison, ApJ, 681, 1183, 2008.
    The 12+log(O/H) is obtained from PP04 (N2) method. The rms 
    residual is 0.09 dex.
    
    Parameters
    ----------
    logM : float
        The stellar mass with unit solar mass in logrithm.
    
    Returns
    -------
    metal : float
        The metallicity in the form 12 + log(O/H).

    Notes
    -----
    The RMS is 0.09 dex.
    """
    metal = 23.9049 - 5.62784 * logM + 0.645142 * logM**2. - 0.0235065 * logM**3.
    return metal

############################ Old functions ############################
def metalFunc(logM, z, elogM=None):
    '''
    This function calculate the metallicity from the stellar
    mass and redshift of the galaxy. The equation comes from
    Berta et al., A&A, 587, A73, 2016.

    Parameters
    ----------
    logM : float
        The stellar mass with unit solar mass in logrithm.
    z : float
        The redshift of the galaxy.

    Returns
    -------
    metal : float
        The metallicity in the form 12 + log(O/H).

    Notes
    -----
    None.
    '''
    a = 8.74
    b = 10.4 + 4.46 * np.log10(1 + z) - 1.78 * (np.log10(1 + z))**2.
    metal = a - 0.087 * (logM - b)**2.0
    if elogM is None:
        return metal
    else:
        emetal = abs(-0.087 * 2 * (logM - b) * elogM)
        return metal, emetal

#logGDRFunc = lambda metal: 10.54-0.99*metal
def gdrFunc(metal, emetal=None):
    """
    The gas to dust ratio calculated from the metallicity. The
    equation comes from Magdis et al., ApJ, 760, 23, 2012. The 
    scatter of the relation is 0.15 dex.

    Parameters
    ----------
    metal : float
        The metallicity in the form 12 + log(O/H).

    Returns
    -------
    gdr : float
        The gas to dust ratio.

    Notes
    -----
    None.
    """
    gdr = 10.54-0.99*metal
    if emetal is None:
        return gdr
    else:
        egdr = np.sqrt((0.99*emetal)**2. + 0.15**2.)
        return gdr, egdr
