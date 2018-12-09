import numpy as np

__all__ = ["mag2mass_Ks_NIR", "logM2L"]

def mag2mass_Ks_NIR(absM_gal, absM_sun, c, c_sun, absM_gal_e=None, h=0.678):
    """
    Calculate the galaxy stellar mass from the absolute magnitude. The
    mass-to-light ratio (M/L) is based on the K-band M/L from Bell et al.
    (2003). The *-Ks color is used to transfer the given band to Ks band.
    The uncertainty of the K-band M/L is 0.2 dex, which is included in
    the uncertainty of the stellar mass.

    Parameters
    ----------
    absM_gal : float
        The absolute magnitude of the galaxy.
    absM_sun : float
        The absolute magnitude of the Sun.
    c : float
        The *-Ks color of the galaxy. The first band should be consistent
        with absM_gal.
    c_sun : float
        The color of the Sun in the corresponding bands.
    absM_gal_e (optional) : float
        The uncertainty of the absolute magnitude of the galaxy.
    h : float; default: 0.678
        The Hubble constant.

    Returns
    -------
    logMstar : float
        The stellar mass of the galaxy.
    elogMstar : float
        The uncertainty of the stellar mass. The uncertainty of the
        mass-to-light ratio is 0.2 dex. If absM_gal_e is provided, it will
        be included by quadrature sum in elogMstar.

    Notes
    -----
    None.
    """
    logMstar = -0.43 - 0.41 * (absM_gal - absM_sun) + 0.41 * (c - c_sun) + 0.068 * np.log10(h)
    if absM_gal_e is None:
        elogMstar = np.ones_like(logMstar) * 0.2
    else:
        elogMstar = np.sqrt((0.41 * absM_gal_e)**2 + 0.2**2)
    return logMstar, elogMstar


def logM2L(color, a, b):
    """
    Calculate the mass-to-light ratio based on the color.
    logM2L = a + (b * color)
    
    Parameters
    ----------
    color : float or array like
        The color of the galaxy.
    a : float
        The normalization of the relation.
    b : float
        The slope of the relation.
    
    Returns
    -------
    logm2l : float or array like
        The logarithmic mass-to-light ratio.
        
    Notes
    -----
    None.
    """
    logm2l = a + b * color
    return logm2l
