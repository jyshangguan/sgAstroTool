from __future__ import division
from __future__ import absolute_import
import numpy as np
from .constants import k_erg, ls_km

def effective_area(N, d_dish=7., band=6):
    """
    The effective area.

    Parameters
    ----------
    N : int
        The number of antennas.
    d_dish : float
        Diameter of the dish, units: m.
    band : int
        The number of the band.

    Returns
    -------
    aeff : float
        The effective area of the telescope, units: m.
    """
    itaDict_12 = {
        3: 0.71,
        4: 0.70,
        5: 0.69,
        6: 0.68,
        7: 0.63,
        8: 0.60,
        9: 0.43,
        10: 0.31
    }
    itaDict_7 = {
        3: 0.71,
        4: 0.71,
        5: 0.70,
        6: 0.69,
        7: 0.66,
        8: 0.64,
        9: 0.52,
        10: 0.42
    }
    if d_dish == 7.:
        itaDict = itaDict_7
    elif d_dish == 12.:
        itaDict = itaDict_12
    else:
        raise ValueError("The dish diameter ({0}) is not correct!".format(d_dish))
    if not band in itaDict:
        raise ValueError("The band ({0}) is not recognized!".format(band))
    aeff = np.pi * (d_dish / 2.)**2. * itaDict[band]
    return aeff

def omega(theta):
    """
    Calculate the beam solid angle.

    Parameters
    ----------
    theta : float
        The spatial resolution, units: arcsec.

    Returns
    -------
    omg : float
        The beam solid angle, units: steradian.
    """
    theta *= np.pi / 180 / 3600 # Convert to radian
    omg = np.pi * theta**2 / (4 * np.log(2))
    return omg

def sigma_S(Tsys, wr=1.1, Aeff=1., fs=0, itaq=0.96, itac=0.88, N=10,
            npol=2, delta_nu=1., tint=1):
    """
    The fomular to calculate the point source sensitivity.
    Please refer to ALMA Technical Handbook.

    Parameters
    ----------
    Tsys : float
        The system temperature, unit: K.
    wr : float
        Robust weighting factor, default: 1.1.
    Aeff : float
        Effective area, units: m^2.
    fs : float
        Shadowing fraction.
    itaq : float
        Quantization efficiency, default: 0.96.
    itac : float
        Correlator efficiency, default: 0.88.
    N : int
        The number of antennas.
    npol : int
        The number of polarization, 1 or 2, default: 2.
    delta_nu : float
        Resolution element width, units: GHz.
    tint : float
        Integration time, units: minutes.

    Return
    ------
    sigma : float
        The sensitivity, units: Jy.
    """
    Aeff *= 1e4 # Convert to cm^2
    delta_nu *= 1e9 # Convert to Hz
    tint *= 60 # Convert to second
    sigma = (wr * 2 * k_erg * Tsys) / (itaq * itac * Aeff * (1 - fs) * \
            (N * (N - 1) * npol * delta_nu * tint)**0.5)
    sigma *= 1e23 # Convert to Jy
    return sigma

def Sensitivity_ALMA(band, N, Tsys, tint, delta_nu, array="ACA", **kws_sig):
    """
    Calculate the sensitivity of ALMA.

    Parameters
    ----------
    band : int
        The number of the band.
    N : int
        The number of antennas.
    Tsys : float
        The system temperature, unit: K.
    tint : float
        Integration time, units: minutes.
    delta_nu : float
        Resolution element width, units: GHz.
    array : string
        The type of the array, "ACA" or "12m", default: "ACA".
    **kws_sig : other parameters of sigma_S.

    Returns
    -------
    sigma : float
        The sensitivity, unit: Jy.
    """
    if array == "ACA":
        d_dish = 7
        if N > 20:
            raise ValueError("The number of antennas ({0}) is likely wrong!!".format(N))
    elif array == "12m":
        d_dish = 12
    else:
        raise ValueError("The array type ({0}) is not recognized!".format(array))
    Aeff = effective_area(N, d_dish, band)
    kws_sig["N"] = N
    kws_sig["tint"] = tint
    kws_sig["delta_nu"] = delta_nu
    if not "Aeff" in kws_sig:
        kws_sig["Aeff"] = Aeff
    sigma = sigma_S(Tsys, **kws_sig)
    return sigma
