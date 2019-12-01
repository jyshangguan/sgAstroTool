from __future__ import division
import numpy as np
__all__ = ["coLuminosity", "Lco2MH2"]

def coLuminosity(Ico, nu_rest, DL, z):
    """
    Calculate the line luminosity of CO, based on Equation (3) of
    Solomon & Vanden Bout (2005).

    Parameters
    ----------
    Ico : array like
        The integrated line flux, units: Jy km/s.
    nu_rest : float
        The rest-frame frequency.
    DL : float
        The luminosity distance.
    z : float
        The redshift.

    Returns
    -------
    Lco : array like
        The CO line luminosity, units: K km s^-1 pc^2.
    """
    Lco = 3.25e7 * Ico / nu_rest**2. * DL**2. / (1 + z)
    return Lco


def Lco2MH2(Lco, alphaCO=4.3, rLines=1.):
    """
    Calculate the molecular gas mass.

    Parameters
    ----------
    Lco : array like
        The CO line luminosity, units: K km s^-1 pc^2.
    alphaCO : float
        The CO-to-MH2 conversion factor.
    rLines : float
        The line luminosity ratio, say r21=L_CO(2-1)/L_CO(1-0).

    Returns
    -------
    mH2 : array like
        The molecular gas mass, units: Msun.
    """
    mH2 = alphaCO * Lco / rLines
    return mH2
