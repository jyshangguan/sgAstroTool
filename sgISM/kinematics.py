from __future__ import division
import numpy as np

__all__ = ["q2InclinationAngle"]

def q2InclinationAngle(q, q0=0.2):
    """
    Calculate the inclination angle from the axis ratio.

    Parameters
    ----------
    q : array like
        The axis ratio, b/a.
    q0 : float
        The intrinsic axis ratio when the galaxy is edge-on.

    Returns
    -------
    i : array like
        The inclination angle, units: radian.

    Notes
    -----
    The equation is from Topal et al. (2018MNRAS.479.3319T), equation (1).
    """
    i = np.arccos(np.sqrt( (q**2 - q0**2) / (1.0 - q0**2) ))
    return i
