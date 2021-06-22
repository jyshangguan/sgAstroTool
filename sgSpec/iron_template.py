import os
import numpy as np
from astropy.table import Table
from astropy.modeling.models import custom_model
from scipy.ndimage.filters import gaussian_filter1d

__all__ = ['IronTemplate']

pathList = os.path.abspath(__file__).split("/")
package_path = '/'.join(pathList[:-1])

irontemp = Table.read('{0}/data/irontemplate.ipac'.format(package_path),
                      format='ipac')
wave_temp = irontemp['Spectral_axis'].data
flux_temp = irontemp['Intensity'].data


@custom_model
def IronTemplate(x, amplitude=1, stddev=400, z=0):
    '''
    This is a Fe template of AGN from I Zw 1 (Boroson & Green 1992).

    Parameters
    ----------
    x : array like
        Wavelength, units: Angstrom.
    amplitude : float
        Amplitude of the template, units: arbitrary.
    stddev : float
        Velocity dispersion of the AGN, units: km/s. Lower limit about 390 km/s.
    z : float
        Redshift of the AGN.

    Returns
    -------
    flux_intp : array like
        The interpolated flux of iron emission.
    '''
    stddev_intr = 900 / 2.3548  # Velocity dispersion of I Zw 1.
    if stddev < stddev_intr:
        stddev = 910 / 2.3548

    # Get Gaussian kernel width (channel width 103.6 km/s)
    sig = np.sqrt(stddev**2 - stddev_intr**2) / 103.6
    flux_conv = gaussian_filter1d(flux_temp, sig)
    flux_intp = amplitude * np.interp(x, wave_temp * (1 + z), flux_conv)
    return flux_intp
