import os
import numpy as np
from astropy.table import Table
from astropy.modeling.models import custom_model
from scipy.ndimage.filters import gaussian_filter1d
from astropy.modeling.core import Fittable1DModel
from astropy.modeling.parameters import Parameter

__all__ = ['IronTemplate']

pathList = os.path.abspath(__file__).split("/")
package_path = '/'.join(pathList[:-1])

irontemp = Table.read('{0}/data/irontemplate.ipac'.format(package_path),
                      format='ipac')
wave_temp = irontemp['Spectral_axis'].data
flux_temp = irontemp['Intensity'].data
flux_temp /= np.max(flux_temp)


class IronTemplate(Fittable1DModel):
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

    amplitude = Parameter(default=1, bounds=(0, None))
    stddev = Parameter(default=910/2.3548, bounds=(910/2.3548, None))
    z = Parameter(default=0, bounds=(0, 10))

    @staticmethod
    def evaluate(x, amplitude, stddev, z):
        """
        Gaussian model function.
        """
        stddev_intr = 900 / 2.3548  # Velocity dispersion of I Zw 1.
        if stddev < stddev_intr:
            stddev = 910 / 2.3548

        # Get Gaussian kernel width (channel width 103.6 km/s)
        sig = np.sqrt(stddev**2 - stddev_intr**2) / 103.6
        flux_conv = gaussian_filter1d(flux_temp, sig)
        f = amplitude * np.interp(x, wave_temp * (1 + z), flux_conv)

        return f
