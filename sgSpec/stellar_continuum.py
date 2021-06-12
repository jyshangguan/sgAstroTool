import os
import numpy as np
from astropy.modeling.models import custom_model
from scipy.ndimage import gaussian_filter

ls_km = 2.99792458e5  # km/s
pathList = os.path.abspath(__file__).split("/")
package_path = '/'.join(pathList[:-1])

__all__ = ['stellar_11Gyr', 'stellar_300Myr']


spec_11gyr = np.loadtxt('{0}/data/sed_bc03_11Gyr.dat'.format(package_path))
wave_11gyr = spec_11gyr[:, 0]
flux_11gyr = spec_11gyr[:, 1]
logw_11gyr = np.log(wave_11gyr)
logw_even_11gyr = np.linspace(logw_11gyr[0], logw_11gyr[-1], 5*len(logw_11gyr))
logf_even_11gyr = np.interp(logw_even_11gyr, logw_11gyr, np.log(flux_11gyr))
wave_even_11gyr = np.exp(logw_even_11gyr)
flux_even_11gyr = np.exp(logf_even_11gyr)
flux_even_11gyr /= np.max(flux_even_11gyr)

spec_300myr = np.loadtxt('{0}/data/sed_bc03_300Myr.dat'.format(package_path))
wave_300myr = spec_300myr[:, 0]
flux_300myr = spec_300myr[:, 1]
logw_300myr = np.log(wave_300myr)
logw_even_300myr = np.linspace(logw_300myr[0], logw_300myr[-1], 5*len(logw_300myr))
logf_even_300myr = np.interp(logw_even_300myr, logw_300myr, np.log(flux_300myr))
wave_even_300myr = np.exp(logw_even_300myr)
flux_even_300myr = np.exp(logf_even_300myr)
flux_even_300myr /= np.max(flux_even_300myr)


@custom_model
def stellar_11Gyr(x, fmax=1, sigma=100, z=0):
    '''
    The stellar continuum with 11 Gyr single stellar population.

    Parameters
    ----------
    x : array like
        Wavelength, units: Angstrom.
    fmax : float
        The maximum flux of the stellar continuum.
    sigma : float
        The velocity dispersion, units: km s^-1.
    z : float
        The redshift.

    Returns
    -------
    flux : array like
        The SED flux of the stellar continuum, units: unit: per Angstrom.
    '''
    x = x / (1 + z)
    assert (np.min(x) >= wave_even_11gyr[0]) & (np.max(x) <= wave_even_11gyr[-1]), \
           'The wavelength is out of the supported range ({0:.0f}-{1:.0f})!'.format(wave_even_11gyr[0], wave_even_11gyr[-1])
    s = sigma / ls_km
    nsig = s / (logw_even_11gyr[1] - logw_even_11gyr[0])
    flux_conv = gaussian_filter(flux_even_11gyr, nsig)
    flux = np.interp(x, wave_even_11gyr, fmax * flux_conv)
    return flux

@custom_model
def stellar_300Myr(x, fmax=1, sigma=100, z=0):
    '''
    The stellar continuum with 300 Myr single stellar population.

    Parameters
    ----------
    x : array like
        Wavelength, units: Angstrom.
    fmax : float
        The maximum flux of the stellar continuum.
    sigma : float
        The velocity dispersion, units: km s^-1.
    z : float
        The redshift.

    Returns
    -------
    flux : array like
        The SED flux of the stellar continuum, units: unit: per Angstrom.
    '''
    x = x / (1 + z)
    assert (np.min(x) >= wave_even_300myr[0]) & (np.max(x) <= wave_even_300myr[-1]), \
           'The wavelength is out of the supported range ({0:.0f}-{1:.0f})!'.format(wave_even_300myr[0], wave_even_300myr[-1])
    s = sigma / ls_km
    nsig = s / (logw_even_300myr[1] - logw_even_300myr[0])
    flux_conv = gaussian_filter(flux_even_300myr, nsig)
    flux = np.interp(x, wave_even_300myr, fmax * flux_conv)
    return flux
