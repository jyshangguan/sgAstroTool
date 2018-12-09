import ezgal
import numpy as np
import pysynphot as S
from scipy.interpolate import interp1d
from bandpass import averageFnu

__all__ = ["loadSPModel", "kcorrect", "KCorrect", "galColor"]

filter_path = "/Users/shangguan/Softwares/my_module/sgPhot/filters/"

def loadSPModel(model_name='bc03_ssp_z_0.02_chab.model', Om=0.308, Ol=0.692, h=0.678):
    """
    Load the stellar population model. The model we adopt is from EzGal. We
    adopt the Planck cosmology (2015) as default.

    Parameters
    ----------
    model_name : str; default: BC03 SSP solar metallicity Chabrier IMF.
        The name of the model.
    Om : float
        The cosmological parameter, Omega matter.
    Ol : float
        The cosmological parameter, Omega lambda.
    h : float
        The Hubble constant normalized by 100 km/s/Mpc.

    Returns
    -------
    model : EzGal
        The EzGal stellar sythesis model.

    Notes
    -----
    None.
    """
    model = ezgal.ezgal(model_name)
    model.set_cosmology(Om=Om, Ol=Ol, h=h)
    return model

def loadBandPass(filter_name, band_name=None, wave_unit="micron", band_unit="angstrom"):
    """
    Load the bandpass. The filter position and names are obtained and
    specified by myself.

    Parameters
    ----------
    filter_name : string
        The name of the fitler.
    band_name (optional) : string
        The name of the band.
    wave_unit : string, default: "micron"
        The unit of the wavelength in the filter file.
    band_unit : string, default: "angstrom"
        The unit of the wavelength used in the bandpass data.

    Returns
    -------
    bp : pysynphot bandpass
        The bandpass generated from pysynphot.

    Notes
    -----
    None.
    """
    bp_array = np.genfromtxt(filter_path+"{0}.dat".format(filter_name))
    bp = S.ArrayBandpass(bp_array[:, 0], bp_array[:, 1], waveunits=wave_unit,
                         name=band_name)
    bp.convert(band_unit)
    return bp

def kcorrect(z, age, model, bandpass, wave_norm=None):
    """
    Calculate the K-correction based on stellar synthesis model and the input
    bandpass. The K-correction is defined as the observed magnitude subtracted
    by the intrinsic magnitude: kmag = m_obs - m_int.

    Parameters
    ----------
    z : float
        The redshift of the target.
    age : float
        The age of the stellar population, unit: Gyr.
    model : EzGal
        The EzGal stellar sythesis model.
    bandpass : pysynphot bandpass
        The bandpass generated from pysynphot.
    wave_norm (optional) : float
        The wavelength to normalize the spectrum. If not specified, the spectrum
        will not be normalized.

    Returns
    -------
    kmag : float
        The k-correction result.

    Notes
    -----
    None.
    """
    wave   = model.ls
    wave_z = wave * (1 + z)
    sed  = model.get_sed(age, age_units="gyrs", units='Fv')
    if wave_norm is None:
        sed_norm = sed
    else:
        sed_int  = interp1d(wave, sed)
        sed_norm  = sed / sed_int(wave_norm)
    fnu   = averageFnu(wave, sed_norm, bandpass, wave_units="angstrom")
    fnu_z = averageFnu(wave_z, sed_norm, bandpass, wave_units="angstrom")
    kmag = -2.5 * np.log10(fnu_z / fnu)
    return kmag

def KCorrect(z, age, filter_name, model_name="bc03_ssp_z_0.02_chab.model",
             band_name=None, wave_norm=None, wave_unit="micron",
             band_unit="angstrom", Om=0.308, Ol=0.692, h=0.678, return_all=False):
    """
    Calculate the K-correction.

    Parameters
    ----------
    z : float
        The redshift of the target.
    age : float
        The age of the stellar population, units: Gyr.
    filter_name : string
        The name of the fitler.
     model_name : str; default: BC03 SSP solar metallicity Chabrier IMF.
        The name of the model.
    band_name (optional) : string
        The name of the band.
    wave_norm (optional) : float
        The wavelength to normalize the spectrum. If not specified, the spectrum
        will not be normalized.
    band_unit : string, default: "angstrom"
        The unit of the wavelength used in the bandpass data.
    Om : float
        The cosmological parameter, Omega matter.
    Ol : float
        The cosmological parameter, Omega lambda.
    h : float
        The Hubble constant normalized by 100 km/s/Mpc.
    return_all : bool
        If True, return all the results of kcorrect as well as the model and
        bandpass.

    Returns
    -------
    If return_all is True, return all the results of kcorrect as well as the model
    and bandpass. Otherwise, return the k-correction value only.

    Notes
    -----
    None.
    """
    model = loadSPModel(model_name, Om, Ol, h)
    bp = loadBandPass(filter_name, band_name, wave_unit, band_unit)
    if wave_norm is None:
        wave_norm = bp.avgwave()
    kmag = kcorrect(z, age, model, bp, wave_norm)
    if return_all:
        return kmag, model, bp
    else:
        return kmag

def galColor(age, model, bp1, bp2, wave_norm=None):
    wave   = model.ls
    sed  = model.get_sed(age, age_units="gyrs", units='Fv')
    if wave_norm is None:
        sed_norm = sed
    else:
        sed_int  = interp1d(wave, sed)
        sed_norm  = sed / sed_int(wave_norm)
    sp = S.ArraySpectrum(wave, sed_norm, waveunits="angstrom", fluxunits="fnu",
                         name="{0:.1f} Gyr".format(age))
    obs1 = S.Observation(sp, bp1, binset=wave)
    obs2 = S.Observation(sp, bp2, binset=wave)
    mag1 = obs1.effstim("vegamag")
    mag2 = obs2.effstim("vegamag")
    color = mag1 - mag2
    return color
