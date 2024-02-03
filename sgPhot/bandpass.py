from __future__ import division
import numpy as np
import pysynphot as S

__all__ = ["loadBandPass", "averageFnu"]

filter_path = "/Users/shangguan/Softwares/my_module/sgAstroTool/sgPhot/filters/"
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

def averageFnu(wavelength, flux, bandpass, wave_units="micron", tol=1e-3, QuietMode=False):
    """
    Calculate the average flux density (fnu) based on the input spectrum
    (wavelength and flux) and bandpass.  The input bandpass should be a photon
    response function:
        fnu = integrate (fnu * bp dnu / nu) / integrate (bp dnu / nu)

    Parameters
    ----------
    wavelength : array like
        The array of the wavelength of the spectrum.
    flux : array like
        The array of the flux density of the spectrum.
    bandpass : pysynphot bandpass
        The filter response curve, which should be a photon response function.
    wave_units : string
        The units of the wavelength that should be matched for both spectrum and
        bandpass.
    tol : float; default: 0.001
        The tolerance of the maximum response outside the overlapping wavelength.
    QuietMode : bool; default: False
        Do not raise warning or print anything, if True.

    Returns
    -------
    fnu : float
        The band-average flux density.

    Notes
    -----
    None.
    """
    bandpass.convert(wave_units)
    #-> Find the overlaping wavelength regime.
    wave_bp = bandpass.wave
    wmin = np.max([np.nanmin(wavelength), np.nanmin(wave_bp)])
    wmax = np.min([np.nanmax(wavelength), np.nanmax(wave_bp)])
    #-> Check the throughput
    thrp = bandpass.throughput
    thrp_max = np.max(thrp)
    fltr_left = wave_bp <= wmin
    fltr_rght = wave_bp >= wmax
    if np.sum(fltr_left) > 0:
        thrp_left = np.max(thrp[fltr_left])
    else:
        thrp_left = 0
    if np.sum(fltr_rght) > 0:
        thrp_rght = np.max(thrp[fltr_rght])
    else:
        thrp_rght = 0
    thrp_out = np.max([thrp_left, thrp_rght])
    if ((thrp_out/thrp_max) > tol) & (not QuietMode):
        raise Warning("Warning [averageFnu]: There may be significant emission missed due to the wavelength mismatch!")
    #-> Calculate the average flux density
    fltr = (wavelength >= wmin) & (wavelength <= wmax)
    wave = wavelength[fltr]
    flux = flux[fltr]
    thrp = bandpass.sample(wave)
    signal = np.trapz(thrp/wave*flux, x=wave)
    norm   = np.trapz(thrp/wave, x=wave)
    fnu    = signal / norm
    return fnu
