from __future__ import division
import ezgal
from scipy.optimize import fsolve

__all__ = ["age2zf", "extractSED"]

def age2zf(age, zs, model, zf_ini=None, zf_max=20., *args, **kwargs):
    """
    Calculate the zf (formation redshift) based on the input age.

    Parameters
    ----------
    age : float
      The age of the galaxy.
    zs : float
      The redshift of the source.
    model : EzGal
        The EzGal model.
    zf_ini (optional) : float; default: 10.0 Gyr
        The initial guess of the formation redshift, unit: Gyr.
    zf_max (optional) : float; default: 20.0 Gyr
        The maximum redshift allow to calculate.
    other parameters for fsolve()

    Returns
    -------
    zf : float
        The formation redshift.

    Notes
    -----
    This is a small tool.  Not intentionally to be comprehensive.
    """
    if zf_ini is None:
        zf_ini = zs + 0.5
    assert (zs < zf_ini) & (zf_ini < zf_max)
    age_lim = model.get_age(zf_max, zs)
    if age > age_lim:
        raise ValueError("The age ({0}) Gyr is inconsistent with the zs ({1})!".format(age, zs))
    fun = lambda z: model.get_age(z, zs) - age
    res = fsolve(fun, zf_ini, *args, **kwargs)
    zf  = res[0]
    return zf

def extractSED(age, mass=None, model_name="bc03_ssp_z_0.02_chab.model",
               Om=0.308, Ol=0.692, h=0.678):
    """
    Generate the SED of stellar emission using EzGal. The output spectrum
    is at 10 pc.

    Parameters
    ----------
    age : float
        The age of the galaxy, unit: Gyr.
    mass (optional) : float
        The stellar mass required for the SED, unit: solar mass.
    model_name : str; default: BC03 SSP solar metallicity Chabrier IMF.
        The name of the model.
    Om : float; default: 0.308
        The cosmological parameter, Omega matter.
    Ol : float; default: 0.692
        The cosmological parameter, Omega lambda.
    h : float; default: 0.678
        The Hubble constant normalized by 100 km/s/Mpc.

    Returns
    -------
    wave : ndarray
        The wavelength of the SED, unit: angstrom.
    flux : ndarray
        The flux of the SED, unit: erg/s/cm^2/Hz.

    Notes
    -----
    None.
    """
    model = ezgal.ezgal(model_name)
    model.set_cosmology(Om=Om, Ol=Ol, h=h)
    model.add_filter( 'sloan_i' ) #Have to specify an arbitrary band.
    flux_org = model.get_sed(age, age_units="gyrs", units='Fv')
    zs = 0.0 #Have to start from an arbitrary zs.
    zf = age2zf(age, zs, model)
    mass_org = model.get_masses(zf, zs)
    if mass is None:
        mass = mass_org
    wave = model.ls
    flux = flux_org * mass / mass_org
    wave = wave[::-1] #To make it start from small wavelength
    flux = flux[::-1]
    return wave, flux
