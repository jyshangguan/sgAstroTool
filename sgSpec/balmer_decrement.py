import numpy as np
import extinction as e
__all__ = ["eBmV_dc", "Alamda", "Tau_Lam", "Factor_obs2int"]

#-> The dict containing all the extinction curves.
eDict = {
    "ccm89": e.ccm89,
    "odonnell94": e.odonnell94,
    "calzetti00": e.calzetti00,
    "fitzpatrick99": e.fitzpatrick99,
}

wave_Ha = 6563.0
wave_Hb = 4861.0

def eBmV_dc(ld_obs, ld_int=3.1, wave_l1=wave_Ha, wave_l2=wave_Hb, Rv=3.1, ecname="ccm89"):
    """
    Calculate the E(B-V) using line decrement, presumably the Balmer decrement.

    Parameters
    ----------
    ld_obs : float
        The observed line ratio, presumably the observed ratio of Halpha and Hbeta.
    ld_int : float, default: 3.1 (for AGN)
        The intrinsic line ratio, presumably the intrinsic ratio of Halpha and Hbeta.
    wave_l1 : float, default: 6563
        The wavelength of the longer line, presumably Halpha, in units AA.
    wave_l2 : float, default: 4861
        The wavelength of the shorter line, presumably Hbeta, in units AA.
    Rv : float, default: 3.1
        Rv = Av / E(B-V)
    ecname : string, default: ccm89
        The name of the extinction curves:
            "ccm89": Cardelli, Clayton & Mathis (1989);
            "odonnell94": O'Donnell (1994);
            "calzetti00": Calzetti (2000);
            "fitzpatrick99": Fitzpatrick (1999).

    Returns
    -------
    ebmv : float
        The E(B-V) estimated from the line decrement.

    Notes
    -----
    Fully checked in Halpha and Hbeta case.
    """
    extFunc = eDict[ecname]
    kappa_l1 = Rv * extFunc(np.atleast_1d(wave_l1), a_v=1., r_v=Rv)[0]
    kappa_l2 = Rv * extFunc(np.atleast_1d(wave_l2), a_v=1., r_v=Rv)[0]
    ebmv = -2.5 / (kappa_l1 - kappa_l2) * np.log10(ld_obs / ld_int)
    return ebmv

def Alamda(wavelength, eBmV, Rv=3.1, ecname="ccm89"):
    """
    Calculate the extinction in magnitude (A_lambda) given the E(B-V) and Rv.

    Parameters
    ----------
    wavelength: float
        The wavelength to calculate the extinction, in units AA.
    eBmV : float
        E(B-V)
    Rv : float, default: 3.1
        Rv = Av / E(B-V)
    ecname : string, default: ccm89
        The name of the extinction curves:
            "ccm89": Cardelli, Clayton & Mathis (1989);
            "odonnell94": O'Donnell (1994);
            "calzetti00": Calzetti (2000);
            "fitzpatrick99": Fitzpatrick (1999).

    Returns
    -------
    a_lam : float
        The extinction in magnitude, A_lambda.

    Notes
    -----
    None.
    """
    extFunc = eDict[ecname]
    Av = eBmV * Rv
    a_lam = extFunc(np.atleast_1d(wavelength).astype(np.float), a_v=1., r_v=Rv)[0] * Av
    return a_lam

def Tau_Lam(a_lam):
    """
    Convert Av to the optical depth.

    Parameters
    ----------
    a_lam : float
        The extinction in magnitude, A_lambda.

    Returns
    -------
    tau_lam : float
        The optical depth.

    Notes
    -----
    None.
    """
    tau_lam = a_lam / (2.5 * np.log10(np.e))
    return tau_lam

def Factor_obs2int(a_lam):
    """
    Calculate the factor to correct the observed flux or luminosity to the intrinsic
    one.

    Parameters
    ----------
    a_lam : float
        The extinction in magnitude, A_lambda.

    Returns
    -------
    f : float
        The correction factor to convert the observed values to the intrinsic ones.

    Notes
    -----
    None.
    """
    f = 10**(0.4 * a_lam)
    return f

if __name__ == "__main__":
    bd  = 5.17
    wO3 = 5007
    ebmv = eBmV_dc(bd)
    alam = Alamda(wO3, ebmv)
    taulam = Tau_Lam(alam)
    f = Factor_obs2int(alam)
    print ebmv, f, np.log10(f)
    print np.log10(np.exp(taulam))
