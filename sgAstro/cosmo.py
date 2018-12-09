##
# The code is firstly developed by SGJY in KIAA-PKU at 2018-5-10.
# HubbleFlowDistance() is developed in:
#   http://localhost:8888/notebooks/Softwares/my_module/dev_Hubble%20flow%20velocity%20correction.ipynb
##

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
__all__ = ["z2DL", "v_LocalGroup", "Attractor", "v_Cosmic",
           "HubbleFlowDistance"]

ls_km   = 2.99792458e5 # km/s
deg2rad = np.pi / 180.

def z2DL(z, H0=67.8, Om0=0.308, verbose=True):
    '''
    This function calculate the luminosity distance from the redshift.
    The default cosmology comes from Planck Collaboration (2015).

    Parameters
    ----------
    z : float
        The redshift
    H0 : float
        The Hubble constant, default: 67.8
    Om0 : float
        Omega_0, default: 0.308

    Returns
    -------
    DL : float
        The luminosity distance, unit: Mpc.

    Notes
    -----
    None.
    '''
    if (np.sum(z < 0.02) > 0) & verbose:
        print("There are redshifts too small. The correction for peculiar velocity is necessary!")
        print("The function HubbleFlowDistance() can be used.")
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    DL = cosmo.luminosity_distance(z).value #Luminosity distance in unit Mpc.
    return DL

def ApexConversionCoefficient(v_a, l_a, b_a):
    """
    The conversion coefficient of heliocentric radial velocities to
    a solar apex of direction (l_a, b_a) and amplitude v_a in any
    reference frame can be expressed as:
        v_corr = v_h + v_a * [cos(b) cos(b_a) cos(l - l_a) + sin(b) sin(b_a)],
    which can be alwayse converted to:
        v_corr = v_h + X * cos(l) cos(b) + Y * sin(l) cos(b) + Z * sin(b).
    Therefore, the coefficients are:
        X = v_a * cos(l_a) * cos(b_a)
        Y = v_a * sin(l_a) * cos(b_a)
        Z = v_a * sin(b_a)

    Parameters
    ----------
    v_a : float
        The the velocity amplitude of the apex, units: km/s.
    l_a, b_a : float, float
        The coordinate (longitude, latitude) of the apex, units: degree.

    Returns
    -------
    [X, Y, Z] : list
        The three coefficients.

    Notes
    -----
    The details are clearly presented in Courteau & van den Bergh, ApJ, 118, 337 (1999).
    """
    l_a = l_a * deg2rad
    b_a = b_a * deg2rad
    X = v_a * np.cos(l_a) * np.cos(b_a)
    Y = v_a * np.sin(l_a) * np.cos(b_a)
    Z = v_a * np.sin(b_a)
    return [X, Y, Z]

def v_LocalGroup(v_h, l, b, v_a=316., l_a=93., b_a=-4., verbose=False):
    """
    Correct the observed heliocentric velocity to the centroid of the Local Group.

    Parameters
    ----------
    v_h : float
        The heliocentric velocity of the target, units: km/s.
    l, b : float, float
        The coordinate (longitude, latitude) of the target, units: degree.
    v_a : float; default: 316.
        The apex velocity of the Local Group, units: km/s.
    l_a, b_a : float, float; default: (93, -4)
        The coordinate (longitude, latitude) of the Local Group apex, units: degree.

    Returns
    -------
    v_lg : float
        The radial velocity with respect to the centroid of the Local Group, units: km/s.

    Notes
    -----
    The default values of Local Group apex is adopted from Karachentsev & Makarov, AJ,
    111, 794 (1996) in order to be consistent with the NASA/IPAC Extragalactic Database.
    """
    X, Y, Z = ApexConversionCoefficient(v_a, l_a, b_a)
    if verbose:
        print "X={0:.4f}, Y={1:.4f}, Z={2:.4f}".format(X, Y, Z)
    l = l * deg2rad
    b = b * deg2rad
    v_lg = v_h + X * np.cos(l) * np.cos(b) + Y * np.sin(l) * np.cos(b) + \
           Z * np.sin(b)
    return v_lg

def ang_distance(ra1, dec1, ra2, dec2):
    """
    Angular distance.

    Parameters
    ----------
    ra1, dec1 : float, float
        The coordinates of the first object, units: degree.
    ra2, dec2 : float, float
        The coordinates of the second object, units: degree.

    Returns
    -------
    theta : degree
        The angular distance, units: radian.

    Notes
    -----
    None.
    """
    c1  = SkyCoord(ra1, dec1, frame='icrs', unit='deg')
    c2  = SkyCoord(ra2, dec2, frame='icrs', unit='deg')
    sep = c1.separation(c2)
    theta = sep.deg * deg2rad
    return theta

def vel_distance(v1, v2, theta):
    """
    Velocity distance.  The Equation (2) of Mould et al. (2000).

    Parameters
    ----------
    v1 : float
        The velocity of the first object, units: km/s.
    v2 : float
        The velocity of the second object, units: km/s.
    theta : float
        The angular distance of the two objects, units: radian.

    Returns
    -------
    roa : float
        The velocity distance, units: km/s.

    Notes
    -----
    None.
    """
    roa = np.sqrt(v1**2 + v2**2 - 2. * v1 * v2 * np.cos(theta))
    return roa

class Attractor(object):
    """
    The attractor introduces the inflow affecting the targets.
    The effect of multiple attractors can be linearly added.

    Parameters
    ----------
    ra, dec : float, float
        The ra and dec of the attractor, units: degree.
    v_helio : float
        The observed mean heliocentric velocity of the attractor, units: km/s.
    v_LG : float
        The Velocity corrected to the centroid of the Local Group, units: km/s.
    v_fid : float
        Adopted model infall velocity at the position of the LG, units: km/s.
    radius : float
        Assumed cluster radius, units: degree.
    v_range: [float, float]
        Velocity range collapsed for the cluster core (heliocentric), units: km/s.
        The radius and range give the partial cone that is zeroed to the attractor
        center in the flow-field program.
    """
    def __init__(self, ra, dec, v_helio, v_LG, v_fid, radius, v_range):
        self.ra = ra
        self.dec = dec
        self.v_helio = v_helio
        self.v_LG    = v_LG
        self.v_fid   = v_fid
        self.radius  = radius
        self.v_range = v_range

    def flag_onstream(self, v_LG, ra, dec):
        """
        Flag whether the object is on the stream of the attractor.

        Parameters
        ----------
        v_LG : float
            The velocity of the object with respect to the local group, units: km/s.
        ra, dec : float, float
            The coordinates of the object, units: degree.

        Returns
        -------
        flag : bool
            Flag whether the object is on the stream (True) or not (False).

        Notes
        -----
        None.
        """
        theta = ang_distance(self.ra, self.dec, ra, dec)
        r0a   = vel_distance(self.v_LG, v_LG, theta)
        if (theta < self.radius) and ((self.v_LG - r0a) > self.v_range[0]) \
           and ((self.v_LG + r0a) < self.v_range[1]):
            flag = True
        else:
            flag = False
        return flag

    def v_Infall(self, v_LG, ra, dec, gamma=2, brute=False):
        """
        The infall velocity of the object due to this attractor.

        Parameters
        ----------
        v_LG : float
            The velocity of the object with respect to the local group, units: km/s.
        ra, dec : float, float
            The coordinates of the object, units: degree.
        gamma : float
            The slope of the attractor's density profile.
        brute : bool; default: False
            Return the v_infall without considering the stream area.

        Returns
        -------
        v_inf : bool
            The infall velocity, units: km/s.

        Notes
        -----
        None.
        """
        theta = ang_distance(self.ra, self.dec, ra, dec)
        r0a   = vel_distance(self.v_LG, v_LG, theta)
        v_inf = self.v_fid * np.cos(theta) + self.v_fid * \
                (v_LG - self.v_LG * np.cos(theta)) / r0a * \
                (r0a / self.v_LG)**(1 - gamma)
        if brute:
            return v_inf
        if self.flag_onstream(v_LG, ra, dec):
            v_inf = 0
        return v_inf

def v_Cosmic(v_h, ra, dec, verbose=False, brute=False):
    """
    Correct the pecular velocity due to the local structures assuming
    the 3-attractor flow model of Mould et al., ApJ, 529, 786 (2000).

    The conversion to the radial velocity with respect to the centroid
    of the Local Group is based on Karachentsev and Makarov, AJ, 111,
    794 (1996).

    The conversion method adopted here is exactly the same as what used
    in NASA/IPAC Extragalactic Database.

    Parameters
    ----------
    v_h : float
        float
        The heliocentric velocity of the target, units: km/s.
    ra, dec : float, float
        The coordinate (R.A., Dec.) of the target, units: degree.
    verbose : bool; default: False
        Print the details of the results if True.
    brute : bool; default: False
        Return the v_infall without considering the stream area.

    Returns
    -------
    v_cosmic : float
        The cosmic velocity of the target, which can be used to calculate
        the luminosity distance, units: km/s.
    flag_v : int
        The flag of the cosmic velocity:
           -1 -- Normal result.
            0 -- The object is on more than one stream, which is probably not possible...
            1 -- The object is on Virgo stream.
            2 -- The object is on Great Attractor stream.
            3 -- The object is on Shapley Supercluster.
        For the cases except 0, the velocity cannot be used to calculate
        the luminosity distance.

    Notes
    -----
    The v_cosmic is consistent with NED results of V (Virgo + GA + Shapley).
    """
    #-> Virgo
    c_vir   = SkyCoord('12:28:19', '+12:40:00', unit=(u.hourangle, u.deg))
    ra_vir  = c_vir.ra.deg
    dec_vir = c_vir.dec.deg
    virgo   = Attractor(ra_vir, dec_vir, 1035, 957, 200, 10, [600, 2300])

    #-> Great Attractor
    c_ga   = SkyCoord('13:20:00', '-44:00:00', unit=(u.hourangle, u.deg))
    ra_ga  = c_ga.ra.deg
    dec_ga = c_ga.dec.deg
    ga     = Attractor(ra_ga, dec_ga, 4600, 4380, 400, 10, [2600, 6600])

    #-> Shapley Supercluster
    c_shap   = SkyCoord('13:30:00', '-31:00:00', unit=(u.hourangle, u.deg))
    ra_shap  = c_shap.ra.deg
    dec_shap = c_shap.dec.deg
    shap     = Attractor(ra_shap, dec_shap, 13800, 13600, 85, 12, [10000, 16000])

    c_obj = SkyCoord(ra, dec, unit="deg")
    l     = c_obj.galactic.l.deg
    b     = c_obj.galactic.b.deg
    v_LG  = v_LocalGroup(v_h, l, b)
    nameList = ["Virgo", "Great Attractor", "Shapley Supercluster"]
    flagList = [virgo.flag_onstream(v_LG, ra, dec),
                ga.flag_onstream(v_LG, ra, dec),
                shap.flag_onstream(v_LG, ra, dec)]
    if brute:
        infallList = [virgo.v_Infall(v_LG, ra, dec, brute=True),
                      ga.v_Infall(v_LG, ra, dec, brute=True),
                      shap.v_Infall(v_LG, ra, dec, brute=True)]
        v_cosmic = v_LG + np.sum(infallList)
        flag_v = -1
        if verbose:
            print("Brute force calculation...")
            vDetailList = [v_LG + infallList[0],
                           v_LG + infallList[0] + infallList[1],
                           v_cosmic]
            for loop in range(len(infallList)):
                print("  v_infall({0}): {1:.1f}".format(" + ".join(nameList[:loop+1]), vDetailList[loop]))
        return v_cosmic, flag_v
    if np.sum(flagList) == 0:
        infallList = [virgo.v_Infall(v_LG, ra, dec),
                      ga.v_Infall(v_LG, ra, dec),
                      shap.v_Infall(v_LG, ra, dec)]
        v_cosmic = v_LG + np.sum(infallList)
        flag_v = -1
        if verbose:
            print("Normal calculation...")
            vDetailList = [v_LG + infallList[0],
                           v_LG + infallList[0] + infallList[1],
                           v_cosmic]
            for loop in range(len(infallList)):
                print("  v_infall({0}): {1:.1f}".format(" + ".join(nameList[:loop+1]), vDetailList[loop]))
    elif np.sum(flagList) != 1:
        v_cosmic = np.nan
        flag_v   = 0
        if verbose:
            print("The object is on more than one stream...")
    else:
        vlgList  = [virgo.v_LG, ga.v_LG, shap.v_LG]
        idx = flagList.index(1)
        v_cosmic = vlgList[idx]
        flag_v = idx + 1
        if verbose:
            print("The object is on {0}".format(nameList[idx]))
    return v_cosmic, flag_v

def HubbleFlowDistance(z, ra, dec, H0=67.8, Om0=0.308, verbose=False):
    """
    The Hubble flow distance calculated with the cosmic velocity
    of the target after correcting the peculiar velocity assuming
    a 3-attractor flow model of Mould et al., ApJ, 529, 786 (2000).

    The conversion to the radial velocity with respect to the centroid
    of the Local Group is based on Karachentsev and Makarov, AJ, 111,
    794 (1996).

    The conversion method adopted here is exactly the same as what used
    in NASA/IPAC Extragalactic Database, although we use the Planck2015
    cosmology by default.

    Parameters
    ----------
    z : float
        Redshift.
    ra, dec : float, float
        The coordinate (R.A., Dec.) of the target, units: degree.
    H0 : float; default: 67.8
        The Hubble constant.
    Om0 : float; default: 0.308
        The dimensionless density of the baryonic matter, Omega_0.
    verbose : bool; default: False
        Print the details of the results if True.

    Returns
    -------
    d_hf : float
        The Hubble flow distance, units: Mpc.
    flag_v : int
        The flag of the cosmic velocity:
           -1 -- Normal result.
            0 -- The object is on more than one stream, which is probably not possible...
            1 -- The object is on Virgo stream.
            2 -- The object is on Great Attractor stream.
            3 -- The object is on Shapley Supercluster.
        For the cases except 0, the velocity cannot be used to calculate
        the luminosity distance.

    Notes
    -----
    None.
    """
    v_h = ls_km * z # Convert to heliocentric velocity
    v_cosmic, flag_v = v_Cosmic(v_h, ra, dec, verbose)
    if flag_v == -1:
        z_cosmic = v_cosmic / ls_km
        d_hf = z2DL(z_cosmic, H0=H0, Om0=Om0, verbose=False) #Luminosity distance in unit Mpc.
    elif flag_v >0:
        v_cosmic, flag_u = v_Cosmic(v_h, ra, dec, verbose, brute=True)
        z_cosmic = v_cosmic / ls_km
        d_hf = z2DL(z_cosmic, H0=H0, Om0=Om0, verbose=False)
        if verbose:
            print("**The object is on No.{0} attractor.".format(flag_v))
            print("**We ignore the assumption that the object should follow the stream, made by Mould et al. (2000).")
            print("**But this velocity should be close to the NED value.")
    else:
        d_hf = np.nan
        if verbose:
            print("The object is on more than one attractors.")
    return d_hf, flag_v
