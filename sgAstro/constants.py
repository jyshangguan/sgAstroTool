from __future__ import division
import numpy as np

#-> Solar units
Msun = 1.98847e33  # g
Lsun = 3.828e33  # erg/s

#-> Light speed
ls_km = 2.99792458e5  # km/s
ls_m = 2.99792458e8  # m/s
ls_cm = 2.99792458e10  # cm/s
ls_micron = 2.99792458e14  # micron/s

#-> Boltzmann constant
k_j   = 1.38064852e-23  # J/K
k_ev  = 8.6173303e-5  # eV/K
k_erg = 1.38064852e-16  # erg/K

#-> Planck constant
h_j   = 6.626070040e-34  # J*s
h_erg = 6.62606957e-27  #erg s
h_ev  = 4.135667662e-15  # eV*s

#-> Time
yr = 3.154e+7  # second
day = 86400.   # second

#-> Distance
pc = 3.086e+18  # cm
kpc = 3.086e+21  # cm
Mpc = 3.086e+24  # cm

#-> Newton G
G_astro = 4.30091e-3  # pc Msun^-1 (km/s)^2

#-> Convertors
J2erg = 1e7 # Joule -> erg
Mpc2cm = 3.086e24  # Mpc -> cm
pc2cm = 3.086e18  # pc -> cm
as2rad = np.pi / 180. / 3600.  # arcsec -> radian
deg2rad = np.pi / 180.  # degree -> radian
Ry2eV = 13.605693009 # Rydberg -> eV
eV2J = 1.60218e-19 # eV -> J
Ry2Hz = Ry2eV * eV2J / h_j # Rydberg -> Hz
