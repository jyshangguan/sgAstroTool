import numpy as np

__all__ = ["Planck_Function"]

h = 6.62606957e-27 #erg s
c = 29979245800.0 #cm/s
k = 1.3806488e-16 #erg/K

def Planck_Function(nu, T=1e4):
    '''
    This is a single Planck function.
    The input parameter are frequency (nu) with unit Hz and temperature (T) with unit K.
    The system of units is Gaussian units, thus the brightness has unit erg/s/cm^2/Hz/ster.
    '''
    Bnu = 2.0*h*nu**3.0/c**2.0 / (np.exp(h*nu/k/T) - 1.0)
    return Bnu
