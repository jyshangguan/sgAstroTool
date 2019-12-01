#This code have some functions to help me on photomentry.
#
#The functions are:
#mag2flx(): to convert magnitude to flux
#flx2L(): to convert flux to magnitude
#
from __future__ import division
import numpy as np

Lsun = 3.846e33 #erg/s
Mpc = 3.086e24 #cm

#Func_bgn:
#-------------------------------------#
#	by SGJY, Jul. 15, 2015        #
#-------------------------------------#
def mag2flx(mag, m0, fe):
  L = fe * 10**((mag + m0)/-2.5)
  return L
#Func_end

#Func_bgn:
#-------------------------------------#
#	by SGJY, Jul. 15, 2015        #
#-------------------------------------#
def flx2L(flx, DL):
  L = flx * (4*np.pi) * DL**2. / Lsun
  return L
#Func_end

#Func_bgn:
#-------------------------------------#
#	by SGJY, Jul. 15, 2015        #
#-------------------------------------#
def mag2L(mag, m0, fe, DL):
  flx = mag2flx(mag, m0, fe)
  L = flx * (4*np.pi) * DL**2. / Lsun
  return L
#Func_end
