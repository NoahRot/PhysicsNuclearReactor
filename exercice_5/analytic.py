import numpy as np
import matplotlib.pyplot as plt
import scipy as sci

from Element import *

"""
Return the analytical solution for all isotopes
"""

# Analytic solution for Uranium 235/238
def analytic_U(t : float, n_0 : float, phi : float, U : Element):
    a5 = phi*(U.sigma_c + U.sigma_f)
    return n_0*np.exp(-a5*t)

# Analytic solution of Plutonium 239
def analytic_Pu(t : float, phi : float, n_U_238 : float, Pu : Element, U_238 : Element):
    aPu     = phi*(Pu.sigma_c + Pu.sigma_f)
    bPu     = phi*U_238.sigma_c*n_U_238
    aU238   = phi*(U_238.sigma_c + U_238.sigma_f)

    return (np.exp(-aU238*t) - np.exp(-aPu*t))*bPu/(aPu - aU238)

# Analytic solution of F.P. X
def analytic_X(t : float, phi : float, n_U_235 : float, X : Element, U_235 : Element):
    aX      = phi*X.sigma_c + X.decay_cst()
    bX      = phi*U_235.sigma_f*X.FY*n_U_235
    aU235   = phi*(U_235.sigma_c + U_235.sigma_f)

    return (np.exp(-aU235*t) - np.exp(-aX*t))*bX/(aX - aU235)

# Analytic solution of F.P. Y
def analytic_Y(t : float, phi : float, n_U_235 : float, Y : Element, U_235 : Element):
    aY      = phi*Y.sigma_c
    bY      = phi*U_235.sigma_f*Y.FY*n_U_235
    aU235   = phi*(U_235.sigma_c + U_235.sigma_f)

    return (np.exp(-aU235*t) - np.exp(-aY*t))*bY/(aY - aU235)