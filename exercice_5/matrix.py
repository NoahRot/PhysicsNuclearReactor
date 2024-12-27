import numpy as np

from Element import *

# Get the matrix for a constant flux Phi
def matrix_A_const_phi(phi, U_235 : Element, U_238 : Element, Pu_239 : Element, X : Element, Y : Element):

    A = np.zeros([5,5])
    A[0][0] = -phi*(U_235.sigma_c + U_235.sigma_f)
    A[1][1] = -phi*(U_238.sigma_c + U_238.sigma_f)
    A[2][1] = phi*U_238.sigma_c
    A[2][2] = -phi*(Pu_239.sigma_c + Pu_239.sigma_f)
    A[3][0] = phi*U_235.sigma_f*X.FY
    A[3][3] = -(X.decay_cst() + phi*X.sigma_c)
    A[4][0] = phi*U_235.sigma_f*Y.FY
    A[4][4] = -phi*Y.sigma_c

    return A

# Get the matrix for a constant power P
def matrix_A_const_P(phi, U_235 : Element, U_238 : Element, Pu_239 : Element, X : Element, Y : Element):

    A = np.zeros([5,5])
    A[0][0] = -phi*(U_235.sigma_c + U_235.sigma_f)
    A[1][1] = -phi*(U_238.sigma_c + U_238.sigma_f)
    A[2][1] = phi*U_238.sigma_c
    A[2][2] = -phi*(Pu_239.sigma_c + Pu_239.sigma_f)
    A[3][0] = phi*U_235.sigma_f*X.FY
    A[3][1] = phi*U_238.sigma_f*X.FY
    A[3][2] = phi*Pu_239.sigma_f*X.FY
    A[3][3] = -(X.decay_cst() + phi*X.sigma_c)
    A[4][0] = phi*U_235.sigma_f*Y.FY
    A[4][1] = phi*U_238.sigma_f*Y.FY
    A[4][2] = phi*Pu_239.sigma_f*Y.FY
    A[4][4] = -phi*Y.sigma_c

    return A