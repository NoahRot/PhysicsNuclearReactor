import numpy as np
from Batch import *

# Build matrix M
def matrix_M(x, delta_x, D, sig_a, sig_s_from_1, sig_s_from_2, show = False):
    # Get the number of mech
    nbr_mech = int(len(x)//2)

    # Create 4 empty square matrices representing the 4 bloc of the total
    # M matrix. Diffusion are the streaming + absorption tridiagonal matrix.
    diffusion_fast  = np.zeros([nbr_mech, nbr_mech])    # Upper left
    diffusion_ther  = np.zeros([nbr_mech, nbr_mech])    # Lower right
    upscattering    = np.zeros([nbr_mech, nbr_mech])    # Upper right
    downscattering  = np.zeros([nbr_mech, nbr_mech])    # Lower left

    # Create fast tridiagonal
    for i in range(nbr_mech):
        # Left BC
        if i == 0:
            beta_right = 2*D[i]*D[i+1]/(D[i] + D[i+1])

            diffusion_fast[i][i]   = -beta_right/np.power(delta_x, 2) - sig_s_from_1[nbr_mech + i] - sig_a[i]
            diffusion_fast[i][i+1] = beta_right/np.power(delta_x, 2)

        # Right BC
        elif i == nbr_mech-1:            
            beta_left  = 2*D[i]*D[i-1]/(D[i] + D[i-1])
            
            BC = 1/(2*(delta_x/(4*D[i])+1))
            diffusion_fast[i][i] = -BC/delta_x - beta_left/np.power(delta_x, 2) - sig_s_from_1[nbr_mech + i] - sig_a[i]
            diffusion_fast[i][i-1] = beta_left/np.power(delta_x, 2)

        # Other  
        else:
            beta_left  = 2*D[i]*D[i-1]/(D[i] + D[i-1])
            beta_right = 2*D[i]*D[i+1]/(D[i] + D[i+1]) 

            diffusion_fast[i][i]   = -(beta_left + beta_right)/np.power(delta_x, 2) - sig_s_from_1[nbr_mech + i] - sig_a[i]
            diffusion_fast[i][i+1] = beta_right/np.power(delta_x, 2)
            diffusion_fast[i][i-1] = beta_left/np.power(delta_x, 2)

    # Create thermal tridiagonal
    for i in range(nbr_mech):
        j = i + nbr_mech

        # Left BC
        if i == 0:
            beta_right = 2*D[j]*D[j+1]/(D[j] + D[j+1])

            diffusion_ther[i][i]   = -beta_right/np.power(delta_x, 2) - sig_s_from_2[i] - sig_a[j]
            diffusion_ther[i][i+1] = beta_right/np.power(delta_x, 2)

        # Right BC
        elif i == nbr_mech-1:            
            beta_left  = 2*D[j]*D[j-1]/(D[j] + D[j-1])
            
            BC = 1/(2*(delta_x/(4*D[j])+1))
            diffusion_ther[i][i] = -BC/delta_x - beta_left/np.power(delta_x, 2) - sig_s_from_2[i] - sig_a[j]
            diffusion_ther[i][i-1] = beta_left/np.power(delta_x, 2)

        # Other  
        else:
            beta_left  = 2*D[j]*D[j-1]/(D[j] + D[j-1])
            beta_right = 2*D[j]*D[j+1]/(D[j] + D[j+1]) 

            diffusion_ther[i][i]   = -(beta_left + beta_right)/np.power(delta_x, 2) - sig_s_from_2[i] - sig_a[j] 
            diffusion_ther[i][i+1] = beta_right/np.power(delta_x, 2)
            diffusion_ther[i][i-1] = beta_left/np.power(delta_x, 2)

        # Upscattering and down scattering
        for i in range(nbr_mech):
            upscattering[i][i]      = sig_s_from_2[i]
            downscattering[i][i]    = sig_s_from_1[i+nbr_mech]

        # Create full matrix by packing all 4 block matrices toghether
        mat = np.block([[diffusion_fast, upscattering], [downscattering, diffusion_ther]])

    # For debugging / getting the matrix coefficients
    if show:
        print("Diffusion fast " + str(diffusion_fast.shape) + "\n", diffusion_fast)
        print("Diffusion ther " + str(diffusion_ther.shape) + "\n", diffusion_ther)
        print("Upscattering " + str(upscattering.shape) + "\n", upscattering)
        print("Downscattering " + str(downscattering.shape) + "\n", downscattering)
    
    return mat

# Build Matrix F
def matrix_F(x, delta_x, nusig_f, show = False):
    # Get the number of mech
    nbr_mech = int(len(x)//2)

    # Two square matrix representing the fast and thermal fission
    fast_fission = np.zeros([nbr_mech, nbr_mech])
    ther_fission = np.zeros([nbr_mech, nbr_mech])

    # Populate the matrices
    for i in range(nbr_mech):
        fast_fission[i][i] = -nusig_f[i]
        ther_fission[i][i] = -nusig_f[i+nbr_mech]

    # Pack the matrices toghether into the full F matrix
    mat = np.block([[fast_fission, ther_fission], [np.zeros([nbr_mech, nbr_mech*2])]])

    # For debugging / getting the matrix coefficients
    if show:
        print("Fast fission \n", fast_fission)
        print("Ther fission \n", ther_fission)

    return mat