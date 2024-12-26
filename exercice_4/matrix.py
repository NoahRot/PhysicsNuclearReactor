import numpy as np
from Batch import *

# M matrix
def matrix_M(x, delta_x, D, sig_a, sig_s_from_1, sig_s_from_2, show = False):
    nbr_mech = int(len(x)//2)
    diffusion_fast = np.zeros([nbr_mech, nbr_mech])
    diffusion_ther = np.zeros([nbr_mech, nbr_mech])
    upscattering = np.zeros([nbr_mech, nbr_mech])
    downscattering = np.zeros([nbr_mech, nbr_mech])

    # Create fast diffusion
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

    # Create thermal diffucion
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

        # Create full matrix
        mat = np.block([[diffusion_fast, upscattering], [downscattering, diffusion_ther]])

    if show:
        print("Diffusion fast " + str(diffusion_fast.shape) + "\n", diffusion_fast)
        print("Diffusion ther " + str(diffusion_ther.shape) + "\n", diffusion_ther)
        print("Upscattering " + str(upscattering.shape) + "\n", upscattering)
        print("Downscattering " + str(downscattering.shape) + "\n", downscattering)
    
    """
    diag = np.diagonal(mat)
    diag_up = np.diagonal(mat, offset=1)
    diag_low = np.diagonal(mat, offset=-1)

    
    A = open("Mat_A.txt", "r")
    A = A.readlines()
    A_mat = []
    for i in range(len(A)):
        tmp = A[i].split(" ")
        tmp[-1] = tmp[-1][:-1]
        current = 0
        A_mat.append([])
        for j in range(len(tmp)):
            if tmp[j] != "":
                current += 1
                A_mat[i].append(float(tmp[j]))
                
    A_mat = np.array(A_mat)
    print("A_mat shape =", A_mat.shape)

    A_comparison = np.array([diag, np.diagonal(A_mat), np.diagonal(A_mat) + diag, 
                             np.append(0, diag_up),  np.append(0, np.diagonal(A_mat, offset=1)),  np.append(0, np.diagonal(A_mat, offset=1) + diag_up), 
                             np.append(0, diag_low), np.append(0, np.diagonal(A_mat, offset=-1)), np.append(0, np.diagonal(A_mat, offset=-1) + diag_low)])
    A_comparison = np.array([np.diagonal(A_mat) + diag, 
                             np.append(0, np.diagonal(A_mat, offset=1) + diag_up), 
                             np.append(0, np.diagonal(A_mat, offset=-1) + diag_low)])
    np.savetxt("comparison.csv", np.transpose(A_comparison), delimiter=";")

    #print(mat)
    np.savetxt("A.csv", mat, delimiter=";")    
    """
    
    return mat

# Matrix F
def matrix_F(x, delta_x, nusig_f, show = False):
    nbr_mech = int(len(x)//2)
    fast_fission = np.zeros([nbr_mech, nbr_mech])
    ther_fission = np.zeros([nbr_mech, nbr_mech])

    for i in range(nbr_mech):
        fast_fission[i][i] = -nusig_f[i]
        ther_fission[i][i] = -nusig_f[i+nbr_mech]

    mat = np.block([[fast_fission, ther_fission], [np.zeros([nbr_mech, nbr_mech*2])]])

    if show:
        print("Fast fission \n", fast_fission)
        print("Ther fission \n", ther_fission)

    return mat