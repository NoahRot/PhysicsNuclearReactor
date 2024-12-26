import numpy as np

def matrix_M(nbr_mech, delta_x, D, Sigma_a):

    # Create an empty matrix
    mat = np.zeros([int(nbr_mech), int(nbr_mech)])

    # Populate the matrix
    for i in range(nbr_mech):

        # Left BC
        if i == 0:
            beta_right = 2*D[i]*D[i+1]/(D[i] + D[i+1])

            mat[i][i]   = -beta_right/np.power(delta_x, 2) - Sigma_a[i]
            mat[i][i+1] = beta_right/np.power(delta_x, 2)

        # Right BC
        elif i == int(nbr_mech)-1:
            beta_left  = 2*D[i]*D[i-1]/(D[i] + D[i-1])
            
            BC = 1/(2*(delta_x/(4*D[i])+1))
            mat[i][i] = -BC/delta_x - beta_left/np.power(delta_x, 2) - Sigma_a[i]
            mat[i][i-1] = beta_left/np.power(delta_x, 2)

        # Other  
        else:
            beta_left  = 2*D[i]*D[i-1]/(D[i] + D[i-1])
            beta_right = 2*D[i]*D[i+1]/(D[i] + D[i+1]) 

            mat[i][i]   = -(beta_left + beta_right)/np.power(delta_x, 2) - Sigma_a[i]
            mat[i][i+1] = beta_right/np.power(delta_x, 2)
            mat[i][i-1] = beta_left/np.power(delta_x, 2)

    return mat

def matrix_F(nbr_mech, nu_Sigma_f_C):

    # Create an empty matrix
    mat = np.zeros([int(nbr_mech), int(nbr_mech)])

    # Populate the matrix
    for i in range(nbr_mech):
        mat[i][i] = -nu_Sigma_f_C[i]

    return mat