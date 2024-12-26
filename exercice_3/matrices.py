import numpy as np

def matrix_M_no_ref(nbr_mech, nbr_group, delta_x, D, Sigma_1_to_2, Sigma_a):

    # Create an empty matrix
    mat = np.zeros([int(nbr_mech*nbr_group), int(nbr_mech*nbr_group)])

    # Populate the matrix
    for i in range(nbr_mech*nbr_group):
        
        # Diffusion part

        # Left BC
        if i % int(nbr_mech) == 0:
            beta_right = 2*D[i]*D[i+1]/(D[i] + D[i+1])

            mat[i][i]   = -beta_right/np.power(delta_x, 2) - Sigma_1_to_2[i] - Sigma_a[i]
            mat[i][i+1] = beta_right/np.power(delta_x, 2)

        # Right BC
        elif i % int(nbr_mech) == nbr_mech-1:
            beta_left  = 2*D[i]*D[i-1]/(D[i] + D[i-1])
            
            BC = 1/(2*(delta_x/(4*D[i])+1))
            mat[i][i] = -BC/delta_x - beta_left/np.power(delta_x, 2) - Sigma_1_to_2[i] - Sigma_a[i]
            mat[i][i-1] = beta_left/np.power(delta_x, 2)

        # Other  
        else:
            beta_left  = 2*D[i]*D[i-1]/(D[i] + D[i-1])
            beta_right = 2*D[i]*D[i+1]/(D[i] + D[i+1]) 

            mat[i][i]   = -(beta_left + beta_right)/np.power(delta_x, 2) - Sigma_1_to_2[i] - Sigma_a[i]
            mat[i][i+1] = beta_right/np.power(delta_x, 2)
            mat[i][i-1] = beta_left/np.power(delta_x, 2)

        # Transfert from a group to another

        # From group 1 to group 2
        if i//nbr_mech == 1:
            mat[i][i%nbr_mech] = Sigma_1_to_2[i-nbr_mech]

    return mat

def matrix_F_no_ref(nbr_mech, nbr_group, nu_Sigma_f_C):

    # Create an empty matrix
    mat = np.zeros([int(nbr_mech*nbr_group), int(nbr_mech*nbr_group)])

    # Populate the matrix
    for i in range(nbr_mech*nbr_group):
        mat[i%nbr_mech][i] = -nu_Sigma_f_C[i]

    return mat

def matrix_M_ref(nbr_mech, nbr_group, delta_x, D, Sigma_1_to_2, Sigma_a):

    # Create an empty matrix
    mat = np.zeros([int(nbr_mech*nbr_group), int(nbr_mech*nbr_group)])

    # Populate the matrix
    for i in range(nbr_mech*nbr_group):
        
        # Diffusion part

        # Left BC
        if i % int(nbr_mech) == 0:
            beta_right  = 2*D[i]*D[i+1]/(D[i] + D[i+1]) 
            
            BC = 1/(2*(delta_x/(4*D[i])+1))
            mat[i][i] = -BC/delta_x - beta_right/np.power(delta_x, 2) - Sigma_1_to_2[i] - Sigma_a[i]
            mat[i][i+1] = beta_right/np.power(delta_x, 2)

        # Right BC
        elif i % int(nbr_mech) == nbr_mech-1:
            beta_left  = 2*D[i]*D[i-1]/(D[i] + D[i-1])
            
            BC = 1/(2*(delta_x/(4*D[i])+1))
            mat[i][i] = -BC/delta_x - beta_left/np.power(delta_x, 2) - Sigma_1_to_2[i] - Sigma_a[i]
            mat[i][i-1] = beta_left/np.power(delta_x, 2)

        # Other  
        else:
            beta_left  = 2*D[i]*D[i-1]/(D[i] + D[i-1])
            beta_right = 2*D[i]*D[i+1]/(D[i] + D[i+1]) 

            mat[i][i]   = -(beta_left + beta_right)/np.power(delta_x, 2) - Sigma_1_to_2[i] - Sigma_a[i]
            mat[i][i+1] = beta_right/np.power(delta_x, 2)
            mat[i][i-1] = beta_left/np.power(delta_x, 2)

        # Transfert from a group to another

        # From group 1 to group 2
        if i//nbr_mech == 1:
            mat[i][i%nbr_mech] = Sigma_1_to_2[i-nbr_mech]

    np.set_printoptions(precision=3)

    return mat