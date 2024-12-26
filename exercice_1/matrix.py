import numpy as np

# Compute matrix for finit element method
def create_matrix(nbr_step, delta_x, D, Sigma_a, show = False):

    # Beta factor (all beta are the same)
    beta = 2*D*D/(D+D)

    # Create an empty matrix
    mat = np.zeros([int(nbr_step), int(nbr_step)])

    # Populate the matrix
    for i in range(int(nbr_step)):

        if i == 0:
            # Boundary condition at left (i-1/2)
            mat[i][i] = -beta/np.power(delta_x, 2) - Sigma_a
            mat[i][i+1] = beta/np.power(delta_x, 2)

        elif i == int(nbr_step)-1:
            # Boundary condition at right (i+1/2)
            BC = 1/(2*(delta_x/(4*D)+1))
            mat[i][i] = -BC/delta_x - beta/np.power(delta_x, 2) - Sigma_a
            mat[i][i-1] = beta/np.power(delta_x, 2)

        else:
            # Other part of the matrix
            mat[i][i] = -2*beta/np.power(delta_x, 2) - Sigma_a
            mat[i][i+1] = beta/np.power(delta_x, 2)
            mat[i][i-1] = beta/np.power(delta_x, 2)

    # Print matrix component
    if show:
        print("="*10 + " Matrix components " + "="*10)
        print("A i,i = " + str(-2*beta/np.power(delta_x, 2) - Sigma_a))
        print("A i,i-1 = A i,i+1 = " + str(beta/np.power(delta_x, 2)))
        print("Size", len(mat))
    
        print(mat)

    # Invert matrix
    mat = np.linalg.inv(mat)
    
    return mat