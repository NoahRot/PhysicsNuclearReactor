import numpy as np
import scipy.linalg as lin

# Return the exponential of a matrix
# (Taylor expansion to 3rd order)
def matrix_exp(mat, order = 3):
    fact = 1                                                # Factorial
    mat_power = np.identity(mat.shape[0], dtype=mat.dtype)  # Matrix power tracker
    exp = np.identity(mat.shape[0], dtype=mat.dtype)        # Result of exponential

    # Taylor expansion
    for i in range(1, order + 1):
        fact *= i                           # Update factorial
        mat_power = np.dot(mat_power, mat)  # Update matrix power
        exp += mat_power/fact               # Add element to taylor serie
    
    return exp

# Exponential method solver
def exponential(nbr_step, dt, N_0, A):
    exp_A = matrix_exp(A*dt, 3)                 # Matrix exponential
    N_exp = np.zeros([nbr_step+1, len(N_0)])    # Results
    N_exp[0,:] = N_0                            # Put initial values inside results

    # Iteration
    for i in range(nbr_step):
        N_exp[i+1,:] = np.dot(exp_A, N_exp[i,:])

    return N_exp

    