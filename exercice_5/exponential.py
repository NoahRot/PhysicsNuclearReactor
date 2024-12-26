import numpy as np

# Return the exponential of a matrix
# (Taylor expansion to 3rd order)
def matrix_exp(mat, order = 3):
    fact = 1
    mat_power = np.identity(mat.shape[0], dtype=mat.dtype)
    exp = np.identity(mat.shape[0], dtype=mat.dtype)

    for i in range(1, order + 1):
        fact *= i
        mat_power = np.dot(mat_power, mat)
        exp += mat_power/fact

    return exp

# Exponential method solver
def exponential(nbr_step, dt, N_0, A):
    exp_A = matrix_exp(A*dt, 3)
    N_exp = np.zeros([nbr_step+1, len(N_0)])
    N_exp[0,:] = N_0

    for i in range(nbr_step):
        N_exp[i+1,:] = np.dot(exp_A, N_exp[i,:])

    return N_exp

    