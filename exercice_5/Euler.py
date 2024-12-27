import numpy as np
import matplotlib.pyplot as plt
import scipy as sci

from Element import *

# Euler method solver
def Euler(nbr_step, dt, N_0, A):
    N_euler = np.zeros([nbr_step+1, len(N_0)])  # Results matrix
    N_euler[0,:] = N_0                          # Put initial value in the results

    # Euler method
    for i in range(nbr_step):
        N_euler[i+1,:] = np.dot(A,N_euler[i,:])*dt + N_euler[i,:]

    return N_euler