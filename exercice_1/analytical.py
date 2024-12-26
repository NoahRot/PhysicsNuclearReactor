import numpy as np

# Compute analytical solution
def analytical(S, L, D, a, x):
    return (S*L)/(2*D)*np.sinh((a-2*np.abs(x))/(2*L))/np.cosh(a/(2*L))