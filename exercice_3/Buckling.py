import numpy as np
import matplotlib.pyplot as plt
import scipy as sci

# Cosine function callback for fitting
# It is the fit function
def cosine_callback(x, A, B):
    return A*np.cos(B*x)

# Get the buckling
def buckling(x, phi):
    # Fit to find the buckling
    para, cov = sci.optimize.curve_fit(cosine_callback, x, phi, p0=[1, 1/50])

    return para