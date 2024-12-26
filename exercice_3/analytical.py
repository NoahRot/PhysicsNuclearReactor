import numpy as np

def analytical_no_ref(A, B, x):
    return A * np.cos(B*x)