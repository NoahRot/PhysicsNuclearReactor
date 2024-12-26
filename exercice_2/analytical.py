import numpy as np

# Analytical solution (without refractor)
def analytical_no_ref(A, B, x):
    return A * np.cos(B*x)