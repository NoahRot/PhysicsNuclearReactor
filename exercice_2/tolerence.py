import numpy as np

# Check if the difference between the previous computation and the current one is smaller
# than the tolerate value.
def check_tolerence(k_1, k_2, tol_k, phi_1, phi_2, tol_phi):
    return np.abs((k_1 - k_2)/k_1) > tol_k or np.max(np.abs((phi_1 - phi_2)/phi_1)) > tol_phi