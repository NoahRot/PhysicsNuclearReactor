import numpy as np

# Element class
class Element:
    def __init__(self, sig_c, sig_s, nu, sig_f, kappa, T, FY):
        self.sigma_c = sig_c
        self.sigma_s = sig_s
        self.sigma_f = sig_f
        self.nu = nu
        self.kappa = kappa
        self.T = T
        self.FY = FY

    # Return the decay constant
    def decay_cst(self):
        return np.log(2)/self.T