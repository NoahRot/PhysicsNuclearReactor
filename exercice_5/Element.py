import numpy as np

# Element class. Contain the parameters of an element 
class Element:
    def __init__(self, sig_c, sig_s, nu, sig_f, kappa, T, FY):
        self.sigma_c = sig_c    # Capture cross-section
        self.sigma_s = sig_s    # Scattering cross-section
        self.sigma_f = sig_f    # Fission cross-section
        self.nu = nu            # Mean number of neutron per fission
        self.kappa = kappa      # Energy per fission
        self.T = T              # Half life
        self.FY = FY            # Fission yield

    # Return the decay constant
    def decay_cst(self):
        return np.log(2)/self.T