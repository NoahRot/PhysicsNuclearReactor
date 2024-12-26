
# A class that contain all parameters of a batch
class Batch:
    def __init__(self, width, D_1, D_2, sig_a_1, sig_a_2, nusig_f_1, nusig_f_2, ksig_f_1, ksig_f_2, sig_s_1_to_1, sig_s_1_to_2, sig_s_2_to_1, sig_s_2_to_2):
        self.width          = width             # Width of the batch
        self.D_1            = D_1               # Diffusion coefficients
        self.D_2            = D_2
        self.sig_a_1        = sig_a_1           # Absorption cross-sections
        self.sig_a_2        = sig_a_2
        self.nusig_f_1      = nusig_f_1         # Fission cross-sections
        self.nusig_f_2      = nusig_f_2
        self.ksig_f_1       = ksig_f_1          # Fission power coefficient
        self.ksig_f_2       = ksig_f_2
        self.sig_s_1_to_1   = sig_s_1_to_1      # Scattering cross-sections
        self.sig_s_1_to_2   = sig_s_1_to_2
        self.sig_s_2_to_1   = sig_s_2_to_1
        self.sig_s_2_to_2   = sig_s_2_to_2