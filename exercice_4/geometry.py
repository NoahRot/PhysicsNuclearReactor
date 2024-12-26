import numpy as np
from Batch import *

# Function that build the geometry of the reactor
def build_geometry(x, nbr_mech : int, width : float, list_batch : list[Batch]):
    
    # Initial values at 0
    D_fast              = np.zeros_like(x)  # Fast diffusion coefficient
    D_ther              = np.zeros_like(x)  # Thermal diffusion coefficient
    sig_a_fast          = np.zeros_like(x)  # Absorption cross-section fast
    sig_a_ther          = np.zeros_like(x)  # Absorption cross-section thermal
    nusig_f_fast        = np.zeros_like(x)  # Fission cross-section fast
    nusig_f_ther        = np.zeros_like(x)  # Fission cross-section thermal
    ksig_f_fast         = np.zeros_like(x)  # Fission power coefficient fast
    ksig_f_ther         = np.zeros_like(x)  # Fission power coefficient thermal
    sig_s_from_1_fast   = np.zeros_like(x)  # Scattering from fast to fast
    sig_s_from_1_ther   = np.zeros_like(x)  # Scattering from fast to ther
    sig_s_from_2_fast   = np.zeros_like(x)  # Scattering from ther to fast
    sig_s_from_2_ther   = np.zeros_like(x)  # Scattering from ther to ther

    # For each mech, get the value of the current batch
    for i in range(len(x)):

        # Index of the current batch
        id_batch = int(x[i]//width)

        # Write each elements
        D_fast[i]               = list_batch[id_batch].D_1
        D_ther[i]               = list_batch[id_batch].D_2
        sig_a_fast[i]           = list_batch[id_batch].sig_a_1
        sig_a_ther[i]           = list_batch[id_batch].sig_a_2
        nusig_f_fast[i]         = list_batch[id_batch].nusig_f_1
        nusig_f_ther[i]         = list_batch[id_batch].nusig_f_2
        ksig_f_fast[i]          = list_batch[id_batch].ksig_f_1
        ksig_f_ther[i]          = list_batch[id_batch].ksig_f_2

        sig_s_from_1_fast[i]    = list_batch[id_batch].sig_s_1_to_1
        sig_s_from_1_ther[i]    = list_batch[id_batch].sig_s_1_to_2
        sig_s_from_2_fast[i]    = list_batch[id_batch].sig_s_2_to_1
        sig_s_from_2_ther[i]    = list_batch[id_batch].sig_s_2_to_2

    # Packing the two groups toghether in one array (for each parameter)
    x = np.append(x,x)
    D = np.append(D_fast, D_ther)
    sig_a = np.append(sig_a_fast, sig_a_ther)
    nusig_f = np.append(nusig_f_fast, nusig_f_ther)
    ksig_f = np.append(ksig_f_fast, ksig_f_ther)

    sig_s_from_1 = np.append(np.zeros(nbr_mech), sig_s_from_1_ther)
    sig_s_from_2 = np.append(sig_s_from_2_fast, np.zeros(nbr_mech))

    return x, D, sig_a, nusig_f, ksig_f, sig_s_from_1, sig_s_from_2
