import numpy as np

def simulation(nbr_mech, M, F, height, delta_x, ksig_f, P_total):
    # === Simulation ===
    max_iteration = int(1e3)
    M_inv = np.linalg.inv(M)

    # Initial guess
    list_k = [] # (just for evolution of k)
    k_last = 1
    k = 1
    list_k.append(k)
    phi_last = np.zeros(int(nbr_mech)*2) + 1
    phi = np.zeros(int(nbr_mech)*2) + 1
    S   = 1/k*np.dot(F, phi)

    # First iteration
    phi = np.dot(M_inv, S)
    k   = k_last*(np.sum(F @ phi))/(np.sum(F @ phi_last))
    list_k.append(k)
    S   = 1/k*np.dot(F, phi)

    # Iterative method for solving eigen problem
    iteration = 1
    while abs(k - k_last) > 1e-7 and iteration <= max_iteration:

        # Store last values
        phi_last = phi
        k_last = k

        # Step
        phi = np.dot(M_inv, S)
        k   = k_last*(np.sum(F @ phi))/(np.sum(F @ phi_last))
        list_k.append(k)
        S   = 1/k*np.dot(F, phi)

        iteration += 1

    #print("Iteration : ", iteration)
    #print("k =", k)

    # Normalisation
    P_fiss = height*delta_x*delta_x*ksig_f*phi
    P_fiss = P_fiss[:nbr_mech] + P_fiss[nbr_mech:]
    P_rel = P_fiss/np.mean(P_fiss)

    phi = phi*P_total*0.5/np.sum(P_fiss)

    return phi, P_fiss, P_rel, k, list_k