"""
    __  __         _     _ _                                                     _ _    _   _           _                                      _             _      
   |  \/  |___  __| |___| (_)_ _  __ _   __ _   _ __  ___ _ _ ___   _ _ ___ __ _| (_)__| |_(_)__   _ __| |__ _ _ _  __ _ _ _   _ _ ___ __ _ __| |_ ___ _ _  (_)_ _  
   | |\/| / _ \/ _` / -_) | | ' \/ _` | / _` | | '  \/ _ \ '_/ -_) | '_/ -_) _` | | (_-<  _| / _| | '_ \ / _` | ' \/ _` | '_| | '_/ -_) _` / _|  _/ _ \ '_| | | ' \ 
   |_|  |_\___/\__,_\___|_|_|_||_\__, | \__,_| |_|_|_\___/_| \___| |_| \___\__,_|_|_/__/\__|_\__| | .__/_\__,_|_||_\__,_|_|   |_| \___\__,_\__|\__\___/_|   |_|_||_|
   | |___ __ _____   __ _ _ _ ___|___/ _ __ ___                                                   |_|                                                               
   |  _\ V  V / _ \ / _` | '_/ _ \ || | '_ (_-<                                                                                                                     
    \__|\_/\_/\___/ \__, |_| \___/\_,_| .__/__/                                                                                                                     
                    |___/             |_|                                                                                                                           
"""

# Imports 
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci
from itertools import permutations
from Batch import *
from matrix import *
from simulation import *
from geometry import *

# === Matplotlib specification ===
font_size = 16
lw = 1.4

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : font_size}

plt.rc('font', **font)
plt.rc('lines', lw=lw, mew=lw)
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams['text.usetex'] = True

# Read file of Batch Spec
file = open("XS_table.txt", "r")
data = file.readlines()
file.close()

# Extract data from the file
data_extract = []
for line in data:
    # Remove end line char
    if line[-1] == "\n":
        line = line[:-1]

    # Split elements
    line = line.split(" ")

    # To float list
    new_line = []
    for element in line:
        try:
            f = float(element)
            new_line.append(f)
        except:
            if element != "":
                print("Discard :", element)
    
    data_extract.append(new_line)

# Create Batches types
batches = []
width = 20
for i in range(len(data_extract[0])):

    # Batch coefficient extraction
    D_1, D_2 = data_extract[2][i*2 + 0], data_extract[2][i*2 + 1]
    sig_a_1, sig_a_2 = data_extract[3][i*2 + 0], data_extract[3][i*2 + 1]
    nusig_f_1, nusig_f_2 = data_extract[4][i*2 + 0], data_extract[4][i*2 + 1]
    ksig_f_1, ksig_f_2 = data_extract[5][i*2 + 0], data_extract[5][i*2 + 1]

    # Scatter matrix
    sig_s_1_to_1, sig_s_1_to_2 =  data_extract[8][i*2 + 0], data_extract[8][i*2 + 1]
    sig_s_2_to_1, sig_s_2_to_2 =  data_extract[9][i*2 + 0], data_extract[9][i*2 + 1]

    batches.append(Batch(width, D_1, D_2, sig_a_1, sig_a_2, nusig_f_1, nusig_f_2, ksig_f_1, ksig_f_2, sig_s_1_to_1, sig_s_1_to_2, sig_s_2_to_1, sig_s_2_to_2))

# Parameters
P_total = 90e6  # Power of the reactor [W]
height = 400    # Height of the reactor fuel rods [cm]
delta_x = 1     # Mech size [cm]
a = 120         # Radius of the core [cm]
b = 20          # Radius of the relector [cm]
nbr_mech = (a + b)//delta_x # Number of mech
# Space mech for one group
x_one = np.linspace(delta_x*0.5, (a+b)-0.5*delta_x, int(nbr_mech))

# Batch geometry
list_batch : list[Batch] = [batches[2], batches[2], batches[1], batches[1], batches[0], batches[0], batches[3]]

# Build geometry of the reactor
x, D, sig_a, nusig_f, ksig_f, sig_s_from_1, sig_s_from_2 = build_geometry(
    x_one, nbr_mech, width, list_batch)

# Build the matrix
M = matrix_M(x, delta_x, D, sig_a, sig_s_from_1, sig_s_from_2, False)
F = matrix_F(x, delta_x, nusig_f, False)

# Simulation
phi, P_fiss, P_rel, k, list_k = simulation(nbr_mech, M, F, height, delta_x, ksig_f, P_total)

# Print results
print("*"*26)
print("*** Simulation results ***")
print("*"*26)

print("k =", k)
print("Fission power =", np.sum(P_fiss))
print("Phi at x =", x[10], "is", phi[10]*1e-14)
print("Phi at x =", x[nbr_mech+10], "is", phi[nbr_mech+10]*1e-13)
print("P rel at x =", x[10], "is", P_rel[10])
print("Peak power :", np.max(height*delta_x*delta_x*ksig_f*phi))
print("Maxium peaking : ", np.max(P_rel))

# Flux at the boundary of the core
flux_boundary = (phi[120] - phi[121])/delta_x
flux_boundary = (D[121]*phi[121] + D[120]*phi[120])/(D[121] + D[120])
print("Fast Flux at boudary :", flux_boundary*1e-15, " 10^15")

# Plots
# Plot fast and thermal flux
# Create the mirrored data
x_mirror = -x[::-1]  # Reverse X and change sign
phi_mirror = phi[::-1]  # Reverse Phi

# Combine the original and mirrored data
x_full = np.concatenate((x_mirror, x[1:]))  # Exclude the duplicate center point
phi_full = np.concatenate((phi_mirror, phi[1:]))

# Mirror the results to plot the full flux shape
x_fast = x[0:nbr_mech]
phi_fast = phi[0:nbr_mech]
x_fast_mirror = -x_fast[::-1]
phi_fast_mirror = phi_fast[::-1]
x_fast_full = np.concatenate((x_fast_mirror, x_fast[1:]))
phi_fast_full = np.concatenate((phi_fast_mirror, phi_fast[1:]))

x_ther = x[nbr_mech:]
phi_ther = phi[nbr_mech:]
x_ther_mirror = -x_ther[::-1]
phi_ther_mirror = phi_ther[::-1]
x_ther_full = np.concatenate((x_ther_mirror, x_ther[1:]))
phi_ther_full = np.concatenate((phi_ther_mirror, phi_ther[1:]))

fig = plt.figure()
ax = fig.subplots()
ax.plot(x_fast_full, phi_fast_full)
ax.plot(x_ther_full, phi_ther_full)
ax.grid()
ax.set_ylabel("$\Phi$ [n cm$^{-2}$ s$^{-1}$]")
ax.set_xlabel("$x$ [cm]")
ax.legend(["Fast", "Thermal"])

fig.savefig("Ex4_Flux.pdf")

# Plot relative power
# Mirror the results to plot the full power
x_half = x[0:nbr_mech]
x_half_mirror = -x_half[::-1]
x_full = np.concatenate((x_half_mirror, x_half[1:]))

P_rel_mirror = P_rel[::-1]
P_rel_full = np.concatenate((P_rel_mirror, P_rel[1:]))

fig = plt.figure()
ax = fig.subplots()
ax.plot(x_full, P_rel_full)
ax.grid()
ax.set_xlabel("$x$ [cm]")
ax.set_ylabel("$P_{rel}$")

fig.savefig("Ex4_Power.pdf")

# Copy the results of the flux (for comparison with the opti patern)
phi_1 = np.copy(phi)

# ***************************
# *** Optimization patern ***
# ***************************

print("*"*27)
print("*** Optimization patern ***")
print("*"*27)

# Batch index list and all possible permutations
list_batch_index : list[int] = [0, 0, 1, 1, 2, 2]
unique_permutations = set(permutations(list_batch_index))

# Store valid permutations
list_valid_perm = []

# Store results for the best pattern
best_perm = None
best_k = 0
best_phi = None
best_D = None
best_power_rel = None
best_ksig_f = None

# Simulation on each permutation
count = 1

for perm in unique_permutations:

    # Print how many simulations have been done
    if count%10 == 0:
        print("Simulation", count, "/", len(unique_permutations))
    count += 1
    
    # Batch geometry
    list_batch : list[Batch] = [
        batches[perm[0]], 
        batches[perm[1]],
        batches[perm[2]],
        batches[perm[3]],
        batches[perm[4]],
        batches[perm[5]],
        batches[3]
    ]

    # Build geometry of the reactor
    x, D, sig_a, nusig_f, ksig_f, sig_s_from_1, sig_s_from_2 = build_geometry(
        x_one, nbr_mech, width, list_batch)
    
    # Build the matrix
    M = matrix_M(x, delta_x, D, sig_a, sig_s_from_1, sig_s_from_2)
    F = matrix_F(x, delta_x, nusig_f)

    # Simulation
    phi, P_fiss, P_rel, k, list_k = simulation(nbr_mech, M, F, height, delta_x, ksig_f, P_total)

    # Check if the conditions are verify
    # flux < 10^13 and relative power < 3
    if phi[nbr_mech-1] + phi[2*nbr_mech-1] < 1e13 and np.max(P_rel) < 3.0:

        # Add the permutation as a valid one
        list_valid_perm.append(perm)

        # Check if current permutation better than the previous better
        # If so replace the best results by the current results
        if best_perm is None or best_k < k:
            best_perm = perm
            best_k = k
            best_phi = np.copy(phi)
            best_D = np.copy(D)
            best_power_rel = np.copy(P_rel)
            best_ksig_f = np.copy(ksig_f)

# Compute the flux at the boudrary between core and reflector
#flux_boundary = (best_D[121]*best_phi[121] + best_D[120]*best_phi[120])/(best_D[121] + best_D[120])
flux_boundary = best_phi[nbr_mech-1] + best_phi[2*nbr_mech-1]

# Print results
print("Valid permutations :", list_valid_perm)
print("Best permutation :", perm)
print("Best k =", k)
print("Peak power : ", np.max(height*delta_x*delta_x*best_ksig_f*best_phi))
print("Fast flux at the core boundary : ", flux_boundary*1e-13, " 10^13")
print("Phi peak :", np.max(phi))

# Plots
# Plot the flux

x_fast = x[0:nbr_mech]
phi_fast = best_phi[0:nbr_mech]
x_fast_mirror = -x_fast[::-1]
phi_fast_mirror = phi_fast[::-1]
x_fast_full = np.concatenate((x_fast_mirror, x_fast[1:]))
phi_fast_full = np.concatenate((phi_fast_mirror, phi_fast[1:]))

x_ther = x[nbr_mech:]
phi_ther = best_phi[nbr_mech:]
x_ther_mirror = -x_ther[::-1]
phi_ther_mirror = phi_ther[::-1]
x_ther_full = np.concatenate((x_ther_mirror, x_ther[1:]))
phi_ther_full = np.concatenate((phi_ther_mirror, phi_ther[1:]))

fig = plt.figure()
ax = fig.subplots()

#ax.plot(x[0:nbr_mech], best_phi[0:nbr_mech])
#ax.plot(x[nbr_mech:], best_phi[nbr_mech:])

ax.plot(x_fast_full, phi_fast_full)
ax.plot(x_ther_full, phi_ther_full)

ax.vlines(np.linspace(1, len(list_batch), len(list_batch))*width, 0, np.max(np.append(best_phi, phi_1)), linestyle="--", color="black")
ax.grid()
ax.set_ylabel("$\Phi$ [n cm$^{-2}$ s$^{-1}$]")
ax.set_xlabel("$x$ [cm]")
ax.legend(["Fast", "Thermal"])

fig.savefig("Ex4_Flux_opti.pdf")

# Plot the relative power

# Mirror the results to plot the full power
x_half = x[0:nbr_mech]
x_half_mirror = -x_half[::-1]
x_full = np.concatenate((x_half_mirror, x_half[1:]))

P_rel_mirror = best_power_rel[::-1]
P_rel_full = np.concatenate((P_rel_mirror, best_power_rel[1:]))

fig = plt.figure()
ax = fig.subplots()
#ax.plot(x[:nbr_mech], best_power_rel)
ax.plot(x_full, P_rel_full)
ax.set_xlabel("$x$ [cm]")
ax.set_ylabel("$P_{rel}$")
ax.grid()

fig.savefig("Ex4_Power_opti.pdf")

plt.show()