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

# Create Batches
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
P_total = 90e6  # [W]
height = 400    # [cm]
delta_x = 1     # [cm]
a = 120         # [cm]
b = 20          # [cm]
nbr_mech = (a + b)//delta_x
x_one = np.linspace(delta_x*0.5, (a+b)-0.5*delta_x, int(nbr_mech))

# Batch geometry
list_batch : list[Batch] = [batches[2], batches[2], batches[1], batches[1], batches[0], batches[0], batches[3]]

# Build geometry of the reactor
x, D, sig_a, nusig_f, ksig_f, sig_s_from_1, sig_s_from_2 = build_geometry(
    x_one, nbr_mech, width, list_batch)

# Build the matrix
M = matrix_M(x, delta_x, D, sig_a, sig_s_from_1, sig_s_from_2, True)
F = matrix_F(x, delta_x, nusig_f, True)

# Simulation
phi, P_fiss, P_rel, k, list_k = simulation(nbr_mech, M, F, height, delta_x, ksig_f, P_total)

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
print("Fast Flux at boudary :", flux_boundary*1e-14, " 10^14")

# Plots
fig = plt.figure()
ax = fig.subplots()
ax.plot(x[0:nbr_mech], phi[0:nbr_mech])
ax.plot(x[nbr_mech:], phi[nbr_mech:])
ax.vlines(np.linspace(1, len(list_batch), len(list_batch))*width, 0, np.max(phi), linestyle="--", color="black")
ax.grid()
ax.set_ylabel("$\Phi$ [n cm$^{-2}$ s$^{-1}$]")
ax.set_xlabel("$x$ [cm]")
ax.legend(["Fast", "Thermal"])

fig.savefig("Flux.pdf")

fig = plt.figure()
ax = fig.subplots()
ax.plot(x[0:nbr_mech], P_rel)
ax.grid()
ax.set_xlabel("$x$ [cm]")
ax.set_ylabel("$P_{rel}$ [W]")

fig.savefig("Power.pdf")

fig = plt.figure()
ax = fig.subplots()
ax.plot(list_k)
ax.grid()
ax.set_ylabel("$k_{eff}$")
ax.set_xlabel("Nbr steps")

phi_1 = np.copy(phi)

# ***************************
# *** Optimization patern ***
# ***************************

# Batch index list and permutations
list_batch_index : list[int] = [0, 0, 1, 1, 2, 2]
unique_permutations = set(permutations(list_batch_index))

list_valid_perm = []
best_perm = None
best_k = 0
best_phi = None
best_D = None
best_power_rel = None
best_ksig_f = None

fig = plt.figure()
ax = fig.subplots()
ax.set_xlabel("$x$ [cm]")
ax.set_ylabel("$\Phi$ [n cm$^{-2}$ s$^{-1}$]")
ax.grid()

# Simulation on each permutation
count = 1
for perm in unique_permutations:
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
    if phi[nbr_mech-1] + phi[2*nbr_mech-1] < 1e13 and np.max(P_rel) < 3.0:
        list_valid_perm.append(perm)

        # Check if current permutation better than the previous better
        if best_perm is None or best_k < k:
            best_perm = perm
            best_k = k
            best_phi = np.copy(phi)
            best_D = np.copy(D)
            best_power_rel = np.copy(P_rel)
            best_ksig_f = np.copy(ksig_f)

            ax.plot(x[:nbr_mech], phi[:nbr_mech])
            ax.plot(x[nbr_mech:], phi[nbr_mech:])

flux_boundary = (best_D[121]*best_phi[121] + best_D[120]*best_phi[120])/(best_D[121] + best_D[120])
print(list_valid_perm)
print("Best permutation :", perm)
print("Best k =", k)
print("Peak power : ", np.max(height*delta_x*delta_x*best_ksig_f*best_phi))
print("Fast flux at the core boundary : ", flux_boundary*1e-13, " 10^13")

# Plots
fig = plt.figure()
ax = fig.subplots()
#ax.plot(x[0:nbr_mech], phi_1[0:nbr_mech])
#ax.plot(x[nbr_mech:], phi_1[nbr_mech:])

ax.plot(x[0:nbr_mech], best_phi[0:nbr_mech])
ax.plot(x[nbr_mech:], best_phi[nbr_mech:])

ax.vlines(np.linspace(1, len(list_batch), len(list_batch))*width, 0, np.max(np.append(best_phi, phi_1)), linestyle="--", color="black")
ax.grid()
ax.set_ylabel("$\Phi$ [n cm$^{-2}$ s$^{-1}$]")
ax.set_xlabel("$x$ [cm]")
ax.legend(["Fast", "Thermal"])

fig.savefig("Flux_opti.pdf")

fig = plt.figure()
ax = fig.subplots()
ax.plot(x[:nbr_mech], best_power_rel)
ax.set_xlabel("$x$ [cm]")
ax.set_ylabel("$P_{rel}$")
ax.grid()

fig.savefig("Power_opti.pdf")

plt.show()