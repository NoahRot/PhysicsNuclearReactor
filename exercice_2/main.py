"""
 __  __         _     _ _                         _                                      _           
|  \/  |___  __| |___| (_)_ _  __ _   __ _   _ __| |__ _ _ _  __ _ _ _   _ _ ___ __ _ __| |_ ___ _ _ 
| |\/| / _ \/ _` / -_) | | ' \/ _` | / _` | | '_ \ / _` | ' \/ _` | '_| | '_/ -_) _` / _|  _/ _ \ '_|
|_|  |_\___/\__,_\___|_|_|_||_\__, | \__,_| | .__/_\__,_|_||_\__,_|_|   |_| \___\__,_\__|\__\___/_|  
                              |___/         |_|                                                      

"""

# === Importation ===
import numpy as np
import matplotlib.pyplot as plt

from analytical import *
from matrices import *
from tolerence import *

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

# === Simulation parameters ===
nbr_mech = 200          # Number of mech

Sigma_a_R = 0.01        # Cross-section absorption of reflector [cm^-1]
Sigma_s_R = 0.3         # Cross-section scattering of reflector [cm^-1]
Sigma_a_C = 0.01        # Cross-section absorption of core [cm^-1]
Sigma_s_C = 0.3         # Cross-section scattering of core [cm^-1]
nu_Sigma_f_C = 0.015    # nu : neutron per reaction, Sigma : Cross-section of fusion [cm^-1]

a = 20                  # Radius of core
b = 10                  # Radius of reflector

# === No reflector ===
# Parameter calculation
delta_x = a/nbr_mech    # Mech size

# Tolerence for convergence and maximum iteration (avoid infinit loop when creating the project)
tol_k = 1e-7                # Tolerence on k
tol_phi = 1e-7              # Tolerence on phi
max_iteration = int(1e2)    # Maximum number of iteration (for iterative method of eigen problem)

# Build geometry of the reactor
# Diffusion
D = np.zeros(int(nbr_mech))
D.fill(1/(3*(Sigma_a_C + Sigma_s_C)))
# Cross section (absorption and fission)
Sigma_a = np.full(int(nbr_mech), Sigma_a_C)
nu_Sigma_f = np.full(int(nbr_mech), nu_Sigma_f_C)

# Extrapolated length
d = 2*D[0]

# Buckling material and geometrical
B_m = np.sqrt(-(Sigma_a_C - nu_Sigma_f_C)/D)
B_0 = np.pi / (2*(a + d))

# Position
x = np.linspace(delta_x*0.5, a-0.5*delta_x, int(nbr_mech))

# Evolution matrices
M = matrix_M(nbr_mech, delta_x, D, Sigma_a)
M_inv = np.linalg.inv(M)
F = matrix_F(nbr_mech, nu_Sigma_f)

# === Simulation ===

# Initial guess
list_k = [] # (just for evolution of k)
k_last = 1                              # Previous k evaluation (set to 1 initially)
k = 1                                   # k of the reactor
list_k.append(k)
phi_last = np.zeros(int(nbr_mech)) + 1  # Previous phi evaluation (set to 1 initially)
phi = np.zeros(int(nbr_mech)) + 1       # phi of the reactor
S   = 1/k*np.dot(F, phi)                # S term

# First iteration
phi = np.dot(M_inv, S)
k   = k_last*(np.sum(np.dot(F, phi)))/(np.sum(np.dot(F, phi_last)))
list_k.append(k)
S   = 1/k*np.dot(F, phi)

# Iterative method for solving eigen problem
iteration = 1
while check_tolerence(k, k_last, tol_k, phi, phi_last, tol_phi) and iteration <= max_iteration:

    # Store last values
    phi_last = phi
    k_last = k

    # Step
    phi = np.dot(M_inv, S)
    k   = k_last*(np.sum(np.dot(F, phi)))/(np.sum(np.dot(F, phi_last)))
    list_k.append(k)
    S   = 1/k*np.dot(F, phi)

    iteration += 1

# Add boudary values
# Left
x = np.append(0, x)
phi = np.append(phi[0], phi)
# Right
x = np.append(x, x[-1]+delta_x/2)
phi = np.append(phi, phi[-1]/(1+delta_x/(4*D[-1])))

# Normalization
phi = phi / phi[0]

# Phi solution
phi_ana_mat = analytical_no_ref(1, B_m[0], x)   # Material solution
phi_ana_geo = analytical_no_ref(1, B_0, x)      # Geometrical solution

# Error
error_geo = np.abs(phi- phi_ana_geo)
error_mat = np.abs(phi - phi_ana_mat)

# Print results
print("*"*44)
print("*"*11, "Results - Analytical", "*"*11)
print("*"*44)

# k effective computation analytical
L = np.sqrt(1/(3*(Sigma_a_C + Sigma_s_C)*Sigma_a_C))
k_eff_1 = nu_Sigma_f_C / Sigma_a_C / (1 + B_0*B_0*L*L)
print("keff analytical =", k_eff_1)

# Current at boundary computation analytical
J = D[0]*1*B_0*np.sin(B_0*a)
print("Current at x =", a, ":", J)

print("*"*44)
print("*"*10, "Results - No Refractor", "*"*10)
print("*"*44)

print("# iteration = " + str(iteration))
print("keff numerical =", k)
print("Current at x =", x[-1], x[-2]," :", -D[-1]*2*(phi[-1]-phi[-2])/delta_x)

# Print if the max iteration have been reached
if iteration == max_iteration+1:
    print("MAX ITER REACH")

print("Phi numerical at " + str(x[11]) + " = " + str(phi[11]))
print("Phi analytical (for material buckling) at " + str(x[11]) + " = " + str(analytical_no_ref(1, B_m[0], x[11])))
print("Phi analytical (for geometrical buckling) at " + str(x[11]) + " = " + str(np.cos(B_0*x[11])))

# Plot numerical result of the flux
fig = plt.figure()
ax = fig.subplots()
ax.set_title("No refractor - Numerical")
ax.plot(x, phi, linestyle="-", marker=" ")
ax.grid()
ax.set_xlabel("$x$ [cm]")
ax.set_ylabel("$\Phi$ [n cm$^{-2}$ s$^{-1}]$")
ax.legend(["Numerical"])

fig.savefig("Ex2_Q1.pdf")

# Plot absolute error of the flux
fig = plt.figure()
ax = fig.subplots()
ax.set_title("No refractor - Error")
ax.plot(x, error_geo, linestyle="-", marker=" ")
ax.plot(x, error_mat, linestyle="-", marker=" ")
ax.grid()
ax.set_xlabel("$x$ [cm]")
ax.set_ylabel("$\Delta \Phi$ [n cm$^{-2}$ s$^{-1}]$")
ax.legend(["Geometrical", "Material"])

fig.savefig("Ex2_Q2.pdf")

# === With refractor ===

# Parameter
nbr_mech = 300              # Number of mech
delta_x = (a+b)/nbr_mech    # Mech size

tol_k = 1e-7                # Tolerence on k
tol_phi = 1e-7              # Tolerence on phi
max_iteration = int(1e2)    # Maximum number of iteration (for iterative method of eigen problem)

# Mech of the space
x = np.linspace(delta_x*0.5, a+b-0.5*delta_x, int(nbr_mech))

# Build the geometry of the reactor
D           = np.zeros(int(nbr_mech))   # Diffusion
Sigma_a     = np.zeros(int(nbr_mech))   # Absorbtion cross-section
nu_Sigma_f  = np.zeros(int(nbr_mech))   # Fission cross-section

for i in range(int(nbr_mech)):

    if (x[i] < a):
        # Inside the core
        D[i] = 1/(3*(Sigma_a_C + Sigma_s_C))
        Sigma_a[i] = Sigma_a_C
        nu_Sigma_f[i] = nu_Sigma_f_C
        
    else:
        # Inside the refractor
        D[i] = 1/(3*(Sigma_a_R + Sigma_s_R))
        Sigma_a[i] = Sigma_a_R
        nu_Sigma_f[i] = 0

# Build evolution matrices
M = matrix_M(nbr_mech, delta_x, D, Sigma_a)
M_inv = np.linalg.inv(M)
F = matrix_F(nbr_mech, nu_Sigma_f)

# === Simulation ===
# Initial guess
list_k = [] # (just for evolution of k)
k_last = 1                              # Previous k evaluation (set to 1 initially)
k = 1                                   # k of the reactor
list_k.append(k)
phi_last = np.zeros(int(nbr_mech)) + 1  # Previous phi evaluation (set to 1 initially)
phi = np.zeros(int(nbr_mech)) + 1       # phi of the reactor
S   = 1/k*np.dot(F, phi)                # S term

# First iteration
phi = np.dot(M_inv, S)
k   = k_last*(np.sum(np.dot(F, phi)))/(np.sum(np.dot(F, phi_last)))
list_k.append(k)
S   = 1/k*np.dot(F, phi)

# Iterative method for solving eigen problem
iteration = 1
while check_tolerence(k, k_last, tol_k, phi, phi_last, tol_phi) and iteration <= max_iteration:
    
    # Store last values
    phi_last = phi
    k_last = k

    # Step
    phi = np.dot(M_inv, S)
    k   = k_last*(np.sum(np.dot(F, phi)))/(np.sum(np.dot(F, phi_last)))
    list_k.append(k)
    S   = 1/k*np.dot(F, phi)

    iteration += 1

# Add boudary values
# Left
x = np.append(0, x)
phi = np.append(phi[0], phi)
# Right
x = np.append(x, x[-1]+delta_x/2)
phi = np.append(phi, phi[-1]/(1+delta_x/(4*D[-1])))

# Normalisation
phi = phi / phi[0]

# Print the results
print("*"*46)
print("*"*10, "Results - With Refractor", "*"*10)
print("*"*46)

print("Current between ", x[200], "and", x[201]," :", -(D[200]*phi[200]-D[201]*phi[201])/delta_x)
print("# iteration = " + str(iteration))
print("k =", k)
if iteration == max_iteration+1:
    print("MAX ITER REACH")

print("Phi numerical at " + str(x[11]) + " = " + str(phi[11]))

# Plot
# Plot the numerical result of the flux
fig = plt.figure()
ax = fig.subplots()
ax.set_title("Refractor - Numerical")
ax.plot(x, phi, linestyle="-", marker=" ")
ax.grid()
ax.set_xlabel("$x$ [cm]")
ax.set_ylabel("$\Phi$ [n cm$^{-2}$ s$^{-1}$]")
ax.vlines(np.array([a, a+b]), 0, 1, linestyles="--", color="black")

fig.savefig("Ex2_Q3.pdf")

plt.show()