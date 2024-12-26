"""

"""

# === Importation ===
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci

from analytical import *
from matrices import *
from Buckling import *

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
nbr_group = 2           # Number of group

Sigma_a_R_fast = 0.00        # Cross-section absorption of reflector [cm^-1]
Sigma_a_R_ther = 0.012
Sigma_a_C_fast = 0.002        # Cross-section absorption of core [cm^-1]
Sigma_a_C_ther = 0.060
Sigma_1_2_C = 0.038
Sigma_1_2_R = 0.040
nu_Sigma_f_fast = 0.001    # nu : neutron per reaction, Sigma : Cross-section of fusion [cm^-1]
nu_Sigma_f_ther = 0.069

D_C_fast = 1.130
D_C_ther = 0.160
D_R_fast = 1.130
D_R_ther = 0.160

a = 50                  # Radius of core
b = 10                  # Radius of reflector

# === No reflector ===
# Parameter calculation
nbr_mech = 500          # Number of mech
delta_x = a/nbr_mech

# Diffusion
D = np.zeros(int(nbr_mech*nbr_group))
for i in range(nbr_group):
    for j in range(nbr_mech):
        if i == 0:
            D[i*nbr_mech+j] = D_C_fast
        elif i == 1:
            D[i*nbr_mech+j] = D_C_ther

# Buckling material and geometrical
B_0 = np.pi / (2*a)

# Cross section (absorption and fission)
Sigma_1_2 = np.zeros(int(nbr_mech*nbr_group))
for i in range(nbr_group):
    for j in range(nbr_mech):
        if i == 0:
            Sigma_1_2[i*nbr_mech+j] = Sigma_1_2_C
        elif i == 1:
            Sigma_1_2[i*nbr_mech+j] = 0

Sigma_a = np.zeros(int(nbr_mech*nbr_group))
for i in range(nbr_group):
    for j in range(nbr_mech):
        if i == 0:
            Sigma_a[i*nbr_mech+j] = Sigma_a_C_fast
        elif i == 1:
            Sigma_a[i*nbr_mech+j] = Sigma_a_C_ther
            
nu_Sigma_f = np.zeros(int(nbr_mech*nbr_group))
for i in range(nbr_group):
    for j in range(nbr_mech):
        if i == 0:
            nu_Sigma_f[i*nbr_mech+j] = nu_Sigma_f_fast
        elif i == 1:
            nu_Sigma_f[i*nbr_mech+j] = nu_Sigma_f_ther

# Position
x = np.linspace(delta_x*0.5, a-0.5*delta_x, int(nbr_mech))
x = np.append(x, x)

# Evolution matrices
M = matrix_M_no_ref(nbr_mech, nbr_group, delta_x, D, Sigma_1_2, Sigma_a)
M_inv = np.linalg.inv(M)
F = matrix_F_no_ref(nbr_mech, nbr_group, nu_Sigma_f)

# Simulation
k, phi = np.linalg.eig(np.dot(M_inv, F))
k_id = np.argmax(k)
print(k_id)
k = k[k_id]
phi = phi[:,k_id]
print("k eff =", k)

# Normalization
phi = phi / phi[0]

# Get buckling
B_fast = buckling(x[0:nbr_mech-1], phi[0:nbr_mech-1])
B_ther = buckling(x[nbr_mech:-1], phi[nbr_mech:-1])

# Phi analytic solution
B_ana = np.pi/(2*a)
amplitude_fast = 1
amplitude_ther = Sigma_1_2_C/(B_ana*B_ana*D_C_ther + Sigma_a_C_ther)*amplitude_fast
phi_fast_ana = amplitude_fast*np.cos(B_ana*x[0:nbr_mech-1])
phi_ther_ana = amplitude_ther*np.cos(B_ana*x[nbr_mech:-1])

# K effective analytic
#k_eff_ana = (nu_Sigma_f_fast + nu_Sigma_f_ther*(Sigma_1_2_C)/(B_ana*B_ana*D_C_ther + Sigma_a_C_ther))/(D_C_fast*B_ana*B_ana + Sigma_a_C_fast)
#k_eff_ana = (np.sqrt(nu_Sigma_f_fast) + np.sqrt(nu_Sigma_f_ther)*(Sigma_1_2_C)/(B_ana*B_ana*D_C_ther + Sigma_a_C_ther))/(B_ana*B_ana*D_C_fast - (Sigma_a_C_fast + Sigma_1_2_C))
#k_eff_ana_fast = nu_Sigma_f_fast / (Sigma_a_C_fast + D_C_fast*np.pi*np.pi/(4*a*a))
#k_eff_ana_ther = nu_Sigma_f_ther / (Sigma_a_C_ther + D_C_ther*np.pi*np.pi/(4*a*a))
B_ana = B_0
B_square = np.power(B_ana,2)
Sigma_t_fast = Sigma_a_C_fast + Sigma_1_2_C
Sigma_t_ther = Sigma_a_C_ther
k_eff_ana = (nu_Sigma_f_fast*(B_square*D_C_ther + Sigma_t_ther) + Sigma_1_2_C*nu_Sigma_f_ther)/((B_square*D_C_fast + Sigma_t_fast)*(B_square*D_C_ther + Sigma_t_ther))

# Delta between analytic and numeric
delta_fast = np.abs(phi[0:nbr_mech-1] - phi_fast_ana)
delta_ther = np.abs(phi[nbr_mech:-1] - phi_ther_ana)

# Print results
print("*"*20)
print("*** Bare Reactor ***")
print("*"*20)

print("k eff numerical : " + str(k))
print("Phi numerical (fast) at " + str(x[100]) + " = " + str(phi[100]))
print("Phi numerical (ther) at " + str(x[100 + nbr_mech]) + " = " + str(phi[100 + nbr_mech]))
print("Fast buckling :", B_fast[1])
print("Ther buckling :", B_ther[1])
print("k eff analytic =", k_eff_ana)

# Get fit curve
fit_cos_fast = cosine_callback(x[0:nbr_mech-1], B_fast[0], B_fast[1])
fit_cos_ther = cosine_callback(x[nbr_mech:-1], B_ther[0], B_ther[1])

# Plot
fig = plt.figure()
ax = fig.subplots()
ax.set_title("No refractor - Numerical")
ax.plot(x[0:nbr_mech-1], phi[0:nbr_mech-1], linestyle="-", marker=" ")
ax.plot(x[nbr_mech:-1], phi[nbr_mech:-1], linestyle="-", marker=" ")
#ax.plot(x[0:nbr_mech-1], phi_fast_ana)
#ax.plot(x[nbr_mech:-1], phi_ther_ana)
ax.grid()
ax.set_xlabel("$x$ [cm]")
ax.set_ylabel("$\Phi$ [n cm$^{-2}$ s$^{-1}]$")
ax.legend(["Numerical fast", "Numerical thermal"])

fig.savefig("Ex3_BarePhi.pdf")

fig = plt.figure()
ax = fig.subplots()
ax.set_title("No refractor - Delta")
ax.plot(x[0:nbr_mech-1], delta_fast, linestyle="-", marker=" ")
ax.plot(x[nbr_mech:-1], delta_ther, linestyle="-", marker=" ")
ax.grid()
ax.set_xlabel("$x$ [cm]")
ax.set_ylabel("$\Delta \Phi$ [n cm$^{-2}$ s$^{-1}]$")
ax.legend(["Delta fast", "Delta thermal"])

fig.savefig("Ex3_BareDelta.pdf")

# === With refractor ===

# Parameter calculation
nbr_mech = 1200
delta_x = 2*(a + b)/nbr_mech
print("Delta x =", delta_x)

# Position
x = np.linspace(-(a+b-0.5*delta_x), a+b-0.5*delta_x, int(nbr_mech))
x = np.append(x, x)

# Diffusion
D = np.zeros(int(nbr_mech*nbr_group))
for i in range(nbr_group):
    for j in range(nbr_mech):
        if i == 0:
            if np.abs(x[j]) < a:
                D[i*nbr_mech+j] = D_C_fast
            else:
                D[i*nbr_mech+j] = D_R_fast

        elif i == 1:
            if np.abs(x[j]) < a:
                D[i*nbr_mech+j] = D_C_ther
            else:
                D[i*nbr_mech+j] = D_R_ther

Sigma_1_2 = np.zeros(int(nbr_mech*nbr_group))
for i in range(nbr_group):
    for j in range(nbr_mech):
        if i == 0:
            if np.abs(x[j]) < a:
                Sigma_1_2[i*nbr_mech+j] = Sigma_1_2_C
            else:
                Sigma_1_2[i*nbr_mech+j] = Sigma_1_2_R

        elif i == 1:
            if np.abs(x[j]) < a:
                Sigma_1_2[i*nbr_mech+j] = 0
            else:
                Sigma_1_2[i*nbr_mech+j] = 0

Sigma_a = np.zeros(int(nbr_mech*nbr_group))
for i in range(nbr_group):
    for j in range(nbr_mech):
        if i == 0:
            if np.abs(x[j]) < a:
                Sigma_a[i*nbr_mech+j] = Sigma_a_C_fast
            else:
                Sigma_a[i*nbr_mech+j] = Sigma_a_R_fast

        elif i == 1:
            if np.abs(x[j]) < a:
                Sigma_a[i*nbr_mech+j] = Sigma_a_C_ther
            else:
                Sigma_a[i*nbr_mech+j] = Sigma_a_R_ther
            
nu_Sigma_f = np.zeros(int(nbr_mech*nbr_group))
for i in range(nbr_group):
    for j in range(nbr_mech):
        if i == 0:
            if np.abs(x[j]) < a:
                nu_Sigma_f[i*nbr_mech+j] = nu_Sigma_f_fast
            else:
                nu_Sigma_f[i*nbr_mech+j] = 0

        elif i == 1:
            if np.abs(x[j]) < a:
                nu_Sigma_f[i*nbr_mech+j] = nu_Sigma_f_ther
            else:
                nu_Sigma_f[i*nbr_mech+j] = 0

# Evolution matrices
M = matrix_M_ref(nbr_mech, nbr_group, delta_x, D, Sigma_1_2, Sigma_a)
M_inv = np.linalg.inv(M)
F = matrix_F_no_ref(nbr_mech, nbr_group, nu_Sigma_f)

# Simulation
k, phi = np.linalg.eig(np.dot(M_inv, F))
index_sorted = np.argsort(k)
k_0     = k[index_sorted[-1]]
phi_0   = phi[:,index_sorted[-1]]
k_1     = k[index_sorted[-2]]
phi_1   = phi[:,index_sorted[-2]]

# Normalization
phi_0 = phi_0 / phi_0[600]
phi_1 = phi_1 / np.max(phi_1)

# Print results
print("*"*25)
print("*** Reflected Reactor ***")
print("*"*25)

print("Current (fast) :", -(D[1099]*phi_0[1099] - D[1100]*phi_0[1100])/delta_x)
print("Current (ther) :", -(D[nbr_mech+1099]*phi_0[nbr_mech+1099] - D[nbr_mech+1100]*phi_0[nbr_mech+1100])/delta_x)
print("k eff (harmo 0) =", k_0)
print("k eff (harmo 1) =", k_1)
print("Phi numerical (fast) at " + str(x[700]) + " = " + str(phi_0[700]))
print("Phi numerical (ther) at " + str(x[700 + nbr_mech]) + " = " + str(phi_0[700 + nbr_mech]))

# Plot
fig = plt.figure()
ax = fig.subplots()
ax.set_title("With refractor - Numerical")
ax.plot(x[0:nbr_mech-1], phi_0[0:nbr_mech-1], linestyle="-", marker=" ")
ax.plot(x[nbr_mech:-1], phi_0[nbr_mech:-1], linestyle="-", marker=" ")
ax.vlines(np.array([a, a+b]), 0, 1, colors="black", linestyles="--")
#ax.vlines(np.array([x[nbr_mech+1099], x[nbr_mech+1100]]), 0, 1, colors="red", linestyles="--")
ax.grid()
ax.set_xlabel("$x$ [cm]")
ax.set_ylabel("$\Phi$ [n cm$^{-2}$ s$^{-1}]$")
ax.legend(["Fast", "Thermal"])

fig.savefig("Ex3_RefractorH0.pdf")

fig = plt.figure()
ax = fig.subplots()
ax.set_title("With refractor - Numerical")
ax.plot(x[0:nbr_mech-1], phi_1[0:nbr_mech-1], linestyle="-", marker=" ")
ax.plot(x[nbr_mech:-1], phi_1[nbr_mech:-1], linestyle="-", marker=" ")
ax.vlines(np.array([a, a+b]), -1, 1, colors="black", linestyles="--")
#ax.vlines(np.array([x[nbr_mech+499], x[nbr_mech+500]]), 0, 1, colors="red", linestyles="--")
ax.grid()
ax.set_xlabel("$x$ [cm]")
ax.set_ylabel("$\Phi$ [n cm$^{-2}$ s$^{-1}]$")
ax.legend(["Fast", "Thermal"])

fig.savefig("Ex3_RefractorH1.pdf")

plt.show()