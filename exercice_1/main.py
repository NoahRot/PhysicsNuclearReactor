"""
 __  __         _     _ _                         _                                                     __                 _                   
|  \/  |___  __| |___| (_)_ _  __ _   __ _   _ __| |__ _ _ _  __ _ _ _   ___ ___ _  _ _ _ __ ___   ___ / _|  _ _  ___ _  _| |_ _ _ ___ _ _  ___
| |\/| / _ \/ _` / -_) | | ' \/ _` | / _` | | '_ \ / _` | ' \/ _` | '_| (_-</ _ \ || | '_/ _/ -_) / _ \  _| | ' \/ -_) || |  _| '_/ _ \ ' \(_-<
|_|  |_\___/\__,_\___|_|_|_||_\__, | \__,_| | .__/_\__,_|_||_\__,_|_|   /__/\___/\_,_|_| \__\___| \___/_|   |_||_\___|\_,_|\__|_| \___/_||_/__/
                              |___/         |_|                                                                                                 
"""

# === Importation ===
import numpy as np
import matplotlib.pyplot as plt

from analytical import *
from matrix import *
from convergence import *

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

# === Configuration ===
# Simulation parameters
nbr_mech = int(1e2)                                 # Number of mech (should not be greater than 1e4)

Sigma_a = 0.02                                      # Cross-section absorption [cm^-1]
Sigma_s = 4.0                                       # Cross-section scattering [cm^-1]
x_0     = 10.0                                      # Boundary [cm]
S       = 1000.0                                    # Emission [n cm^-2 s^-1]

D = 1/(3*(Sigma_a + Sigma_s))                       # Diffusion coefficient
L = np.sqrt(1/(3*(Sigma_a + Sigma_s)*Sigma_a))      # Diffusion length [cm]
d = 2*D                                             # Extrapolated length [cm]
a = 2*(x_0 + d)                                     # Expanded boundary [cm]
delta_x = x_0/(nbr_mech)                            # Spatial step [cm]

# Convergence study parameters
order_0 = 1                                         # First order of the convergence (should not be greater than 4)
order_1 = 3                                         # Last order of the convergence (should not be greater than 4)
nbr_simu = 10                                       # Number of simulation for the convergence study

# Mech of the space
x = np.linspace(0.5*delta_x, x_0-0.5*delta_x, int(nbr_mech))

# === Analytical solution ===
phi_ana = analytical(S, L, D, a, x)

# === Simulation ===

# Matrix of the diffusion problem
m = create_matrix(nbr_mech, delta_x, D, Sigma_a, True)

Src = np.zeros(int(nbr_mech))                       # Source
Src[0] = -S/(2*delta_x)

phi_num = np.dot(m, Src)                            # Solver

phi_num_at_0 = 0.5*delta_x/D*0.5*S + phi_num[0]     # Left bound
phi_num_at_10 = 1/(1+delta_x/(4*D))*phi_num[-1]     # Right bound

# Add Left (at x = 0)
x = np.append(0, x)
phi_num = np.append(phi_num_at_0, phi_num)
phi_ana = np.append(analytical(S, L, D, a, 0), phi_ana)

# Add Right (at x = x_0)
x = np.append(x, x_0)
phi_num = np.append(phi_num, phi_num_at_10)
phi_ana = np.append(phi_ana, analytical(S, L, D, a, x_0))

# Error
error = np.abs(phi_ana-phi_num)

# === Plot ===

# Plots analytical and numerical result (Not requested)
# fig = plt.figure()
# ax = fig.subplots()
# ax.set_title("Analytical and numerical flux")
# ax.plot(x, phi_ana, linestyle="-")
# ax.plot(x, phi_num, linestyle="-")
# ax.grid()
# ax.set_xlabel("$x$ [cm]")
# ax.set_ylabel("$\Phi$ [n s$^{-1}$ cm$^{-2}$]")
# ax.legend(["Analytical", "Numerical"])

# Plots analytical
fig4 = plt.figure()
dx = fig4.subplots()
dx.set_title("Analytical flux")
dx.plot(x, phi_ana, linestyle="-")
dx.grid()
dx.set_xlabel("$x$ [cm]")
dx.set_ylabel("$\Phi$ [n s$^{-1}$ cm$^{-2}$]")

fig4.savefig("Ex1_Analytical.pdf")

# Plots Error
fig2 = plt.figure()
bx = fig2.subplots()
bx.set_title("Error of the flux")
bx.plot(x, error)
bx.grid()
bx.set_xlabel("$x$ [cm]")
bx.set_ylabel("$\Delta \Phi$ [n s$^{-1}$ cm$^{-2}$]")

fig2.savefig("Ex1_Error.pdf")

# Plots Relative Error (Not requested)
# fig3 = plt.figure()
# cx = fig3.subplots()
# cx.set_title("Relative error of the flux")
# cx.plot(x, error/phi_ana*100)
# cx.grid()
# cx.set_xlabel("$x$ [cm]")
# cx.set_ylabel("$\Delta \Phi / \Phi_{ana}$ [\%]")

# Print check values
print("="*10 + " Values at specific points "  + "="*10)
print("Phi analytic at x=1.05 cm :", analytical(S, L, D, a, 1.05))
print("Phi analytic at x=0 cm :", analytical(S, L, D, a, 0))
print("Phi analytic at x=10 cm :", analytical(S, L, D, a, 10))
print("Phi numerical at x=" + str(x[11]) + " cm :", phi_num[11])
print("Phi numerical at x=" + str(x[0]) + " cm :", phi_num[0])
print("Phi numerical at x=" + str(x[-1]) + " cm :", phi_num[-1])

# === Convergence study ===
convergence_study(order_0, order_1, nbr_simu, Sigma_a, Sigma_s, x_0, S)

plt.show()

print("="*10 + " End of the program " + "="*10)