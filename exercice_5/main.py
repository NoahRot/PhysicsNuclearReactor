"""
    ___         _   ___         _      _   _                   _ _   _                                          _              
   | __|  _ ___| | | __|_ _____| |_  _| |_(_)___ _ _   __ __ _(_) |_| |_    _____ ___ __  ___ ____  _ _ _ ___  (_)_ _    __ _  
   | _| || / -_) | | _|\ V / _ \ | || |  _| / _ \ ' \  \ V  V / |  _| ' \  / -_) \ / '_ \/ _ (_-< || | '_/ -_) | | ' \  / _` | 
   |_| \_,_\___|_| |___|\_/\___/_|\_,_|\__|_\___/_||_|  \_/\_/|_|\__|_||_| \___/_\_\ .__/\___/__/\_,_|_| \___| |_|_||_| \__,_| 
   | |_  ___ _ __  ___  __ _ ___ _ _  ___ ___ _  _ ___  _ __  ___ __| (_)__ _      |_|                                         
   | ' \/ _ \ '  \/ _ \/ _` / -_) ' \/ -_) _ \ || (_-< | '  \/ -_) _` | / _` |                                                 
   |_||_\___/_|_|_\___/\__, \___|_||_\___\___/\_,_/__/ |_|_|_\___\__,_|_\__,_|                                                 
                       |___/                                                                                                   
"""

# === Importation ===
import numpy as np
import matplotlib.pyplot as plt
import scipy as sci

from Element import *
from analytic import *
from Euler import *
from exponential import *
from convergence import *
from matrix import *

# From hour to second (utility function)
def hour_to_second(h : float):
    return h*3600

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

"""
Vector form
/ U  235 \ 
| U  238 |
| Pu 239 |
| X      |
\ Y      /
"""

# Create elements
#         Element(sig_c,     sig_s,    nu,   sig_f,     kappa,    T,                 FY)
U_235   = Element(1e-24*12,  1e-24*10, 2.44, 1e-24*56,  3.24e-11, 0,                 0)
U_238   = Element(1e-24*4,   1e-24*10, 2.79, 1e-24*1,   3.32e-11, 0,                 0)
Pu_239  = Element(1e-24*81,  1e-24*10, 2.87, 1e-24*144, 3.33e-11, 0,                 0)
X       = Element(1e-24*5e6, 1e-24*10, 0,    1e-24*0,   0,        hour_to_second(9), 0.06)
Y       = Element(1e-24*50,  1e-24*10, 0,    1e-24*0,   0,        0,                 1.94)

listname = ["U235", "U238", "Pu239", "X", "Y"]

# Parameters
phi = 1e13                      # Constant neutron flux
P   = 937.5                     # Constant power density [W cm^-3]

# In part 1, run the convergence study if True
run_convergence_study = True
# Part of the exercice that will be run
run_part = [1, 2]
# Number of day simulated. 
nbr_day_simulated = 365

# Compute number of elements
h = 400                 # Height of the fuel rods [cm]
r = 0.4                 # Radius of the fuel rods [cm]
pitch = 20              # Pitch of the reactor [cm]
rho = 10.4              # Density of the fuel rods [g cm^-3]
M_O = 16                # Molar mass of the oxygen
M_U_235 = 235           # Molar mass of the U-235
M_U_238 = 238           # Molar mass of the U-238
N_A = 6.0221408e+23     # Avogardro number
nbr_rods = 264          # Number of rods inside the reactor

# Compute density inside a fuel rod
M = 2*M_O + 0.03*M_U_235 + 0.97*M_U_238 # Molar mass of the UO_2
n = rho*N_A/M                           # Density of the UO_2 molecules inside the fuel
n_235 = 0.03*n                          # Density of U-235 atoms inside the fuel
n_238 = 0.97*n                          # Density of U-238 atoms inside the fuel

# Take into account rod inside the moderator (not only the fuel)
V_ratio = (np.pi*r*r*nbr_rods)/(pitch*pitch) # Ratio between fuel volume and reactor volume 

n_235 *= V_ratio    # Density of U-235 inside the reactor (not just fuel rod)
n_238 *= V_ratio    # Density of U-238 inside the reactor (not just fuel rod)

print("*** Initial concentration ***")
print("Initial n for U 235 =", n_235)
print("Initial n for U 238 =", n_238)

# ***********************************
# *** First part - Simple problem ***
# ***********************************

if 1 in run_part:
    # Build the initial values
    N_0 = np.array([n_235, n_238, 0, 0, 0])

    # Build the matrix with constant flux
    A = matrix_A_const_phi(phi, U_235, U_238, Pu_239, X, Y)

    # Time parameters
    t = 24*3600*nbr_day_simulated
    t_one_day = 24*3600
    dt = 60
    nbr_step = int(t//dt)
    nbr_step_one_day = int(t_one_day//dt)
    t_array = np.linspace(0, t, nbr_step+1)
    t_array_one_day = np.linspace(0, t_one_day, nbr_step_one_day+1)

    # Exponential method
    #N_exp = exponential(nbr_step, dt, N_0, A)
    #print("Exp N final :",N_exp[-1,:])

    # Euler method
    N_euler = Euler(nbr_step, dt, N_0, A)
    print("Euler N final :",N_euler[-1,:])

    # Analytical results
    n_analytical = np.array([analytic_U(t_array, n_235, phi, U_235),
                            analytic_U(t_array, n_238, phi, U_238),
                            analytic_Pu(t_array, phi, n_238, Pu_239, U_238),
                            analytic_X(t_array, phi, n_235, X, U_235),
                            analytic_Y(t_array, phi, n_235, Y, U_235)])

    n_analytical = np.transpose(n_analytical)

    # Euler relative error (error set at zero if divided by 0)
    delta_euler = np.divide(np.abs(N_euler - n_analytical), n_analytical, out=np.zeros_like(n_analytical), where=(n_analytical!=0))

    # Plots Analytical solution 365 days
    fig = plt.figure()
    ax = fig.subplots()
    ax.set_title("Analytical - " + str(nbr_day_simulated) + " days")
    ax.plot(t_array, n_analytical)
    ax.grid()
    ax.set_yscale("log")
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel("$n$ [cm$^{-3}$]")
    ax.legend(listname)

    fig.savefig("Ex5_analytic_" + str(nbr_day_simulated) + "_day.pdf")

    # Plots Analytical solution 1 days
    fig = plt.figure()
    ax = fig.subplots()
    ax.set_title("Analytical - " + str(1) + " days")
    ax.plot(t_array_one_day, n_analytical[:nbr_step_one_day+1])
    ax.grid()
    ax.set_yscale("log")
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel("$n$ [cm$^{-3}$]")
    ax.legend(listname)

    fig.savefig("Ex5_analytic_" + str(1) + "_day.pdf")

    # Plots Numerical - Euler
    #fig = plt.figure()
    #ax = fig.subplots()
    #ax.set_title("Numerical - Euler")
    #ax.plot(t_array, N_euler)
    #ax.grid()
    #ax.set_yscale("log")
    #ax.set_xlabel("$t$ [s]")
    #ax.set_ylabel("$n$ [cm$^{-3}$]")
    #ax.legend(listname)
    #fig.savefig("Ex5_euler_" + str(nbr_day_simulated) + "_day.pdf")

    # Plots Numerical - Error on Euler
    fig = plt.figure()
    ax = fig.subplots()
    ax.set_title("Numerical error on Euler")
    ax.plot(t_array, delta_euler*100)
    ax.set_yscale("log")
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel("$\Delta n/n$ [$\%$]")
    ax.legend(listname)
    ax.grid()

    fig.savefig("Ex5_euler_error_" + str(nbr_day_simulated) + "_day.pdf")

    # Convergence study
    if run_convergence_study:
        convergence_study(phi, n_235, n_238, U_235, U_238, Pu_239, X, Y)

# ******************************************
# *** Second part - More complex problem ***
# ******************************************

if 2 in run_part:
    print("Run part 2")

    # Time parameters (180 days)
    t = 180*24*3600
    dt = 60
    nbr_step = int(t//dt)
    t_array = np.linspace(0, t, nbr_step+1)

    # Build the initial values
    N_0 = np.array([n_235, n_238, 0, 0, 0])

    N_exp = np.zeros([nbr_step+1, len(N_0)])
    N_exp[0,:] = N_0
    current_N = N_0.copy()
    phi = np.zeros(nbr_step+1)

    # Flux at time t=0
    phi_current = 1.40574e15
    phi[0] = phi_current

    # Exponential method
    for i in range(nbr_step):

        # Print the progression
        if i%5000 == 0:
            print("Progression : " + str(100*(i+1)//nbr_step) + " %")

        # Simulation step
        A = matrix_A_const_P(phi_current, U_235, U_238, Pu_239, X, Y)
        exp_A = matrix_exp(A*dt)
        current_N = np.dot(exp_A, N_exp[i,:])

        # Flux update
        phi_current = P/(
            current_N[0] * U_235.kappa * U_235.sigma_f +
            current_N[1] * U_238.kappa * U_238.sigma_f + 
            current_N[2] * Pu_239.kappa * Pu_239.sigma_f + 
            current_N[3] * X.kappa * X.sigma_f + 
            current_N[4] * Y.kappa * Y.sigma_f
        )

        # Write results
        N_exp[i+1,:] = current_N
        phi[i+1] = phi_current
    

    # Plots
    # Plot the flux evolution
    fig = plt.figure()
    ax = fig.subplots()
    ax.set_title("Flux evolution")
    ax.plot(t_array, phi, color="blue")
    ax.grid()
    ax.set_ylabel("$\Phi$ [n cm$^{-2}$ s$^{-1}$]")
    ax.set_xlabel("$t$ [s]")

    fig.savefig("Ex5_Phi.pdf")

    # Plot the Isotope concentrations evolution
    fig = plt.figure()
    ax = fig.subplots()
    ax.set_title("Isotopes evolution")
    ax.plot(t_array, N_exp)
    ax.grid()
    ax.set_yscale("log")
    ax.set_xlabel("$t$ [s]")
    ax.set_ylabel("$n$ [cm$^{-3}$]")
    ax.legend(listname)

    fig.savefig("Ex5_isotopes.pdf")

    plt.show()