import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from matrix import *
from analytical import *

# Linear callback for curve fitting
def linear_callback(x, a, b):
    return a*x + b



# Use the results of the convergence to extract convergence order and plot the results
def convergence_fit_plot(mech_size_list, error_list, nbr_simu):

    # Fit to obtain the order of convergence
    param, cov = opt.curve_fit(linear_callback, np.log(mech_size_list), np.log(error_list))
    x_ = np.linspace(np.min(mech_size_list), np.max(mech_size_list), 2)

    id = np.argmax(error_list - np.power(mech_size_list, param[0]))
    B = error_list[id]/np.power(mech_size_list[id], param[0])
    y_ = np.power(x_, param[0])*B

    # Plot of the convergence study
    fig2 = plt.figure()
    bx = fig2.subplots()
    bx.set_title("Convergence study")
    bx.grid()
    bx.plot(mech_size_list, error_list, linestyle=" ", marker="x")
    bx.plot(x_, y_, linestyle="--", color="black")
    bx.set_yscale("log")
    bx.set_xscale("log")
    #bx.set_ylabel("$\max(\Delta \Phi)$ [n s$^{-1}$ cm$^{-2}$]")
    bx.set_ylabel("$\Delta \Phi(x_0)$ [n s$^{-1}$ cm$^{-2}$]")
    bx.set_xlabel("$\Delta x$ [cm]")
    bx.legend(["Data", "Conv. order = " + str(param[0])])

    fig2.savefig("Ex1_conv_N"+str(int(nbr_simu))+".pdf")

    # Print results
    print("Order of convergence :", param[0])



# Function to automatize the convergence study
def convergence_study(mech_0, mech_1, nbr_simu, Sigma_a, Sigma_s, x_0, S):

    # Simulation parameters
    D = 1/(3*(Sigma_a + Sigma_s))                       # Diffusion coefficient
    L = np.sqrt(1/(3*(Sigma_a + Sigma_s)*Sigma_a))      # Diffusion length [cm]
    d = 2*D                                             # Extrapolated length [cm]
    a = 2*(x_0 + d)                                     # Expanded boundary [cm]
    
    # Mech and error arrays
    nbr_mech_list = np.logspace(mech_0, mech_1, nbr_simu)
    mech_size_list = x_0/nbr_mech_list
    error_list = np.zeros(nbr_simu)

    # Analytical solution
    phi_ana = analytical(S, L, D, a, x_0)

    # Simulations
    for i in range(nbr_simu):

        # Parameter of the simulation
        nbr_mech = nbr_mech_list[i]
        delta_x = x_0/nbr_mech
        print("Convergence study | Simulation " + str(i+1) + "/" + str(nbr_simu) + ", current mech size : " + str(int(nbr_mech)))

        # Simulation
        m = create_matrix(nbr_mech, delta_x, D, Sigma_a)
        Src = np.zeros(int(nbr_mech))
        Src[0] = -S/(2*delta_x)
        phi_num = np.dot(m, Src)
        phi_num_at_10 = 1/(1+delta_x/(4*D))*phi_num[-1]     # Right bound

        # Error (use the error at x_0)
        error = np.abs(phi_num_at_10 - phi_ana)
        error_list[i] = error

    # Save error results (for big data)
    np.save("erro_" + str(mech_0) + "-" + str(mech_1) + "N" + str(nbr_simu) + ".npy", error_list)

    # Convergence fit
    convergence_fit_plot(mech_size_list, error_list, nbr_simu)



# Load, plot and save figure of already existing simulation 
def convergence_load(mech_0, mech_1, nbr_simu, save_fig):

    # Mech and error arrays
    nbr_mech_list = np.logspace(mech_0, mech_1, nbr_simu)

    # Load error
    error_list = np.load("output/erro_" + str(mech_0) + "-" + str(mech_1) + "N" + str(nbr_simu) + ".npy")

    # Check if the sizes are similar
    if len(nbr_mech_list) != len(error_list):
        print("ERROR::CONVERGENCE_LOAD : Mech and error arrays have different sizes (" + str(len(nbr_mech_list)) + "), (" + str(len(error_list)) + ")")
        return
    
    # Convergence fit
    convergence_fit_plot(nbr_mech_list, error_list, nbr_simu, save_fig)