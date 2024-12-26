import numpy as np
import matplotlib.pyplot as plt
import scipy as sci

from analytic import *
from Element import *
from exponential import *
from Euler import *
from matrix import *

# Callback function for fitting the convergence order
def linear_callback(x, a, b):
    return a*x + b

# Compute the convergence order
def convergence_order(x, y):

    # Log space x and y
    x_log = np.log(x)
    y_log = np.log(y)

    # Find the order of convergence
    param, cov = sci.optimize.curve_fit(linear_callback, x_log, y_log)
    a = param[0]
    b = param[1]

    # Find worse b possible (for plotting)
    #b = y_log - a*x_log

    return a, np.exp(b)

def convergence_study(phi, n_235, n_238, U_235, U_238, Pu_239, X, Y):

    # Time parameters
    t_one_day = 24*3600
    list_dt = np.logspace(1, 4, 100, endpoint=True)

    # Initial concentration
    N_0 = np.array([n_235, n_238, 0, 0, 0])

    id_X = 3 # index of the isotope X
    
    # Analytical solution
    n_analytical = np.array([analytic_U(t_one_day, n_235, phi, U_235),
                            analytic_U(t_one_day, n_238, phi, U_238),
                            analytic_Pu(t_one_day, phi, n_238, Pu_239, U_238),
                            analytic_X(t_one_day, phi, n_235, X, U_235),
                            analytic_Y(t_one_day, phi, n_235, Y, U_235)])
    
    # Error array
    delta_exp = np.zeros([len(list_dt), len(N_0)])
    delta_euler = np.zeros([len(list_dt), len(N_0)])

    # Build the matrix of evolution
    A = matrix_A_const_phi(phi, U_235, U_238, Pu_239, X, Y)

    # Simulations for the convergence study
    for i in range(len(list_dt)):
        
        # Print the progression
        if i%10 == 0:
            print("Simulation n#", i, "/", len(list_dt))

        # Time parameters
        dt = list_dt[i]
        nbr_step = int(t_one_day//dt)

        # Exponential method
        n_exp = exponential(nbr_step, dt, N_0, A)
        delta_exp[i,:] = np.abs(n_exp[-1,:] - n_analytical)/n_analytical

        # Euler method
        n_euler = Euler(nbr_step, dt, N_0, A)
        delta_euler[i,:] = np.abs(n_euler[-1,:] - n_analytical)/n_analytical

    # Compute order of convergence
    euler_order, euler_A = convergence_order(list_dt, delta_euler[:,id_X])
    exp_order, exp_A = convergence_order(list_dt, delta_exp[:,id_X])
    print("Convergence order Euler : " + str(euler_order))
    print("Convergence order Exp : " + str(exp_order))

    # Plot the convergence study

    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(list_dt, delta_euler[:,id_X], linestyle=" ", marker="+", color="blue")
    ax.plot(list_dt, delta_exp[:,id_X], linestyle=" ", marker="+", color="red")
    ax.plot(list_dt, euler_A*np.power(list_dt, euler_order), color="blue")
    ax.plot(list_dt, exp_A*np.power(list_dt, exp_order), color="red")
    ax.grid()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(["Euler", "Exponential", 
               "Conv. Euler: " + str('%.2f' % euler_order), "Conv. exp: " + str('%.2f' % exp_order)])
    
    fig.savefig("convergence.pdf")