'''
Created by Coleman Smith on 1/23/24
NUEN/MEEN 644 HW1
Due 03 May 2024 by 3:00 PM
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Mesh parameters
nx, ny = 160, 80  # number of control volumes in x and y directions
dx = L / nx
dy = H / ny

# Initialize field variables
u = np.zeros((nx+1, ny+2))   # u-velocity on staggered grid
v = np.zeros((nx+2, ny+1))   # v-velocity on staggered grid
p = np.zeros((nx+2, ny+2))   # pressure on main grid
T = np.zeros((nx+2, ny+2))   # temperature on main grid

# Set initial and boundary conditions
u[:, 1:-1] = U_in
T[:, 0] = T_w
T[:, -1] = T_w
T[0, :] = T_in

def update_boundary_conditions():
    # Apply no-slip condition at walls
    u[:, 0] = u[:, -1] = 0
    v[0, :] = v[-1, :] = 0
    # Apply constant temperature at walls
    T[:, 0] = T[:, -1] = T_w
    T[0, :] = T_in

def solve_momentum():
    # Solve the momentum equations using the SIMPLE algorithm
    # Placeholder for the actual solver
    pass

def solve_energy():
    # Solve the energy equation using the power-law scheme
    # Placeholder for the actual solver
    pass

def check_convergence():
    # Check for convergence (placeholder)
    return False

# Main iteration loop
converged = False
while not converged:
    update_boundary_conditions()
    solve_momentum()
    solve_energy()
    converged = check_convergence()

# Post-processing (e.g., plotting results)
# Placeholder for plotting code

print("Simulation complete!")


#os.system('cls')

# Define constants
L                           = 2                                      # m
H                           = 0.02                                   # m
omega_u, omega_v            = 0.3                                    # given under-relaxation factors
omega_p                     = 0.7                                    # given under-relaxation factors
T_H2O                       = 20                                     # Deg C
Ru_tol,Rv_tol,Rp_tol,RT_tol = 1E-6                                   # Tolerance for u-vel residual
Re                          = 200                                    # Unitless Reynolds #
rho_H2O                     = 998.3                                  # kg/m^3
mu_H2O                      = 1.002E-3                               # N*s/m^2
u_0                         = (Re*mu_H2O)/(rho_H2O*L)                # m/s
N_CVs_one                   = [[10,5]]                               # Dimensions of CVs
N_CVs_two                   = [[20,10], [60,20], [120,40], [160,80]] # Dimensions of CVs


###################################
#########  Problem #1 #############
###################################

###################################
#########  Problem #2 #############
###################################

def plot_T_bulk():
    '''
    Plots T_bulk as a function of x given a solved temperature field
    '''