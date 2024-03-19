'''
Created by Coleman Smith on 1/23/24
NUEN/MEEN 644 HW1
Due 05 April 2024
'''
import matplotlib.pyplot as plt
import numpy as np
import os

os.system('cls')
os.system('clear')


# Define constants
L       = 1        # m
omega   = 0.5      # Reccomended relaxation factor
T_H2O   = 20       # Deg C
Ru_tol  = 1E-6     # Tolerance for u-vel residual
Rv_tol  = Ru_tol   # Tolerance for v-vel residual
Rp_tol = 1E-5      # Tolerance for Pressure residual
Re      = 100      # Unitless Reynolds #
rho_H2O = 998.3    # kg/m^3
mu_H2O  = 1.002E-3 # N*s/m^2
N_CVs   = [5,8,16,64, 128, 256]

###################################
#########  Problem #1 #############
###################################

# Set matrix containing nodes for 5x5 CVs
ITCV = 5
JTCV = ITCV
ITMAX = ITCV + 2
JTMAX = ITMAX
Ts_dim = (ITMAX, ITMAX)

# Initialize matrix of zeroes to represent nodes
Ts = np.zeros(Ts_dim)

# Set boundary conditions for problem
Ts[:,0] = 50
Ts[ITMAX-1,:] = 50
Ts[0,:] = 100
print(Ts)

Ru = 1 # Start all residuals at one to force at least one iteration
Rv = Ru
Rp = Ru

def SIMPLE_sol(gridsize):

    dims = (gridsize+2,gridsize+2)
    p_star = np.ones(dims)
    u_star = np.ones(dims)
    v_star = np.ones(dims)
    
    while Ru>Ru_tol and Rv>Rv_tol and Rp>Rp_tol :
        for j in range(1,gridsize+1):
            for i in range(1,gridsize+1):

                # Guess pressure field (p*)

                # Solve for u* & v* using p*

                # Calculate d_u & d_v

                # Solve pressure correction (p')

                # Calculate velocity corrections (u' & v')

                # Solve for any other relevant scalar quantities

                # Convergence Check


###################################
#########  Problem #2 #############
###################################

for griddims in N_CVs:
    print(griddims)
