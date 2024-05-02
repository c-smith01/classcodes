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
    # if p,u,v,ort resid is greater than tol, return converged as False, else return converged as true
    return False

# Main iteration loop
converged = False
while not converged:
    solve_momentum()
    solve_energy()
    converged = check_convergence()
    
def print_results(p,u,v,t,itercount,Rp,Ru,Rv):
    print('SIMPLE Algorithm terminated at {} iterations with Rp = {}, Ru = {}, Rv = {}'.format(itercount,Rp,Ru,Rv))
    print('Printing pressure solution fields')
    print(p)
    print('Printing u-vel solution fields')
    print(u)
    print('Printing v-vel solution fields')
    print(v)
    print('Printing temperature solution fields')
    print(t)

# Post-processing (e.g., plotting results)
# Placeholder for plotting code

def SIMPLE_sol(dimlist):
    
    #init solutions arrays
    psols = []
    usols = []
    vsols = []
    tsols = []

    for dims in dimlist:
        nx = dims[0]
        ny = dims[1]
        # Initialize field variables
        u = np.ones((nx+1, ny+2))   # u-velocity on staggered grid
        v = np.ones((nx+2, ny+1))   # v-velocity on staggered grid
        p = np.ones((nx+2, ny+2))   # pressure on main grid
        T = np.ones((nx+2, ny+2))   # temperature on main grid
        
        Rp = Ru = Rv = 1 # start residuals with values greater than tolerance to force at least one iteration
        itercount    = 1 # start count of iterations to convergence
    print("Simulation complete!")
    return [psols,usols,vsols,tsols]


#os.system('cls')

# Define constants
L                           = 2.00                                   # m
H                           = 0.02                                   # m
omega_u, omega_v            = 0.3                                    # given under-relaxation factors
omega_p                     = 0.7                                    # given under-relaxation factors
T_H2O                       = 20                                     # Deg C
Ru_tol,Rv_tol,Rp_tol,RT_tol = 1E-6                                   # Tolerance for u-vel residual
Re                          = 200                                    # Unitless Reynolds #
rho_f                       = 997.0                                  # kg/m^3
mu_f                        = 8.71E-4                                # N*s/m^2
C_pf                        = 4179.0                                 # J/kg*K
u_0                         = (Re*mu_f)/(rho_f*L)                    # m/s
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