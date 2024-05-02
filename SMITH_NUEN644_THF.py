'''
Created by Coleman Smith on 1/23/24
NUEN/MEEN 644 Take-Home Final
Due 03 May 2024 by 3:00 PM
'''

import matplotlib.pyplot as plt
import numpy as np
import os

# Define constants
L                           = 2.00                                           # m
H                           = 0.02                                           # m
omega_u, omega_v            = 0.3                                            # given under-relaxation factors
omega_p                     = 0.7                                            # given under-relaxation factors
Ru_tol,Rv_tol,Rp_tol,RT_tol = 1E-6                                           # Tolerance for u-vel residual
Re                          = 200                                            # Unitless Reynolds #
rho_f                       = 997.0                                          # kg/m^3
mu_f                        = 8.71E-4                                        # N*s/m^2
C_pf                        = 4179.0                                         # J/kg*K
u_0                         = (Re*mu_f)/(rho_f*L)                            # m/s
T_0                         = 100+273.15                                     # K
T_0                         = 27+273.15                                      # K
N_CVs_one                   = [[10,5]]                                       # Dimensions of CVs
N_CVs_two                   = [[10,5], [20,10], [60,20], [120,40], [160,80]] # Dimensions of CVs

# Iteratively-called methods
def reset(matr,dims):
    matr = np.zeros(dims)

def bnd_conds(u_matr,t_matr):
        u_matr[0][:] = u_0
        t_matr[0][:] = T_0

def ucoeffs(dims,
            Deu,Dwu,Dnu,Dsu,
            Feu,Fwu,Fnu,Fsu,
            Peu,Pwu,Pnu,Psu,
            aEu,aWu,aNu,aSu,aPu,
            u,pstate,dy,dx):

def conv_check(dx,dy,u,v,
               aEu,aWu,aNu,aSu,aPu,
               aEv,aWv,aNv,aSv,aPv):
    Ru = (np.abs(np.sum(np.multiply(aPu,u)-np.multiply(aEu,u)-np.multiply(aWu,u)-np.multiply(aNu,u)-np.multiply(aSu,u))))/(np.sum(np.multiply(aPu,u)))

    Rv = (np.abs(np.sum(np.multiply(aPv,v)-np.multiply(aEv,v)-np.multiply(aWv,v)-np.multiply(aNv,v)-np.multiply(aSv,v))))/(np.sum(np.multiply(aPv,v)))

    Rp = (np.sum(rho_H2O*u - rho_H2O*u*dy - rho_H2O*u - rho_H2O*u*dx))/(rho_H2O*u_0*L)

    Rt = (np.sum(rho_H2O*u - rho_H2O*u*dy - rho_H2O*u - rho_H2O*u*dx))/(rho_H2O*u_0*L)
    return Rp, Ru, Rv, Rt, convstate
    
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
        dx = L/nx
        dy = H/ny
        cent_dy = dy*2
        cent_dx = dx*2
        corn_dy = dy*1.5
        corn_dx = dx*1.5
        # Initialize field variables
        u = np.ones((nx+1, ny+2))   # u-velocity on staggered grid
        v = np.ones((nx+2, ny+1))   # v-velocity on staggered grid
        p = np.ones((nx+2, ny+2))   # pressure on main grid
        T = np.ones((nx+2, ny+2))   # temperature on main grid
        
        Rp = Ru = Rv = Rt = 1     # start residuals with values greater than tolerance to force at least one iteration
        itercount         = 1     # start count of iterations to convergence
        converged         = False # converged boolean starts on false to force at least one iteration
        while converged==False and itercount<iterlim:
            
    print("Simulation complete!")
    return [psols,usols,vsols,tsols]


###################################
#########  Problem #1 #############
###################################

# 1) Tabulated velocity with 5 decimal point truncation
prob_one_sol = SIMPLE_sol()


###################################
#########  Problem #2 #############
###################################

# 2a) Plot U/Uin at channel exit and show that Umax/Uin approaches 1.5 as mesh is refined
def plot_T_bulk():
    '''
    Plots T_bulk as a function of x given a solved temperature field
    '''

# 2b) Plot non-dimensional temperature at x=2.0 as a function of y

# 2c) Plot T_bulk and Tw as a function of x

# 2d) Plot the local nuselt number as a function of x

# Extras for post-processing
# Specify the filenames
filename_1 = 'p1tabs.txt'
filename_2 = 'p2tabs.txt'

# Open the file in write mode
with open(filename_1, 'w') as file:
    # Write each item from the list to the file, each on a new line
    for item in prob_one_sol:
        file.write(f"{item}\n")

print(f"List has been written to {filename_1}")

with open(filename_2, 'w') as file:
    # Write each item from the list to the file, each on a new line
    for item in prob_two_sol:
        file.write(f"{(item*1e5)/1e5}\n") #multiplying and dividing the decimal values by 1e5 truncates to 5 decimal places

print(f"List has been written to {filename_2}")