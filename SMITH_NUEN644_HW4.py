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

itermax = 100 # define maximum allowable iterations of SIMPLE algorithm in event of runaway

def SIMPLE_sol(gridsize,iterlim):

    dims    = (gridsize+2,gridsize+2)
    dx      = L/gridsize
    dy      = dx
    Fe      = np.ones(dims)
    Fw      = np.ones(dims)
    Fn      = np.ones(dims)
    Fs      = np.ones(dims)
    De      = np.ones(dims)
    Dw      = np.ones(dims)
    Dn      = np.ones(dims)
    Ds      = np.ones(dims)
    Pe      = np.ones(dims)
    Pw      = np.ones(dims)
    Pn      = np.ones(dims)
    Ps      = np.ones(dims)
    aE      = np.ones(dims)
    aW      = np.ones(dims)
    aN      = np.ones(dims)
    aS      = np.ones(dims)
    aP      = np.ones(dims)
    p       = np.ones(dims)
    u       = np.ones(dims)
    v       = np.ones(dims)
    p_star  = np.ones(dims)
    u_star  = np.ones(dims)
    v_star  = np.ones(dims)
    p_prm  = np.ones(dims)
    u_prm  = np.ones(dims)
    v_prm  = np.ones(dims)
    iternum = 0
    
    while Ru>Ru_tol and Rv>Rv_tol and Rp>Rp_tol and iternum<iterlim:
        for j in range(0,gridsize+2):
            for i in range(0,gridsize+2):

                if i == 0 or j == 0 or i == gridsize+1 or gridsize+1:
                    print('volume is a corner or boundary CV')
                else:
                    # Calculate flow strengths
                    Fe[i,j] = rho_H2O*(0.5*(u[i,j] + u[i+1,j]))*dy
                    Fw[i,j] = rho_H2O*(0.5*(u[i,j] + u[i-1,j]))*dy
                    Fn[i,j] = rho_H2O*(0.5*(u[i,j] + u[i,j-1]))*dx
                    Fs[i,j] = rho_H2O*(0.5*(u[i,j] + u[i,j-1]))*dx

                    # Calculate diffusion strengths
                    De[i,j] = mu_H2O*dy/dx
                    Dw[i,j] = mu_H2O*dy/dx
                    Dn[i,j] = mu_H2O*dy/dx
                    Ds[i,j] = mu_H2O*dy/dx

                    # Calculate Peclet #s
                    Pe[i,j] = Fe[i,j]/De[i,j]
                    Pw[i,j] = Fw[i,j]/Dw[i,j]
                    Pn[i,j] = Fn[i,j]/Dn[i,j]
                    Ps[i,j] = Fs[i,j]/Ds[i,j]

                    # Calcuate coeffs
                    aE[i,j] = De[i,j]*np.max(0,(1-0.1*np.abs(Pe[i,j]))^5) + np.max(0,(-Fe[i,j]))
                    aW[i,j] = Dw[i,j]*np.max(0,(1-0.1*np.abs(Pw[i,j]))^5) + np.max(0,(-Fw[i,j]))
                    aN[i,j] = Dn[i,j]*np.max(0,(1-0.1*np.abs(Pn[i,j]))^5) + np.max(0,(-Fn[i,j]))
                    aS[i,j] = Ds[i,j]*np.max(0,(1-0.1*np.abs(Ps[i,j]))^5) + np.max(0,(-Fs[i,j]))
                    aP[i,j] = aE[i,j]+aW[i,j]+aN[i,j]+aS[i,j]

                    # Guess pressure field (p*)

                    # Solve for u* & v* using p*

                    # Calculate d_u & d_v

                    # Solve pressure correction (p')

                    # Calculate velocity corrections (u' & v')

                    # Convergence Check

                    # Zero correction terms
                    p_star = np.zeros(dims)
                    u_star = np.zeros(dims)
                    v_star = np.zeros(dims)


###################################
#########  Problem #2 #############
###################################

for griddims in N_CVs:
    print(griddims)
