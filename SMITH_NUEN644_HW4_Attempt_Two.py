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
L           = 1                         # m
omega       = 0.5                       # Reccomended relaxation factor
T_H2O       = 20                        # Deg C
Ru_tol      = 1E-6                      # Tolerance for u-vel residual
Rv_tol      = Ru_tol                    # Tolerance for v-vel residual
Rp_tol      = 1E-5                      # Tolerance for Pressure residual
Re          = 100                       # Unitless Reynolds #
rho_H2O     = 998.3                     # kg/m^3
mu_H2O      = 1.002E-3                  # N*s/m^2
u_0         = (Re*mu_H2O)/(rho_H2O*L)   # m/s
N_CVs_one   = [5]
N_CVs_two   = [8, 16, 64, 128, 256]     # Dimensions of CVs

# General methods used in both problems

def bnd_conds_one(u_matr):
    u_matr[:][0] = u_0
    u_matr[:][-1] = u_0

def bnd_conds_two(u_matr):
    u_matr[:][0] = u_0

def reset(matr,dims):
    matr = np.zeros(dims)

def ucoeffs(dims,pstate):
    dy = dx = L/dims
    if pstate == True:
        jlim = dims
    else:
        jlim = dims-1
    for j in range(1,jlim):
        for i in range(0,dims):
            Deu[i,j] = mu_H2O*dy/dx # Calculate diffusion strengths
            Dwu[i,j] = mu_H2O*dy/dx
            Dnu[i,j] = mu_H2O*dx/dy
            Dsu[i,j] = mu_H2O*dx/dy

            Feu[i,j] = rho_H2O*(0.5*(v[i,j] + v[i+1,j]))*dy # Calculate flow strengths
            Fwu[i,j] = rho_H2O*(0.5*(v[i,j] + v[i-1,j]))*dy
            Fnu[i,j] = rho_H2O*(0.5*(v[i,j] + v[i,j-1]))*dx
            Fsu[i,j] = rho_H2O*(0.5*(v[i,j] + v[i,j-1]))*dx

            Peu[i,j] = Feu[i,j]/Deu[i,j] # Calculate Peclet #s
            Pwu[i,j] = Fwu[i,j]/Dwu[i,j]
            Pnu[i,j] = Fnu[i,j]/Dnu[i,j]
            Psu[i,j] = Fsu[i,j]/Dsu[i,j]

            aEv[i,j] = Dev[i,j]*np.max(0,(1-0.1*np.abs(Pev[i,j]))^5) + np.max(0,(-Fev[i,j]))
            aWv[i,j] = Dwv[i,j]*np.max(0,(1-0.1*np.abs(Pwv[i,j]))^5) + np.max(0,(-Fwv[i,j]))
            aNv[i,j] = Dnv[i,j]*np.max(0,(1-0.1*np.abs(Pnv[i,j]))^5) + np.max(0,(-Fnv[i,j]))
            aSv[i,j] = Dsv[i,j]*np.max(0,(1-0.1*np.abs(Psv[i,j]))^5) + np.max(0,(-Fsv[i,j]))
            aPv[i,j] = aEv[i,j]+aWv[i,j]+aNv[i,j]+aSv[i,j]

    
def usolve(dims,pstate):
    if pstate == True:
        jlim = dims
    else:
        jlim = dims-1
    for j in range(1,jlim):
        for i in range(0,dims):
            if i == 0:
                u[i,j] = (aEu[i,j]*u[i+1,j]+aNu[i,j]*u[i,j+1]+aSu[i,j]*u[i,j-1])*(omega/aPu[i,j])
            elif i == dims-1:
                u[i,j] = (aWu[i,j]*u[i-1,j]+aNu[i,j]*u[i,j+1]+aSu[i,j]*u[i,j-1])*(omega/aPu[i,j])
            else:
                u[i,j] = (aEu[i,j]*u[i+1,j]+aWu[i,j]*u[i-1,j]+aNu[i,j]*u[i,j+1]+aSu[i,j]*u[i,j-1])*(omega/aPu[i,j])


def vcoeffs():

def vsolve():
    
def pcoeffs():
    
def psolve():
    
def ucorrect():

def vcorrect():
    
def pcorrect():
    
def conv_check():
    Rp = (np.sum(rho_H2O*u-rho_H2O*u,rho_H2O*u-rho_H2O*u))/(rho_H2O*u_0*L)
    Ru = (np.sum(np.multiply(aPu*u)-np.multiply(aEu*u)-np.multiply(aWu*u)-np.multiply(aNu*u)-np.multiply(aSu*u)))/(np.sum(np.multiply(aPu*u)))
    Rv = (np.sum(np.multiply(aPu*u)-np.multiply(aEu*u)-np.multiply(aWu*u)-np.multiply(aNu*u)-np.multiply(aSu*u)))/(np.sum(np.multiply(aPu*u)))

def print_res(ps,us,vs):
    print('Printing pressure solutions')
    print(ps)
    print('Printing u-vel fields')
    print(us)
    print('Printing v-vel fields')
    print(vs)



###################################
#########  Problem #1 #############
###################################

###################################
#########  Problem #2 #############
###################################