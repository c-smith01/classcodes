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

def ucoeffs(dims,
            Deu,Dwu,Dnu,Dsu,
            Feu,Fwu,Fnu,Fsu,
            Peu,Pwu,Pnu,Psu,
            aEu,aWu,aNu,aSu,aPu,
            u,pstate):
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

            Feu[i,j] = rho_H2O*(0.5*(u[i,j] + u[i+1,j]))*dy # Calculate flow strengths
            Fwu[i,j] = rho_H2O*(0.5*(u[i,j] + u[i-1,j]))*dy
            Fnu[i,j] = rho_H2O*(0.5*(u[i,j] + u[i,j-1]))*dx
            Fsu[i,j] = rho_H2O*(0.5*(u[i,j] + u[i,j-1]))*dx

            Peu[i,j] = Feu[i,j]/Deu[i,j] # Calculate Peclet #s
            Pwu[i,j] = Fwu[i,j]/Dwu[i,j]
            Pnu[i,j] = Fnu[i,j]/Dnu[i,j]
            Psu[i,j] = Fsu[i,j]/Dsu[i,j]

            aEu[i,j] = Deu[i,j]*np.max(0,(1-0.1*np.abs(Peu[i,j]))^5) + np.max(0,(-Feu[i,j]))
            aWu[i,j] = Dwu[i,j]*np.max(0,(1-0.1*np.abs(Pwu[i,j]))^5) + np.max(0,(-Fwu[i,j]))
            aNu[i,j] = Dnu[i,j]*np.max(0,(1-0.1*np.abs(Pnu[i,j]))^5) + np.max(0,(-Fnu[i,j]))
            aSu[i,j] = Dsu[i,j]*np.max(0,(1-0.1*np.abs(Psu[i,j]))^5) + np.max(0,(-Fsu[i,j]))
            aPu[i,j] = aEu[i,j]+aWu[i,j]+aNu[i,j]+aSu[i,j]
    
def usolve(dims,u,aEu,aWu,aSu,aNu,aPu,pstate):
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

def vcoeffs(dims,
            Dev,Dwv,Dnv,Dsv,
            Fev,Fwv,Fnv,Fsv,
            Peu,Pwu,Pnu,Psu,
            aEu,aWu,aNu,aSu,aPu,
            v,pstate):
    dy = dx = L/dims

def vsolve(dims,v,aEv,aWv,aSv,aNv,aPv,pstate):
    
def pcoeffs(dims,p,bP,
            aE,aW,aN,aS,aP,
            bPP,aEP,aWP,aNP,aSP,aPP):
    bP[i,j] = rho_H2O*dy*(u[i-1,j]-u[i,j]) + rho_H2O*dx*(v[i,j-1]-v[i,j])
    aE[i,j] = De[i,j]*np.max(0,(1-0.1*np.abs(Pe[i,j]))^5) + np.max(0,(-Fe[i,j]))
    aW[i,j] = Dw[i,j]*np.max(0,(1-0.1*np.abs(Pw[i,j]))^5) + np.max(0,(-Fw[i,j]))
    aN[i,j] = Dn[i,j]*np.max(0,(1-0.1*np.abs(Pn[i,j]))^5) + np.max(0,(-Fn[i,j]))
    aS[i,j] = Ds[i,j]*np.max(0,(1-0.1*np.abs(Ps[i,j]))^5) + np.max(0,(-Fs[i,j]))
    aP[i,j] = aE[i,j]+aW[i,j]+aN[i,j]+aS[i,j]
    bPP[i,j] = rho_H2O*dy*(u[i-1,j]-u[i,j]) + rho_H2O*dx*(v[i,j-1]-v[i,j])
    aEP[i,j] = rho_H2O*du[i,j]*dy
    aWP[i,j] = rho_H2O*du[i,j]*dy
    aNP[i,j] = rho_H2O*du[i,j]*dx
    aSP[i,j] = rho_H2O*du[i,j]*dx
    aPP[i,j] = aEP[i,j]+aWP[i,j]+aNP[i,j]+aSP[i,j]
    
def psolve():
    p_prm[i,j] = (aEP[i,j]*p[i+1,j]+aWP[i,j]*p[i-1,j]+aNP[i,j]*p[i,j+1]+aSP[i,j]*p[i,j-1])*(omega/aPP[i,j])
    
def ucorrect(dims,u_prm,du,p_prm):
    if i == dims+1:
        u_prm[i,j] = du[i,j]*(p_prm[i,j]-p_prm[i-1,j])
        
    else:
        u_prm[i,j] = du[i,j]*(p_prm[i,j]-p_prm[i+1,j])

def vcorrect(dims,v_prm,du,p_prm):
    if i == dims+1:
        v_prm[i,j] = du[i,j]*(p_prm[i,j]-p_prm[i-1,j])
    else:
        v_prm[i,j] = du[i,j]*(p_prm[i,j]-p_prm[i+1,j])
    v = v + v_prm

def pcorrect():
    p = p + (omega*p_prm)
    
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

P1_psols = []
P1_usols = []
P1_vsols = [] # initialize empty array to contain solutions for post-processing

def SIMPLE_sol_1(cv_arr,iter_lim):
    pstate = True #let dependent methods know this is Problem 1
    for dims in cv_arr:





###################################
#########  Problem #2 #############
###################################