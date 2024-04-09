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

def bnd_conds(u_matr,pstate):
    if pstate == True:
        u_matr[:][0] = u_0
        u_matr[:][-1] = u_0
    else:
        u_matr[:][0] = u_0

def reset(matr,dims):
    matr = np.zeros(dims)

def ucoeffs(dims,
            Deu,Dwu,Dnu,Dsu,
            Feu,Fwu,Fnu,Fsu,
            Peu,Pwu,Pnu,Psu,
            aEu,aWu,aNu,aSu,aPu,
            u,pstate,dy,dx):
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
            v,pstate,dx,dy):
    

def vsolve(dims,v,aEv,aWv,aSv,aNv,aPv,pstate):
    
def pcoeffs(dims,u,v,p,
            bPP,aEP,aWP,aNP,aSP,aPP,
            du,dv,dx,dy,jlim):
    for j in range(1,jlim):
        for i in range(0,dims): 
            bPP[i,j] = rho_H2O*dy*(u[i-1,j]-u[i,j]) + rho_H2O*dx*(v[i,j-1]-v[i,j])
            aEP[i,j] = rho_H2O*du[i,j]*dy
            aWP[i,j] = rho_H2O*du[i,j]*dy
            aNP[i,j] = rho_H2O*dv[i,j]*dx
            aSP[i,j] = rho_H2O*dv[i,j]*dx
            aPP[i,j] = aEP[i,j]+aWP[i,j]+aNP[i,j]+aSP[i,j]
    
def psolve(p,p_prm,aEP,aWP,aNP,aSP,aPP):
    for j in range(1,jlim):
        for i in range(0,dims):
            p_prm[i,j] = (aEP[i,j]*p[i+1,j]+aWP[i,j]*p[i-1,j]+aNP[i,j]*p[i,j+1]+aSP[i,j]*p[i,j-1])*(omega/aPP[i,j])
    
def ucorrect(dims,u_prm,du,p_prm):
    for j in range(1,jlim):
        for i in range(0,dims):
            u_prm[i,j] = du[i,j]*(p_prm[i,j]-p_prm[i-1,j])
    u = u + u_prm

def vcorrect(dims,v_prm,dv,p_prm):
    for j in range(1,jlim):
        for i in range(0,dims):
            v_prm[i,j] = dv[i,j]*(p_prm[i,j]-p_prm[i-1,j])
    v = v + v_prm

def pcorrect(p,p_prm):
    p = p + (omega*p_prm)
    
def conv_check(p,u,v,
               aEu,aWu,aNu,aSu,aPu,
               aEv,aWv,aNv,aSv,aPv):
    Rp = (np.sum(rho_H2O*u-rho_H2O*u,rho_H2O*u-rho_H2O*u))/(rho_H2O*u_0*L)
    Ru = (np.sum(np.multiply(aPu*u)-np.multiply(aEu*u)-np.multiply(aWu*u)-np.multiply(aNu*u)-np.multiply(aSu*u)))/(np.sum(np.multiply(aPu*u)))
    Rv = (np.sum(np.multiply(aPv*u)-np.multiply(aEv*u)-np.multiply(aWv*u)-np.multiply(aNv*u)-np.multiply(aSv*u)))/(np.sum(np.multiply(aPu*u)))

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

def SIMPLE_sol(cv_arr,iter_lim,pstate):
    
    for dims in cv_arr:
        #init geometry
        dx = dy = L/dims
        bnd_size = (dims,dims)

        #init necessary matrices
        p = v = u = np.ones(bnd_size)
        p_prm = v_prm = u_prm = np.ones(bnd_size)
        dv = du = np.ones(bnd_size)
        Deu = Dwu = Dnu = Dsu = Dev = Dwv = Dnv = Dsv = np.ones(bnd_size)
        Feu = Fwu = Fnu = Fsu = Fev = Fwv = Fnv = Fsv = np.ones(bnd_size)
        Peu = Pwu = Pnu = Psu = Pev = Pwv = Pnv = Psv = np.ones(bnd_size)
        Peu = Pwu = Pnu = Psu = Pev = Pwv = Pnv = Psv = np.ones(bnd_size)
        aEu = aWu = aNu = aSu = aPu = aEu = aWu = aNu = aSu = aPu = np.ones(bnd_size)
        aEP = aWP = aNP = aSP = aPP = bPP = np.ones(bnd_size)

        Rp = Ru = Rv = 1 #start residuals with values greater than tolerance to force at least one iteration
        itercount = 1 # start count of iterations to convergence

        while Rp>Rp_tol or Ru>Ru_tol or Rv>Rv_tol or itercount<iter_lim:
            bnd_conds(u_matr=u,pstate=pstate)

            


        print_res()





###################################
#########  Problem #2 #############
###################################