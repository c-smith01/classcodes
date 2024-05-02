'''
Created by Coleman Smith on 1/23/24
NUEN/MEEN 644 HW1
Due 05 April 2024
'''

import matplotlib.pyplot as plt
import numpy as np
import os

#os.system('cls')

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
#print(u_0)

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
        jlim = dims-1
    else:
        jlim = dims
    for j in range(1,jlim):
        for i in range(0,dims):
            # Calculate diffusion strengths
            if i == dims-1:
                Deu[i,j] = 1e-3
            else:
                Deu[i,j] = mu_H2O*dy/dx

            if i == 0:
                Dwu[i,j] = 1e-3
            else:
                Dwu[i,j] = mu_H2O*dy/dx

            if j == 0:
                Dnu[i,j] = 1e-3
            else:
                Dnu[i,j] = mu_H2O*dx/dy

            if j == dims-1:
                Dsu[i,j] = 1e-3
            else:
                Dsu[i,j] = mu_H2O*dx/dy

            # Calculate flow strengths
            if i == dims-1:
                Feu[i,j] = 1e-3
            else:
                Feu[i,j] = rho_H2O*(0.5*(u[i,j] + u[i+1,j]))*dy

            if i == 0:
                Fwu[i,j] = 1e-3
            else:
                Fwu[i,j] = rho_H2O*(0.5*(u[i,j] + u[i-1,j]))*dy

            if j == 0:
                Fnu[i,j] = 1e-3
            else:
                Fnu[i,j] = rho_H2O*(0.5*(u[i,j] + u[i,j-1]))*dx

            if j == dims-1:
                Fsu[i,j] = 1e-3
            else:
                Fsu[i,j] = rho_H2O*(0.5*(u[i,j] + u[i,j+1]))*dx

            # Calculate Peclet #s
            if i == dims-1:
                Peu[i,j] = 1e-3
            else:
                Peu[i,j] = Feu[i,j]/Deu[i,j]

            if i == 0:
                Pwu[i,j] = 1e-3
            else:
                Pwu[i,j] = Fwu[i,j]/Dwu[i,j]

            if j == 0:
                Pnu[i,j] = 1e-3
            else:
                Pnu[i,j] = Fnu[i,j]/Dnu[i,j]

            if j == dims-1:
                Psu[i,j] = 1e-3
            else:
                Psu[i,j] = Fsu[i,j]/Dsu[i,j]

            aEu[i,j] = Deu[i,j]*np.max((0,(1-0.1*np.abs(Peu[i,j]))**5)) + np.max((0,(-Feu[i,j])))
            aWu[i,j] = Dwu[i,j]*np.max((0,(1-0.1*np.abs(Pwu[i,j]))**5)) + np.max((0,(Fwu[i,j])))
            aNu[i,j] = Dnu[i,j]*np.max((0,(1-0.1*np.abs(Pnu[i,j]))**5)) + np.max((0,(-Fnu[i,j])))
            aSu[i,j] = Dsu[i,j]*np.max((0,(1-0.1*np.abs(Psu[i,j]))**5)) + np.max((0,(Fsu[i,j])))
            aPu[i,j] = aEu[i,j]+aWu[i,j]+aNu[i,j]+aSu[i,j]
    
def usolve(dims,
           u,aEu,aWu,aSu,aNu,aPu,
           du,dy,pstate):
    if pstate == True:
        jlim = dims-1
    else:
        jlim = dims
    for j in range(1,jlim):
        for i in range(0,dims):
            if i == 0 and j!=0 and j!=dims-1:
                u[i,j] = (aEu[i,j]*u[i+1,j]+aNu[i,j]*u[i,j+1]+aSu[i,j]*u[i,j+1])*(omega/aPu[i,j])
                du[i,j] = dy/((aPu[i,j]/omega)-(aEu[i,j]+aWu[i,j]+aNu[i,j]+aSu[i,j]))
            elif i == dims-1 and j!=0 and j!=dims-1:
                u[i,j] = (aWu[i,j]*u[i-1,j]+aNu[i,j]*u[i,j+1]+aSu[i,j]*u[i,j+1])*(omega/aPu[i,j])
                du[i,j] = dy/((aPu[i,j]/omega)-(aEu[i,j]+aWu[i,j]+aNu[i,j]+aSu[i,j]))
            elif j == 0 and i!=0 and i!=dims-1:
                u[i,j] = (aEu[i,j]*u[i+1,j]+aWu[i,j]*u[i-1,j]+aSu[i,j]*u[i,j+1])*(omega/aPu[i,j])
                du[i,j] = dy/((aPu[i,j]/omega)-(aEu[i,j]+aWu[i,j]+aNu[i,j]+aSu[i,j]))
            elif j == dims-1 and i!=0 and i!=dims-1:
                u[i,j] = (aEu[i,j]*u[i+1,j]+aWu[i,j]*u[i-1,j]+aNu[i,j]*u[i,j-1])*(omega/aPu[i,j])
                du[i,j] = dy/((aPu[i,j]/omega)-(aEu[i,j]+aWu[i,j]+aNu[i,j]+aSu[i,j]))
            elif i == 0 and j == 0:
                u[i,j] = (aEu[i,j]*u[i+1,j]+aNu[i,j]*u[i,j+1])*(omega/aPu[i,j])
                du[i,j] = dy/((aPu[i,j]/omega)-(aEu[i,j]+aWu[i,j]+aNu[i,j]+aSu[i,j]))
            elif i == 0 and j == dims-1:
                u[i,j] = (aEu[i,j]*u[i+1,j]+aNu[i,j]*u[i,j-1])*(omega/aPu[i,j])
                du[i,j] = dy/((aPu[i,j]/omega)-(aEu[i,j]+aWu[i,j]+aNu[i,j]+aSu[i,j]))
            elif i == dims-1 and j == dims-1:
                u[i,j] = (aWu[i,j]*u[i-1,j]+aNu[i,j]*u[i,j-1])*(omega/aPu[i,j])
                du[i,j] = dy/((aPu[i,j]/omega)-(aEu[i,j]+aWu[i,j]+aNu[i,j]+aSu[i,j]))
            elif i == dims-1 and j == 0:
                u[i,j] = (aEu[i,j]*u[i-1,j]+aNu[i,j]*u[i,j+1])*(omega/aPu[i,j])
                du[i,j] = dy/((aPu[i,j]/omega)-(aEu[i,j]+aWu[i,j]+aNu[i,j]+aSu[i,j]))
            else:
                u[i,j] = (aEu[i,j]*u[i+1,j]+aWu[i,j]*u[i-1,j]+aNu[i,j]*u[i,j-1]+aSu[i,j]*u[i,j+1])*(omega/aPu[i,j])
                du[i,j] = dy/((aPu[i,j]/omega)-(aEu[i,j]+aWu[i,j]+aNu[i,j]+aSu[i,j]))

def vcoeffs(dims,
            Dev,Dwv,Dnv,Dsv,
            Fev,Fwv,Fnv,Fsv,
            Pev,Pwv,Pnv,Psv,
            aEv,aWv,aNv,aSv,aPv,
            v,dx,dy):
    for j in range(0,dims):
        for i in range(0,dims):
            # Calculate diffusion strengths
            if i == dims-1:
                Dev[i,j] = 1e-3
            else:
                Dev[i,j] = mu_H2O*dy/dx

            if i == 0:
                Dwv[i,j] = 1e-3
            else:
                Dwv[i,j] = mu_H2O*dy/dx

            if j == 0:
                Dnv[i,j] = 1e-3
            else:
                Dnv[i,j] = mu_H2O*dx/dy

            if j == dims-1:
                Dsv[i,j] = 1e-3
            else:
                Dsv[i,j] = mu_H2O*dx/dy

            # Calculate flow strengths
            if i == dims-1:
                Fev[i,j] = 1e-3
            else:    
                Fev[i,j] = rho_H2O*(0.5*(v[i,j] + v[i+1,j]))*dy

            if i == 0:
                Fwv[i,j] = 1e-3
            else:
                Fwv[i,j] = rho_H2O*(0.5*(v[i,j] + v[i-1,j]))*dy

            if j == 0:
                Fnv[i,j] = 1e-3
            else:
                Fnv[i,j] = rho_H2O*(0.5*(v[i,j] + v[i,j-1]))*dx

            if j == dims-1:
                Fsv[i,j] = 1e-3
            else:
                Fsv[i,j] = rho_H2O*(0.5*(v[i,j] + v[i,j+1]))*dx

            # Calculate Peclet #s
            if i == dims-1:
                Pev[i,j] = 1e-3
            else:
                Pev[i,j] = Fev[i,j]/Dev[i,j]

            if i == 0:
                Pwv[i,j] = 1e-3
            else:
                Pwv[i,j] = Fwv[i,j]/Dwv[i,j]

            if j == 0:
                Pnv[i,j] = 1e-3
            else:
                Pnv[i,j] = Fnv[i,j]/Dnv[i,j]

            if j == dims-1:
                Psv[i,j] = 1e-3
            else:
                Psv[i,j] = Fsv[i,j]/Dsv[i,j]

            aEv[i,j] = Dev[i,j]*np.max((0,(1-0.1*np.abs(Pev[i,j]))**5)) + np.max((0,(-Fev[i,j])))
            aWv[i,j] = Dwv[i,j]*np.max((0,(1-0.1*np.abs(Pwv[i,j]))**5)) + np.max((0,(Fwv[i,j])))
            aNv[i,j] = Dnv[i,j]*np.max((0,(1-0.1*np.abs(Pnv[i,j]))**5)) + np.max((0,(-Fnv[i,j])))
            aSv[i,j] = Dsv[i,j]*np.max((0,(1-0.1*np.abs(Psv[i,j]))**5)) + np.max((0,(Fsv[i,j])))
            aPv[i,j] = aEv[i,j]+aWv[i,j]+aNv[i,j]+aSv[i,j]    

def vsolve(dims,
           v,aEv,aWv,aSv,aNv,aPv,
           dv,dx):
    for i in range(0,dims):
        for j in range(0,dims):
            if i == 0 and j!=0 and j!=dims-1:
                v[i,j] = (aEv[i,j]*v[i+1,j]+aNv[i,j]*v[i,j+1]+aSv[i,j]*v[i,j-1])*(omega/aPv[i,j])
                dv[i,j] = dx/((aPv[i,j]/omega)-(aEv[i,j]+aWv[i,j]+aNv[i,j]+aSv[i,j]))
            elif i == dims-1 and j!=0 and j!=dims-1:
                v[i,j] = (aWv[i,j]*v[i-1,j]+aNv[i,j]*v[i,j+1]+aSv[i,j]*v[i,j-1])*(omega/aPv[i,j])
                dv[i,j] = dx/((aPv[i,j]/omega)-(aEv[i,j]+aWv[i,j]+aNv[i,j]+aSv[i,j]))
            elif j == 0 and i!=0 and i!=dims-1:
                v[i,j] = (aEv[i,j]*v[i+1,j]+aWv[i,j]*v[i-1,j]+aSv[i,j]*v[i,j+1])*(omega/aPv[i,j])
                dv[i,j] = dx/((aPv[i,j]/omega)-(aEv[i,j]+aWv[i,j]+aNv[i,j]+aSv[i,j]))
            elif j == dims-1 and i!=0 and i!=dims-1:
                v[i,j] = (aEv[i,j]*v[i+1,j]+aWv[i,j]*v[i-1,j]+aSv[i,j]*v[i,j-1])*(omega/aPv[i,j])
                dv[i,j] = dx/((aPv[i,j]/omega)-(aEv[i,j]+aWv[i,j]+aNv[i,j]+aSv[i,j]))
            elif i == 0 and j == 0:
                v[i,j] = (aEv[i,j]*v[i+1,j]+aSv[i,j]*v[i,j+1])*(omega/aPv[i,j])
                dv[i,j] = dx/((aPv[i,j]/omega)-(aEv[i,j]+aWv[i,j]+aNv[i,j]+aSv[i,j]))
            elif i == 0 and j == dims-1:
                v[i,j] = (aEv[i,j]*v[i+1,j]+aSv[i,j]*v[i,j-1])*(omega/aPv[i,j])
                dv[i,j] = dx/((aPv[i,j]/omega)-(aEv[i,j]+aWv[i,j]+aNv[i,j]+aSv[i,j]))
            elif i == dims-1 and j == dims-1:
                v[i,j] = (aWv[i,j]*v[i-1,j]+aNv[i,j]*v[i,j-1])*(omega/aPv[i,j])
                dv[i,j] = dx/((aPv[i,j]/omega)-(aEv[i,j]+aWv[i,j]+aNv[i,j]+aSv[i,j]))
            elif i == dims-1 and j == 0:
                v[i,j] = (aWv[i,j]*v[i-1,j]+aSv[i,j]*v[i,j+1])*(omega/aPv[i,j])
                dv[i,j] = dx/((aPv[i,j]/omega)-(aEv[i,j]+aWv[i,j]+aNv[i,j]+aSv[i,j]))
            else:
                v[i,j] = (aEv[i,j]*v[i+1,j]+aWv[i,j]*v[i-1,j]+aNv[i,j]*v[i,j+1]+aSv[i,j]*v[i,j-1])*(omega/aPv[i,j])
                dv[i,j] = dx/((aPv[i,j]/omega)-(aEv[i,j]+aWv[i,j]+aNv[i,j]+aSv[i,j]))
    
def pccoeffs(dims,
             u,v,
             bPP,aEP,aWP,aNP,aSP,aPP,
             du,dv,dx,dy):
    for j in range(0,dims):
        for i in range(0,dims): 
            if i == 0  and j!=0:
                bPP[i,j] = rho_H2O*dy*(u[i+1,j]-u[i,j]) + rho_H2O*dx*(v[i,j-1]-v[i,j])
            elif j == 0:
                bPP[i,j] = rho_H2O*dy*(u[i-1,j]-u[i,j]) + rho_H2O*dx*(v[i,j+1]-v[i,j])
            elif i == 0 and j == 0:
                bPP[i,j] = rho_H2O*dy*(u[i+1,j]-u[i,j]) + rho_H2O*dx*(v[i,j+1]-v[i,j])
            else:
                bPP[i,j] = rho_H2O*dy*(u[i-1,j]-u[i,j]) + rho_H2O*dx*(v[i,j-1]-v[i,j])
            aEP[i,j] = rho_H2O*du[i,j]*dy
            aWP[i,j] = rho_H2O*du[i,j]*dy
            aNP[i,j] = rho_H2O*dv[i,j]*dx
            aSP[i,j] = rho_H2O*dv[i,j]*dx
            aPP[i,j] = aEP[i,j]+aWP[i,j]+aNP[i,j]+aSP[i,j]
    
def pcsolve(dims,
            p,p_prm,aEP,aWP,aNP,aSP,aPP):
    for j in range(0,dims):
        for i in range(0,dims):
            if i == 0 and j!=0 and j!=dims-1:
                p_prm[i,j] = (aEP[i,j]*p[i+1,j]+aNP[i,j]*p[i,j+1]+aSP[i,j]*p[i,j-1])*(omega/aPP[i,j])
            elif i == dims-1 and j!=0 and j!=dims-1:
                p_prm[i,j] = (aWP[i,j]*p[i-1,j]+aNP[i,j]*p[i,j+1]+aSP[i,j]*p[i,j-1])*(omega/aPP[i,j])
            elif j == dims-1 and i!=0 and i!=dims-1:
                p_prm[i,j] = (aEP[i,j]*p[i+1,j]+aWP[i,j]*p[i-1,j]+aSP[i,j]*p[i,j-1])*(omega/aPP[i,j])
            elif j == 0 and i!=0 and i!=dims-1:
                p_prm[i,j] = (aEP[i,j]*p[i+1,j]+aWP[i,j]*p[i-1,j]+aNP[i,j]*p[i,j+1])*(omega/aPP[i,j])
            elif i == 0 and j == 0:
                p_prm[i,j] = (aEP[i,j]*p[i+1,j]+aNP[i,j]*p[i,j+1])*(omega/aPP[i,j])
            elif i == 0 and j == dims-1:
                p_prm[i,j] = (aEP[i,j]*p[i+1,j]+aSP[i,j]*p[i,j-1])*(omega/aPP[i,j])
            elif i == dims-1 and j == dims-1:
                p_prm[i,j] = (aWP[i,j]*p[i-1,j]+aSP[i,j]*p[i,j-1])*(omega/aPP[i,j])
            elif i == dims-1 and j == 0:
                p_prm[i,j] = (aWP[i,j]*p[i-1,j]+aNP[i,j]*p[i,j+1])*(omega/aPP[i,j])
            else:
                p_prm[i,j] = (aEP[i,j]*p[i+1,j]+aWP[i,j]*p[i-1,j]+aNP[i,j]*p[i,j+1]+aSP[i,j]*p[i,j-1])*(omega/aPP[i,j])
    
def ucorrect(dims,
             u,u_prm,du,p_prm):
    for j in range(0,dims):
        for i in range(0,dims):
            if i == 0:
                u_prm[i,j] = du[i,j]*(p_prm[i,j]-p_prm[i+1,j])
            else:
                u_prm[i,j] = du[i,j]*(p_prm[i,j]-p_prm[i-1,j])
    u = u + u_prm

def vcorrect(dims,
             v,v_prm,dv,p_prm):
    for j in range(0,dims):
        for i in range(0,dims):
            if i == 0:
                v_prm[i,j] = dv[i,j]*(p_prm[i,j]-p_prm[i+1,j])
            else:
                v_prm[i,j] = dv[i,j]*(p_prm[i,j]-p_prm[i-1,j])
    v = v + v_prm

def pcorrect(p,p_prm):
    p = p + (omega*p_prm)
    
def conv_check(dx,dy,u,v,
               aEu,aWu,aNu,aSu,aPu,
               aEv,aWv,aNv,aSv,aPv):
    Ru = (np.abs(np.sum(np.multiply(aPu,u)-np.multiply(aEu,u)-np.multiply(aWu,u)-np.multiply(aNu,u)-np.multiply(aSu,u))))/(np.sum(np.multiply(aPu,u)))

    Rv = (np.abs(np.sum(np.multiply(aPv,v)-np.multiply(aEv,v)-np.multiply(aWv,v)-np.multiply(aNv,v)-np.multiply(aSv,v))))/(np.sum(np.multiply(aPv,v)))

    Rp = (np.sum(rho_H2O*u - rho_H2O*u*dy - rho_H2O*u - rho_H2O*u*dx))/(rho_H2O*u_0*L)
    return Rp, Ru, Rv

def print_results(p,u,v,itercount,Rp,Ru,Rv):
    print('SIMPLE Algorithm terminated at {} iterations with Rp = {}, Ru = {}, Rv = {}'.format(itercount,Rp,Ru,Rv))
    print('Printing pressure solutions')
    print(p)
    print('Printing u-vel fields')
    print(u)
    print('Printing v-vel fields')
    print(v)



###################################
#########  Problem #1 #############
###################################

def SIMPLE_sol(cv_arr,iter_lim,pstate):
    
    #init solutions arrays
    psols = []
    usols = []
    vsols = []
        
    for numvol in cv_arr:

        #init geometry
        dx = dy = L/numvol
        cv_size = numvol+2
        bnd_size = (cv_size, cv_size)

        #init necessary matrices
        p = v = u = np.ones(bnd_size)
        p_prm = v_prm = u_prm = np.zeros(bnd_size)
        dv = du = np.ones(bnd_size)
        Deu = Dwu = Dnu = Dsu = Dev = Dwv = Dnv = Dsv = np.ones(bnd_size)
        Feu = Fwu = Fnu = Fsu = Fev = Fwv = Fnv = Fsv = np.ones(bnd_size)
        Peu = Pwu = Pnu = Psu = Pev = Pwv = Pnv = Psv = np.ones(bnd_size)
        Peu = Pwu = Pnu = Psu = Pev = Pwv = Pnv = Psv = np.ones(bnd_size)
        aEu = aWu = aNu = aSu = aPu = aEv = aWv = aNv = aSv = aPv = np.ones(bnd_size)
        aEP = aWP = aNP = aSP = aPP = bPP = np.ones(bnd_size)

        Rp = Ru = Rv = 1 # start residuals with values greater than tolerance to force at least one iteration
        itercount    = 1 # start count of iterations to convergence

        while np.abs(Rp)>Rp_tol and np.abs(Ru)>Ru_tol and np.abs(Rv)>Rv_tol and itercount<iter_lim:
            bnd_conds(u_matr=u,pstate=pstate)
            reset(matr=p_prm,dims=cv_size)
            #reset(matr=u_prm,dims=cv_size)
            #reset(matr=v_prm,dims=cv_size)
            ucoeffs(dims=cv_size,
                    Deu=Deu,Dwu=Dwu,Dnu=Dnu,Dsu=Dsu,
                    Feu=Feu,Fwu=Fwu,Fnu=Fnu,Fsu=Fsu,
                    Peu=Peu,Pwu=Pwu,Pnu=Pnu,Psu=Psu,
                    aEu=aEu,aWu=aWu,aNu=aNu,aSu=aSu,aPu=aPu,
                    u=u,pstate=pstate,dy=dy,dx=dx)
            usolve(dims=cv_size,
                   u=u,aEu=aEu,aWu=aWu,aSu=aSu,aNu=aNu,aPu=aPu,
                   du=du,dy=dy,pstate=pstate)
            vcoeffs(dims=cv_size,
                    Dev=Dev,Dwv=Dwv,Dnv=Dnv,Dsv=Dsv,
                    Fev=Fev,Fwv=Fwv,Fnv=Fnv,Fsv=Fsv,
                    Pev=Pev,Pwv=Pwv,Pnv=Pnv,Psv=Psv,
                    aEv=aEv,aWv=aWv,aNv=aNv,aSv=aSv,aPv=aPv,
                    v=v,dx=dx,dy=dx)
            vsolve(dims=cv_size,
                   v=v,aEv=aEv,aWv=aWv,aSv=aSv,aNv=aNv,aPv=aPv,
                   dv=dv,dx=dx)
            pccoeffs(dims=cv_size,
                     u=u,v=v,
                     bPP=bPP,aEP=aEP,aWP=aWP,aNP=aNP,aSP=aSP,aPP=aPP,
                     du=du,dv=dv,dx=dx,dy=dy)
            pcsolve(dims=cv_size,
                    p=p,p_prm=p_prm,aEP=aEP,aWP=aWP,aNP=aNP,aSP=aSP,aPP=aPP)
            ucorrect(dims=cv_size,
                     u=u,u_prm=u_prm,du=du,p_prm=p_prm)
            vcorrect(dims=cv_size,
                     v=v,v_prm=v_prm,dv=dv,p_prm=p_prm)
            pcorrect(p=p,p_prm=p_prm)
            Rp,Ru,Rv = conv_check(dx=dx,dy=dy,u=u,v=v,
                                  aEu=aEu,aWu=aWu,aNu=aNu,aSu=aSu,aPu=aPu,
                                  aEv=aEv,aWv=aWv,aNv=aNv,aSv=aSv,aPv=aPv)

            itercount +=1 # add one to iteration tally

        print_results(p=p,u=u,v=v,itercount=itercount,Rp=Rp,Ru=Ru,Rv=Rv)
        psols.append(p)
        usols.append(u)
        vsols.append(v)

    return [psols,usols,vsols]

prob_one_sol = SIMPLE_sol(cv_arr=N_CVs_one,iter_lim=3,pstate=True)

x = np.linspace(0,L,N_CVs_one[0]+2)
y = np.linspace(0,L,N_CVs_one[0]+2)
X, Y = np.meshgrid(x, y)

#print(prob_one_sol[0])

#Create a contour plot
# plt.figure()
# cp = plt.contour(X, Y, np.concatenate(prob_one_sol[0]), colors='black', linestyles='dashed')
# plt.clabel(cp, inline=True, fontsize=10)
# plt.title('Contour Plot of 2-D Pressure Array for a(n) 5 by 5 Grid')
# plt.xlabel('X-axis [m]')
# plt.ylabel('Y-axis [m]')
# plt.show()

# plt.figure()
# cp = plt.contour(X, Y, np.concatenate(prob_one_sol[1]), colors='black', linestyles='dashed')
# plt.clabel(cp, inline=True, fontsize=10)
# plt.title('Contour Plot of 2-D V-Velocity Array for a(n) 5 by 5 Grid')
# plt.xlabel('X-axis [m]')
# plt.ylabel('Y-axis [m]')
# plt.show()

# plt.figure()
# cp = plt.contour(X, Y, np.concatenate(prob_one_sol[2]), colors='black', linestyles='dashed')
# plt.clabel(cp, inline=True, fontsize=10)
# plt.title('Contour Plot of 2-D U-Velocity Array for a(n) 5 by 5 Grid')
# plt.xlabel('X-axis [m]')
# plt.ylabel('Y-axis [m]')
# plt.show()

###################################
#########  Problem #2 #############
###################################

test_N_CVs_two = N_CVs_two[0:2] # testing array looking at only smallest CVs for debugging
#print(test_N_CVs_two)

prob_two_sol = SIMPLE_sol(cv_arr=N_CVs_two,iter_lim=3,pstate=True)
#print(prob_two_sol)

for solsets in prob_two_sol:
    print(len(solsets[0]))
    
    
# Plot solution fields to satisfy 2a
# for s in range(0,len(N_CVs_two)):
    
#     x = np.linspace(0, L, N_CVs_two[s]+2)
#     y = np.linspace(0, L, N_CVs_two[s]+2)
#     Xg, Yg = np.meshgrid(x, y)
#     ptext = f'Contour Plot of 2-D Pressure Array for a(n) {N_CVs_two[s]} by {N_CVs_two[s]} Grid'
#     utext = f'Contour Plot of 2-D U-Velocity Array for a(n) {N_CVs_two[s]} by {N_CVs_two[s]} Grid'
#     vtext = f'Contour Plot of 2-D V-Velocity Array for a(n) {N_CVs_two[s]} by {N_CVs_two[s]} Grid'
    
    
#     #print(x,y,prob_two_sol[0][s][:])
#     plt.figure()
#     cp = plt.contour(Xg, Yg, prob_two_sol[0][s], colors='black', linestyles='dashed')
#     plt.clabel(cp, inline=True, fontsize=10)
#     plt.title(ptext)
#     plt.xlabel('X-axis [m]')
#     plt.ylabel('Y-axis [m]')
#     plt.show()

#     plt.figure()
#     cp = plt.contour(Xg, Yg, prob_two_sol[1][s], colors='black', linestyles='dashed')
#     plt.clabel(cp, inline=True, fontsize=10)
#     plt.title(vtext)
#     plt.xlabel('X-axis [m]')
#     plt.ylabel('Y-axis [m]')
#     plt.show()

#     plt.figure()
#     cp = plt.contour(Xg, Yg, prob_two_sol[0][s], colors='black', linestyles='dashed')
#     plt.clabel(cp, inline=True, fontsize=10)
#     plt.title(utext)
#     plt.xlabel('X-axis [m]')
#     plt.ylabel('Y-axis [m]')
#     plt.show()
    
# Plot centerline U and V-Vels for each case
center_us = []
center_vs = []
all_us = []
all_vs = []

plt.figure()
cp = plt.contour(Xg, Yg, prob_two_sol[0][s], colors='black', linestyles='dashed')
plt.clabel(cp, inline=True, fontsize=10)
plt.title('Centerline U-velocity plotted against grid dimensions')
plt.xlabel('# of CVs (NxN)')
plt.ylabel('Centerline U-velocity')
plt.show()

plt.figure()
cp = plt.contour(Xg, Yg, prob_two_sol[0][s], colors='black', linestyles='dashed')
plt.clabel(cp, inline=True, fontsize=10)
plt.title('Centerline V-velocity plotted against grid dimensions')
plt.xlabel('# of CVs (NxN)')
plt.ylabel('Centerline V-velocity')
plt.show()

# Calculate and print order of convergence and GCI at the centerpoint of each using p-field solutions
GCIs = []
OoCs = []

print(N_CVs_two)
print(GCIs)
print(OoCs)
# Print tabulated solutions to txt files
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