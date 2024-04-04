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
u_0     = (Re*mu_H2O)/(rho_H2O*L)
N_CVs   = [5, 8, 16, 64, 128, 256]

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

def SIMPLE_sol_P1(gridsize,iterlim):

    # Initialize all necessary grids, add array of grids for coefficients calculated for u, v, and P at each iteration
    dims    = (gridsize+2,gridsize+2)
    dx      = L/gridsize
    dy      = dx
    du      = np.ones(dims)
    dv      = np.ones(dims)
    Feu      = np.ones(dims)
    Fwu      = np.ones(dims)
    Fnu      = np.ones(dims)
    Fsu      = np.ones(dims)
    Deu      = np.ones(dims)
    Dwu      = np.ones(dims)
    Dnu      = np.ones(dims)
    Dsu      = np.ones(dims)
    Peu      = np.ones(dims)
    Pwu      = np.ones(dims)
    Pnu      = np.ones(dims)
    Psu      = np.ones(dims)
    aEu      = np.ones(dims)
    aWu      = np.ones(dims)
    aNu      = np.ones(dims)
    aSu      = np.ones(dims)
    aPu      = np.ones(dims)
    Fev      = np.ones(dims)
    Fwv      = np.ones(dims)
    Fnv      = np.ones(dims)
    Fsv      = np.ones(dims)
    Dev      = np.ones(dims)
    Dwv      = np.ones(dims)
    Dnv      = np.ones(dims)
    Dsv      = np.ones(dims)
    Pev      = np.ones(dims)
    Pwv      = np.ones(dims)
    Pnv      = np.ones(dims)
    Psv      = np.ones(dims)
    aEv      = np.ones(dims)
    aWv      = np.ones(dims)
    aNv      = np.ones(dims)
    aSv      = np.ones(dims)
    aPv      = np.ones(dims)
    # FeP      = np.ones(dims)
    # FwP      = np.ones(dims)
    # FnP      = np.ones(dims)
    # FsP      = np.ones(dims)
    # DeP      = np.ones(dims)
    # DwP      = np.ones(dims)
    # DnP      = np.ones(dims)
    # DsP      = np.ones(dims)
    # PeP      = np.ones(dims)
    # PwP      = np.ones(dims)
    # PnP      = np.ones(dims)
    # PsP      = np.ones(dims)
    aEP      = np.ones(dims)
    aWP      = np.ones(dims)
    aNP      = np.ones(dims)
    aSP      = np.ones(dims)
    aPP      = np.ones(dims)
    bPP      = np.ones(dims)
    p       = np.ones(dims)
    u       = np.ones(dims)
    v       = np.ones(dims)
    p_prm  = np.ones(dims)
    u_prm  = np.ones(dims)
    v_prm  = np.ones(dims)
    iternum = 0
    
    while Ru>Ru_tol or Rv>Rv_tol or Rp>Rp_tol and iternum<iterlim:
        for j in range(0,gridsize+2):
            for i in range(0,gridsize+2):
                if i == 0 or i == gridsize+1 or j == 0 or j == gridsize+1:
                    # Start with u vel-calcs
                    if j == 0:
                        # Calculate diffusion strengths
                        Deu[i,j] = mu_H2O*dy/dx
                        Dwu[i,j] = mu_H2O*dy/dx
                        Dnu[i,j] = mu_H2O*dx/dy
                        Dsu[i,j] = mu_H2O*dx/(dy/2)
                    elif j == gridsize+1:
                        # Calculate diffusion strengths
                        Deu[i,j] = mu_H2O*dy/dx
                        Dwu[i,j] = mu_H2O*dy/dx
                        Dnu[i,j] = mu_H2O*dx/(dy/2)
                        Dsu[i,j] = mu_H2O*dy/dx
                    else:
                        # Calculate diffusion strengths
                        Deu[i,j] = mu_H2O*dy/dx
                        Dwu[i,j] = mu_H2O*dy/dx
                        Dnu[i,j] = mu_H2O*dy/dx
                        Dsu[i,j] = mu_H2O*dy/dx

                    # Then v vel-calcs
                    if i == 0:
                        # Calculate diffusion strengths
                        Dev[i,j] = mu_H2O*dy/(dx/2)
                        Dwv[i,j] = mu_H2O*dy/dx
                        Dnv[i,j] = mu_H2O*dx/dy
                        Dsv[i,j] = mu_H2O*dx/dy
                    elif i == gridsize+1:
                        # Calculate diffusion strengths
                        Dev[i,j] = mu_H2O*dy/dx
                        Dwv[i,j] = mu_H2O*dy/(dx/2)
                        Dnv[i,j] = mu_H2O*dx/dy
                        Dsv[i,j] = mu_H2O*dx/dy
                    else:
                        # Calculate diffusion strengths
                        Dev[i,j] = mu_H2O*dy/dx
                        Dwv[i,j] = mu_H2O*dy/dx
                        Dnv[i,j] = mu_H2O*dy/dx
                        Dsv[i,j] = mu_H2O*dy/dx

                    if i == 0: #find better way to implement this
                        # Calculate flow strengths
                        Feu[i,j] = rho_H2O*(0.5*(u[i,j] + u[i+1,j]))*dy
                        Fwu[i,j] = rho_H2O*(0.5*(u[i,j] + 0))*dy
                        Fnu[i,j] = rho_H2O*(0.5*(u[i,j] + u[i,j-1]))*dx
                        Fsu[i,j] = rho_H2O*(0.5*(u[i,j] + u[i,j-1]))*dx

                        # Calculate Peclet #s
                        Peu[i,j] = Feu[i,j]/Deu[i,j]
                        Pwu[i,j] = Fwu[i,j]/Dwu[i,j]
                        Pnu[i,j] = Fnu[i,j]/Dnu[i,j]
                        Psu[i,j] = Fsu[i,j]/Dsu[i,j]

                        # Calcuate coeffs
                        aEu[i,j] = Deu[i,j]*np.max(0,(1-0.1*np.abs(Peu[i,j]))^5) + np.max(0,(-Feu[i,j]))
                        aWu[i,j] = Dwu[i,j]*np.max(0,(1-0.1*np.abs(Pwu[i,j]))^5) + np.max(0,(-Fwu[i,j]))
                        aNu[i,j] = Dnu[i,j]*np.max(0,(1-0.1*np.abs(Pnu[i,j]))^5) + np.max(0,(-Fnu[i,j]))
                        aSu[i,j] = Dsu[i,j]*np.max(0,(1-0.1*np.abs(Psu[i,j]))^5) + np.max(0,(-Fsu[i,j]))
                        aPu[i,j] = aEu[i,j]+aWu[i,j]+aNu[i,j]+aSu[i,j]

                        # Then v terms

                        # Calculate flow strengths
                        Fev[i,j] = rho_H2O*(0.5*(v[i,j] + v[i+1,j]))*dy
                        Fwv[i,j] = rho_H2O*(0.5*(v[i,j] + v[i-1,j]))*dy
                        Fnv[i,j] = rho_H2O*(0.5*(v[i,j] + v[i,j-1]))*dx
                        Fsv[i,j] = rho_H2O*(0.5*(v[i,j] + v[i,j-1]))*dx

                        # Calculate Peclet #s
                        Pev[i,j] = Fev[i,j]/Dev[i,j]
                        Pwv[i,j] = Fwv[i,j]/Dwv[i,j]
                        Pnv[i,j] = Fnv[i,j]/Dnv[i,j]
                        Psv[i,j] = Fsv[i,j]/Dsv[i,j]

                        # Calcuate coeffs
                        aEv[i,j] = Dev[i,j]*np.max(0,(1-0.1*np.abs(Pev[i,j]))^5) + np.max(0,(-Fev[i,j]))
                        aWv[i,j] = Dwv[i,j]*np.max(0,(1-0.1*np.abs(Pwv[i,j]))^5) + np.max(0,(-Fwv[i,j]))
                        aNv[i,j] = Dnv[i,j]*np.max(0,(1-0.1*np.abs(Pnv[i,j]))^5) + np.max(0,(-Fnv[i,j]))
                        aSv[i,j] = Dsv[i,j]*np.max(0,(1-0.1*np.abs(Psv[i,j]))^5) + np.max(0,(-Fsv[i,j]))
                        aPv[i,j] = aEv[i,j]+aWv[i,j]+aNv[i,j]+aSv[i,j]

                        # Solve for u and v
                        u[i,j] = (aEu[i,j]*u[i+1,j]+aWu[i,j]*u[i-1,j]+aNu[i,j]*u[i,j+1]+aSu[i,j]*u[i,j-1])*(omega/aPu[i,j])
                        v[i,j] = (aEv[i,j]*v[i+1,j]+aWv[i,j]*v[i-1,j]+aNv[i,j]*v[i,j+1]+aSv[i,j]*v[i,j-1])*(omega/aPv[i,j])

                        # Calculate d_u & d_v
                        du[i,j] = dy/((aPu[i,j]/omega)-(aEu[i,j]+aWu[i,j]+aNu[i,j]+aSu[i,j]))
                        dv[i,j] = dx/((aPv[i,j]/omega)-(aEv[i,j]+aWv[i,j]+aNv[i,j]+aSv[i,j]))

                        # Solve pressure correction (p')
                        bPP[i,j] = rho_H2O*dy*(u[i-1,j]-u[i,j]) + rho_H2O*dx*(v[i,j-1]-v[i,j])
                        aEP[i,j] = rho_H2O*du[i,j]*dy
                        aWP[i,j] = rho_H2O*du[i,j]*dy
                        aNP[i,j] = rho_H2O*du[i,j]*dx
                        aSP[i,j] = rho_H2O*du[i,j]*dx
                        aPP[i,j] = aEP[i,j]+aWP[i,j]+aNP[i,j]+aSP[i,j]
                        
                        p_prm[i,j] = (aEP[i,j]*p[i+1,j]+aWP[i,j]*p[i-1,j]+aNP[i,j]*p[i,j+1]+aSP[i,j]*p[i,j-1])*(omega/aPP[i,j])

                        # Calculate velocity corrections (u' & v')
                        u_prm[i,j] = du[i,j]*(p_prm[i,j]-p_prm[i+1,j])
                        v_prm[i,j] = du[i,j]*(p_prm[i,j]-p_prm[i+1,j])

                        # Correct p, u, & v
                        p = p + (omega*p_prm)
                        u = u + u_prm
                        v = v + v_prm
                else:
                    # Start with u vel-calcs
                    if j == 1:
                        # Calculate diffusion strengths
                        Deu[i,j] = mu_H2O*dy/dx
                        Dwu[i,j] = mu_H2O*dy/dx
                        Dnu[i,j] = mu_H2O*dx/dy
                        Dsu[i,j] = mu_H2O*dx/(dy/2)
                    elif j == gridsize:
                        # Calculate diffusion strengths
                        Deu[i,j] = mu_H2O*dy/dx
                        Dwu[i,j] = mu_H2O*dy/dx
                        Dnu[i,j] = mu_H2O*dx/(dy/2)
                        Dsu[i,j] = mu_H2O*dy/dx
                    else:
                        # Calculate diffusion strengths
                        Deu[i,j] = mu_H2O*dy/dx
                        Dwu[i,j] = mu_H2O*dy/dx
                        Dnu[i,j] = mu_H2O*dy/dx
                        Dsu[i,j] = mu_H2O*dy/dx

                    # Calculate flow strengths
                    Feu[i,j] = rho_H2O*(0.5*(u[i,j] + u[i+1,j]))*dy
                    Fwu[i,j] = rho_H2O*(0.5*(u[i,j] + u[i-1,j]))*dy
                    Fnu[i,j] = rho_H2O*(0.5*(u[i,j] + u[i,j-1]))*dx
                    Fsu[i,j] = rho_H2O*(0.5*(u[i,j] + u[i,j-1]))*dx

                    # Calculate Peclet #s
                    Peu[i,j] = Feu[i,j]/Deu[i,j]
                    Pwu[i,j] = Fwu[i,j]/Dwu[i,j]
                    Pnu[i,j] = Fnu[i,j]/Dnu[i,j]
                    Psu[i,j] = Fsu[i,j]/Dsu[i,j]

                    # Calcuate coeffs
                    aEu[i,j] = Deu[i,j]*np.max(0,(1-0.1*np.abs(Peu[i,j]))^5) + np.max(0,(-Feu[i,j]))
                    aWu[i,j] = Dwu[i,j]*np.max(0,(1-0.1*np.abs(Pwu[i,j]))^5) + np.max(0,(-Fwu[i,j]))
                    aNu[i,j] = Dnu[i,j]*np.max(0,(1-0.1*np.abs(Pnu[i,j]))^5) + np.max(0,(-Fnu[i,j]))
                    aSu[i,j] = Dsu[i,j]*np.max(0,(1-0.1*np.abs(Psu[i,j]))^5) + np.max(0,(-Fsu[i,j]))
                    aPu[i,j] = aEu[i,j]+aWu[i,j]+aNu[i,j]+aSu[i,j]


                    # Then v vel-calcs
                    if i == 1:
                        # Calculate diffusion strengths
                        Dev[i,j] = mu_H2O*dy/(dx/2)
                        Dwv[i,j] = mu_H2O*dy/dx
                        Dnv[i,j] = mu_H2O*dx/dy
                        Dsv[i,j] = mu_H2O*dx/dy
                    elif i == gridsize:
                        # Calculate diffusion strengths
                        Dev[i,j] = mu_H2O*dy/dx
                        Dwv[i,j] = mu_H2O*dy/(dx/2)
                        Dnv[i,j] = mu_H2O*dx/dy
                        Dsv[i,j] = mu_H2O*dx/dy
                    else:
                        # Calculate diffusion strengths
                        Dev[i,j] = mu_H2O*dy/dx
                        Dwv[i,j] = mu_H2O*dy/dx
                        Dnv[i,j] = mu_H2O*dy/dx
                        Dsv[i,j] = mu_H2O*dy/dx

                    # Calculate flow strengths
                    Fev[i,j] = rho_H2O*(0.5*(v[i,j] + v[i+1,j]))*dy
                    Fwv[i,j] = rho_H2O*(0.5*(v[i,j] + v[i-1,j]))*dy
                    Fnv[i,j] = rho_H2O*(0.5*(v[i,j] + v[i,j-1]))*dx
                    Fsv[i,j] = rho_H2O*(0.5*(v[i,j] + v[i,j-1]))*dx

                    # Calculate Peclet #s
                    Pev[i,j] = Fev[i,j]/Dev[i,j]
                    Pwv[i,j] = Fwv[i,j]/Dwv[i,j]
                    Pnv[i,j] = Fnv[i,j]/Dnv[i,j]
                    Psv[i,j] = Fsv[i,j]/Dsv[i,j]

                    # Calcuate coeffs
                    aEv[i,j] = Dev[i,j]*np.max(0,(1-0.1*np.abs(Pev[i,j]))^5) + np.max(0,(-Fev[i,j]))
                    aWv[i,j] = Dwv[i,j]*np.max(0,(1-0.1*np.abs(Pwv[i,j]))^5) + np.max(0,(-Fwv[i,j]))
                    aNv[i,j] = Dnv[i,j]*np.max(0,(1-0.1*np.abs(Pnv[i,j]))^5) + np.max(0,(-Fnv[i,j]))
                    aSv[i,j] = Dsv[i,j]*np.max(0,(1-0.1*np.abs(Psv[i,j]))^5) + np.max(0,(-Fsv[i,j]))
                    aPv[i,j] = aEv[i,j]+aWv[i,j]+aNv[i,j]+aSv[i,j]

                    # Solve for u and v
                    u[i,j] = (aEu[i,j]*u[i+1,j]+aWu[i,j]*u[i-1,j]+aNu[i,j]*u[i,j+1]+aSu[i,j]*u[i,j-1])*(omega/aPu[i,j])
                    v[i,j] = (aEv[i,j]*v[i+1,j]+aWv[i,j]*v[i-1,j]+aNv[i,j]*v[i,j+1]+aSv[i,j]*v[i,j-1])*(omega/aPv[i,j])

                    # Calculate d_u & d_v
                    du[i,j] = dy/((aPu[i,j]/omega)-(aEu[i,j]+aWu[i,j]+aNu[i,j]+aSu[i,j]))
                    dv[i,j] = dx/((aPv[i,j]/omega)-(aEv[i,j]+aWv[i,j]+aNv[i,j]+aSv[i,j]))

                    # Solve pressure correction (p')
                    bPP[i,j] = rho_H2O*dy*(u[i-1,j]-u[i,j]) + rho_H2O*dx*(v[i,j-1]-v[i,j])
                    aEP[i,j] = rho_H2O*du[i,j]*dy
                    aWP[i,j] = rho_H2O*du[i,j]*dy
                    aNP[i,j] = rho_H2O*du[i,j]*dx
                    aSP[i,j] = rho_H2O*du[i,j]*dx
                    aPP[i,j] = aEP[i,j]+aWP[i,j]+aNP[i,j]+aSP[i,j]
                    
                    p_prm[i,j] = (aEP[i,j]*p[i+1,j]+aWP[i,j]*p[i-1,j]+aNP[i,j]*p[i,j+1]+aSP[i,j]*p[i,j-1])*(omega/aPP[i,j])

                    # Calculate velocity corrections (u' & v')
                    u_prm[i,j] = du[i,j]*(p_prm[i,j]-p_prm[i+1,j])
                    v_prm[i,j] = du[i,j]*(p_prm[i,j]-p_prm[i+1,j])

                    # Correct p, u, & v
                    p = p + (omega*p_prm)
                    u = u + u_prm
                    v = v + v_prm

            # Convergence check 
            Rp = (np.sum(rho_H2O*u-rho_H2O*u,rho_H2O*u-rho_H2O*u))/(rho_H2O*u_0*L)
            Ru = (np.sum(np.multiply(aPu*u)-np.multiply(aEu*u)-np.multiply(aWu*u)-np.multiply(aNu*u)-np.multiply(aSu*u)))/(np.sum(np.multiply(aPu*u)))
            Rv = (np.sum(np.multiply(aPu*u)-np.multiply(aEu*u)-np.multiply(aWu*u)-np.multiply(aNu*u)-np.multiply(aSu*u)))/(np.sum(np.multiply(aPu*u)))

            # Zero correction terms
            p_prm = np.zeros(dims)
            u_prm = np.zeros(dims)
            v_prm = np.zeros(dims)

            iternum+=1
        
        # print converged results
        print(p,u,v)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.contourf(X, Y, p, cmap='viridis')
plt.colorbar()
plt.title('Pressure Field (dx={:.3f})'.format(dx))
plt.xlabel('X')
plt.ylabel('Y')

plt.subplot(1, 2, 2)
plt.quiver(X, Y, u, v)
plt.title('Velocity Field (dx={:.3f})'.format(dx))
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.show()


###################################
#########  Problem #2 #############
###################################

for griddims in N_CVs:
    print(griddims)
