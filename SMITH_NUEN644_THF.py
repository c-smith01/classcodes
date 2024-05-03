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
omega_u = omega_v            = 0.3                                            # given under-relaxation factors
omega_p                     = 0.7                                            # given under-relaxation factors
Ru_tol=Rv_tol=Rp_tol=Rt_tol = 1E-6                                           # Tolerance for u-vel residual
Re                          = 200                                            # Unitless Reynolds #
rho_f                       = 997.0                                          # kg/m^3
mu_f                        = 8.71E-4                                        # N*s/m^2
C_pf                        = 4179.0                                         # J/kg*K
k                           = 0.563                                          # W/m*K
u_0                         = (Re*mu_f)/(rho_f*L)                            # m/s
T_w                         = 100+273.15                                     # K
T_0                         = 27+273.15                                      # K
N_CVs_one                   = [[10,5]]                                       # Dimensions of CVs
N_CVs_two                   = [[10,5], [20,10], [60,20], [120,40], [160,80]] # Dimensions of CVs

# Iteratively-called methods
def reset(matr,dims):
    matr = np.zeros(dims)

def bnd_conds(p_matr,u_matr,v_matr,t_matr):
    
    # set bnd values for start and end columns
    p_matr[0][:] = p_matr[1][:]
    p_matr[-1][:] = p_matr[-2][:]
    u_matr[0][:] = u_0
    u_matr[-1][:] = u_matr[-2][:]
    v_matr[0][:] = 0
    v_matr[-1][:] = v_matr[-2][:]
    t_matr[0][:] = T_0
    t_matr[-1][:] = t_matr[-2][:]

    # set bnd values for start and end rows
    p_matr[:][0] = p_matr[:][1]
    p_matr[:][-1] = p_matr[:][-2]
    u_matr[:][0] = u_matr[:][1]
    u_matr[:][-1] = u_matr[:][-2]
    v_matr[:][0] = v_matr[:][1]
    v_matr[:][-1] = v_matr[:][-2]
    t_matr[:][0] = t_matr[:][1]
    t_matr[:][-1] = t_matr[:][-2]

def ucoeffs(dims,
            Deu,Dwu,Dnu,Dsu,
            Feu,Fwu,Fnu,Fsu,
            Peu,Pwu,Pnu,Psu,
            aEu,aWu,aNu,aSu,aPu,
            u,v,dy,dx):
    for j in range(1,dims-1):
        for i in range(1,dims-1):
            if i==1:
                Feu[i,j] = rho_f*(0.5*(u[i,j]+u[i+1]))*dy
                Fwu[i,j] = rho_f*(0.5*(u[i,j]+u[i-1]))*dy
                Vn = (0.25*v[i-1,j]+0.5*v[i,j]+0.75*v[i+1,j])/1.5
                Vs = (0.25*v[i-1,j+1]+0.5*v[i,j+1]+0.75*v[i+1,j+1])/1.5
                Fnu[i,j] = rho_f*(Vn)*1.5*dx
                Fsu[i,j] = rho_f*(Vs)*1.5*dx
            elif i==dims-1:
                Feu[i,j] = rho_f*(0.5*(u[i,j]+u[i+1]))*dy
                Fwu[i,j] = rho_f*(0.5*(u[i,j]+u[i-1]))*dy
                Vn = (0.75*v[i-1,j]+0.5*v[i,j]+0.25*v[i+1,j])/1.5
                Vs = (0.75*v[i-1,j+1]+0.5*v[i,j+1]+0.25*v[i+1,j+1])/1.5
                Fnu[i,j] = rho_f*(Vn)*1.5*dx
                Fsu[i,j] = rho_f*(Vs)*1.5*dx
            else:
                Feu[i,j] = rho_f*(0.5*u[i,j]+u[i+1])*dy
                Fwu[i,j] = rho_f*(0.5*u[i,j]+u[i-1])*dy
                Fnu[i,j] = rho_f*(0.5*(v[i,j]+v[i+1,j]))*dx
                Fsu[i,j] = rho_f*(0.5*(v[i,j+1]+v[i+1,j+1]))*dx

            Deu[i,j], Dwu[i,j] = mu_f*dy/dx
            Dnu[i,j], Dsu[i,j] = mu_f*dx/dy

            if j==dims-1:
                Dsu[i,j] = Dsu[i,j]/2
            if j==1:
                Dnu[i,j] = Dnu[i,j]/2
            if i==1 or i == dims-1:
                Dnu[i,j] = 1.5*Dnu[i,j]
                Dsu[i,j] = 1.5*Dsu[i,j]

            Peu[i,j] = Feu[i,j]/Deu[i,j]
            Pwu[i,j] = Fwu[i,j]/Dwu[i,j]
            Pnu[i,j] = Fnu[i,j]/Dnu[i,j]
            Psu[i,j] = Fsu[i,j]/Dsu[i,j]

            aEu[i,j] = Deu[i,j]*np.max((0,(1-0.1*np.abs(Peu[i,j]))**5)) + np.max((0,(-Feu[i,j])))
            aWu[i,j] = Dwu[i,j]*np.max((0,(1-0.1*np.abs(Pwu[i,j]))**5)) + np.max((0,(Fwu[i,j])))
            aNu[i,j] = Dnu[i,j]*np.max((0,(1-0.1*np.abs(Pnu[i,j]))**5)) + np.max((0,(-Fnu[i,j])))
            aSu[i,j] = Dsu[i,j]*np.max((0,(1-0.1*np.abs(Psu[i,j]))**5)) + np.max((0,(Fsu[i,j])))
            aPu[i,j] = aEu[i,j]+aWu[i,j]+aNu[i,j]+aSu[i,j]

def usolve(dims,
           u,aEu,aWu,aSu,aNu,aPu,
           du,dy):
    for j in range(1,dims-1):
        for i in range(1,dims-1):
            u[i,j] = (aEu[i,j]*u[i+1,j]+aWu[i,j]*u[i-1,j]+aNu[i,j]*u[i,j-1]+aSu[i,j]*u[i,j+1])*(omega_u/aPu[i,j])
            du[i,j] = dy/((aPu[i,j]/omega_u)-(aEu[i,j]+aWu[i,j]+aNu[i,j]+aSu[i,j]))

def vcoeffs(dims,
            Dev,Dwv,Dnv,Dsv,
            Fev,Fwv,Fnv,Fsv,
            Pev,Pwv,Pnv,Psv,
            aEv,aWv,aNv,aSv,aPv,
            v,u,dx,dy):
    for j in range(1,dims-1):
        for i in range(1,dims-1):
            if j==1:
                Fev[i,j] = rho_f*(0.5*(v[i,j]+v[i+1]))*dy
                Fwv[i,j] = rho_f*(0.5*(v[i,j]+v[i-1]))*dy
                Un = (0.25*u[i-1,j]+0.5*u[i,j]+0.75*u[i+1,j])/1.5
                Us = (0.25*u[i-1,j+1]+0.5*u[i,j+1]+0.75*u[i+1,j+1])/1.5
                Fnv[i,j] = rho_f*(Vn)*1.5*dx
                Fsv[i,j] = rho_f*(Vs)*1.5*dx
            elif j==dims-1:
                Fev[i,j] = rho_f*(0.5*(u[i,j]+u[i+1]))*dy
                Fwv[i,j] = rho_f*(0.5*(u[i,j]+u[i-1]))*dy
                Vn = (0.75*v[i-1,j]+0.5*v[i,j]+0.25*v[i+1,j])/1.5
                Vs = (0.75*v[i-1,j+1]+0.5*v[i,j+1]+0.25*v[i+1,j+1])/1.5
                Fnv[i,j] = rho_f*(Vn)*1.5*dx
                Fsv[i,j] = rho_f*(Vs)*1.5*dx
            else:
                Fev[i,j] = rho_f*(0.5*u[i,j]+u[i+1])*dy
                Fwv[i,j] = rho_f*(0.5*u[i,j]+u[i-1])*dy
                Fnv[i,j] = rho_f*(0.5*(v[i,j]+v[i+1,j]))*dx
                Fsv[i,j] = rho_f*(0.5*(v[i,j+1]+v[i+1,j+1]))*dx

            Dev[i,j], Dwv[i,j] = mu_f*dy/dx
            Dnv[i,j], Dsv[i,j] = mu_f*dx/dy

            if i==dims-1:
                Dev[i,j] = Dev[i,j]/2
            if i==1:
                Dwv[i,j] = Dwv[i,j]/2
            if j==1 or j == dims-1:
                Dev[i,j] = 1.5*Dev[i,j]
                Dwv[i,j] = 1.5*Dwv[i,j]

            Pev[i,j] = Fev[i,j]/Dev[i,j]
            Pwv[i,j] = Fwv[i,j]/Dwv[i,j]
            Pnv[i,j] = Fnv[i,j]/Dnv[i,j]
            Psv[i,j] = Fsv[i,j]/Dsv[i,j]

            aEv[i,j] = Dev[i,j]*np.max((0,(1-0.1*np.abs(Pev[i,j]))**5)) + np.max((0,(-Fev[i,j])))
            aWv[i,j] = Dwv[i,j]*np.max((0,(1-0.1*np.abs(Pwv[i,j]))**5)) + np.max((0,(Fwv[i,j])))
            aNv[i,j] = Dnv[i,j]*np.max((0,(1-0.1*np.abs(Pnv[i,j]))**5)) + np.max((0,(-Fnv[i,j])))
            aSv[i,j] = Dsv[i,j]*np.max((0,(1-0.1*np.abs(Psv[i,j]))**5)) + np.max((0,(Fsv[i,j])))
            aPv[i,j] = aEv[i,j]+aWv[i,j]+aNv[i,j]+aSv[i,j]

def vsolve(dims,
           v,aEv,aWv,aSv,aNv,aPv,
           dv,dx):
    for j in range(1,dims-1):
        for i in range(1,dims-1):
            v[i,j] = (aEv[i,j]*v[i+1,j]+aWv[i,j]*v[i-1,j]+aNv[i,j]*v[i,j+1]+aSv[i,j]*v[i,j-1])*(omega_v/aPv[i,j])
            dv[i,j] = dx/((aPv[i,j]/omega_v)-(aEv[i,j]+aWv[i,j]+aNv[i,j]+aSv[i,j]))

def pccoeffs(dims,
             u,v,
             bPP,aEP,aWP,aNP,aSP,aPP,
             du,dv,dx,dy):
    for j in range(1,dims-1):
        for i in range(1,dims-1): 
            bPP[i,j] = rho_f*dy*(u[i+1,j]-u[i,j]) + rho_f*dx*(v[i,j-1]-v[i,j])
            aEP[i,j] = rho_f*du[i,j]*dy
            aWP[i,j] = rho_f*du[i,j]*dy
            aNP[i,j] = rho_f*dv[i,j]*dx
            aSP[i,j] = rho_f*dv[i,j]*dx
            aPP[i,j] = aEP[i,j]+aWP[i,j]+aNP[i,j]+aSP[i,j]

def pcsolve(dims,
            p,p_prm,aEP,aWP,aNP,aSP,aPP):
    for j in range(1,dims-1):
        for i in range(1,dims-1):
            p_prm[i,j] = (aEP[i,j]*p[i+1,j]+aWP[i,j]*p[i-1,j]+aNP[i,j]*p[i,j+1]+aSP[i,j]*p[i,j-1])*(omega_p/aPP[i,j])

def ucorrect(dims,
             u,u_prm,du,p_prm):
    for j in range(1,dims-1):
        for i in range(1,dims-1):
            u_prm[i,j] = du[i,j]*(p_prm[i,j]-p_prm[i-1,j])
    u = u + u_prm

def vcorrect(dims,
             v,v_prm,dv,p_prm):
    for j in range(1,dims-1):
        for i in range(1,dims-1):
            v_prm[i,j] = dv[i,j]*(p_prm[i,j]-p_prm[i-1,j])
    v = v + v_prm

def tsolve(dims,T,
           aPT,aET,aWT,aNT,aST):
    for j in range(1,dims-1):
        for i in range(1,dims-1):
            T[i,j] = (aET[i,j]*T[i+1,j]+aWT[i,j]*T[i-1,j]+aNT[i,j]*T[i,j+1]+aST[i,j]*T[i,j-1])*(omega_p/aPT[i,j])
           

def conv_check(dx,dy,u,v,t,
               aEu,aWu,aNu,aSu,aPu,
               aEv,aWv,aNv,aSv,aPv,
               aEt,aWt,aNt,aSt,aPt):
    Ru = np.abs(np.sum(np.multiply(aPu,u)-np.multiply(aEu,u)-np.multiply(aWu,u)-np.multiply(aNu,u)-np.multiply(aSu,u)))/(np.sum(np.multiply(aPu,u)))

    Rv = np.abs(np.sum(np.multiply(aPv,v)-np.multiply(aEv,v)-np.multiply(aWv,v)-np.multiply(aNv,v)-np.multiply(aSv,v)))/(np.sum(np.multiply(aPv,v)))

    Rp = (np.sum(rho_f*u - rho_f*u*dy - rho_f*u - rho_f*u*dx))/(rho_f*u_0*L)

    Rt = (np.sum(rho_f*u - rho_f*u*dy - rho_f*u - rho_f*u*dx))/(rho_f*u_0*L)

    if Ru<Ru_tol and Rv<Rv_tol and Rp<Rp_tol and Rt<Rt_tol:
        convstate = True
    else:
        convstate = False
    
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

def SIMPLE_sol(dimlist,iterlim):
    
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
        #init necessary matrices
        p = v = u = T = np.ones((nx+2, ny+2))
        p_prm = v_prm = u_prm = np.zeros((nx+2, ny+2))
        dv = du = np.ones((nx+2, ny+2))
        Deu = Dwu = Dnu = Dsu = Dev = Dwv = Dnv = Dsv = np.ones((nx+2, ny+2))
        Feu = Fwu = Fnu = Fsu = Fev = Fwv = Fnv = Fsv = np.ones((nx+2, ny+2))
        Peu = Pwu = Pnu = Psu = Pev = Pwv = Pnv = Psv = np.ones((nx+2, ny+2))
        Peu = Pwu = Pnu = Psu = Pev = Pwv = Pnv = Psv = np.ones((nx+2, ny+2))
        aEu = aWu = aNu = aSu = aPu = aEv = aWv = aNv = aSv = aPv = np.ones((nx+2, ny+2))
        aEP = aWP = aNP = aSP = aPP = bPP = np.ones((nx+2, ny+2))
        
        Rp = Ru = Rv = Rt = 1     # start residuals with values greater than tolerance to force at least one iteration
        itercount         = 1     # start count of iterations to convergence
        converged         = False # converged boolean starts on false to force at least one iteration
        while converged==False and itercount<iterlim:
            Rp, Ru, Rv, Rt, converged = conv_check()
            
    print("Simulation complete!")
    return [psols,usols,vsols,tsols]


###################################
#########  Problem #1 #############
###################################

# 1) Tabulated velocity with 5 decimal point truncation
#prob_one_sol = SIMPLE_sol()


###################################
#########  Problem #2 #############
###################################
Xs = []
Ys = []
for cvs in N_CVs_two:
    Xs.append(np.linspace(0,L,cvs[0]+2))
    Ys.append(np.linspace(0,L,cvs[1]+2))

Tws = T_w*np.ones(12)
# 2a) Plot U/Uin at channel exit and show that Umax/Uin approaches 1.5 as mesh is refined
plt.figure()
plt.plot(Xs[0], Tws)
plt.title('U/U_in at Channel Exit')
plt.xlabel('# of CVs')
plt.ylabel('Y-axis [m]')
plt.show()

# 2b) Plot non-dimensional temperature at x=2.0 as a function of y
plt.figure()
plt.plot(Xs[0], Tws)
plt.title('Non-dimensional Temp vs Y')
plt.xlabel('Y-axis [m]')
plt.ylabel('non-dimensional temperature')
plt.show()

# 2c) Plot T_bulk and Tw as a function of x
plt.figure()
plt.plot(Xs[0], Tws)
plt.title("Plot Tbulk and Tw as a function of x")
plt.xlabel('X-axis [m]')
plt.ylabel('T_bulk and T_w')
plt.show()

# 2d) Plot the local Nusselt number as a function of x
plt.figure()
plt.plot(Xs[0], Tws)
plt.title("Plot of Local Nusselt and Tw as a function of x")
plt.xlabel('X-axis [m]')
plt.ylabel('T_bulk and T_w')
plt.show()
Nu_exact = 7.5407



# Extras for post-processing
# Specify the filenames
# filename_1 = 'p1tabs.txt'
# filename_2 = 'p2tabs.txt'

# # Open the file in write mode
# with open(filename_1, 'w') as file:
#     # Write each item from the list to the file, each on a new line
#     for item in prob_one_sol:
#         file.write(f"{item}\n")

# print(f"List has been written to {filename_1}")

# with open(filename_2, 'w') as file:
#     # Write each item from the list to the file, each on a new line
#     for item in prob_two_sol:
#         file.write(f"{(item*1e5)/1e5}\n") #multiplying and dividing the decimal values by 1e5 truncates to 5 decimal places

# print(f"List has been written to {filename_2}")