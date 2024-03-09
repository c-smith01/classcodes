'''
Created by Coleman Smith on 1/23/24
NUEN/MEEN 644 HW1
Due 8 March 2024
'''
import numpy as np
import matplotlib.pyplot as plt

# Define constants
L     = 1    # m
k     = 386  # W/m*K
beta  = 100  # W/m^2*C
T_0   = 100  # deg C
T_inf = 30   # deg C
J     = 10E3 # kW/m -> W/m
omega = 1.1
R_t   = 1E-5

###################################
#########  Problem #1 #############
###################################

# Set matrix containing nodes for 5x5 CVs
ITCV      = 5
JTCV      = ITCV
ITMAX     = ITCV + 2
JTMAX     = ITMAX + 2
Ts_dim    = (ITMAX, ITMAX)
capdelx   = L/ITCV
delx      = capdelx/2
capdely   = L/ITCV
dely      = capdely/2

a_N = k/dely
a_S = a_N
a_W = k/delx
a_E = a_W
a_P = a_N+a_S+a_E+a_W
b_P = -J

# Initialize matrix of zeroes to represent nodes
Ts = np.zeros(Ts_dim)

# Set boundary conditions for problem
Ts[:,0] = 50
Ts[ITMAX-1,:] = 50
Ts[0,:] = 100
#print(Ts)

v = 0
conv_tol = 1
while v < 200 and conv_tol > R_t:
    for j in range(1,JTCV+1):
        for i in range(1,ITCV+1):
            if i == ITCV:
                Ts[i,j] = Ts[i,j] + (omega/(a_P))*(a_W*Ts[i-1,j] + a_E*Ts[i+1,j] + a_N*Ts[i,j+1] + a_S*Ts[i,j-1] - a_P*Ts[i,j] + b_P)
            else:
                Ts[i,j] = Ts[i,j] + (omega/(a_P))*(a_W*Ts[i-1,j] + a_E*Ts[i+1,j] + a_N*Ts[i,j+1] + a_S*Ts[i,j-1] - a_P*Ts[i,j])
    v+=1
#print(v)
#print(Ts)


###################################
#########  Problem #2 #############
###################################

New_South = 100 # deg C

Ts_2 = np.zeros(Ts_dim)

Ts_2[:,0] = 50
Ts_2[ITMAX-1,:] = 100
Ts_2[0,:] = 100

v = 0
conv_tol = 1
while v < 200 and conv_tol > R_t:
    for j in range(1,JTCV+1):
        for i in range(1,ITCV+1):
            if i == ITCV:
                Ts_2[i,j] = Ts_2[i,j] + (omega/(a_P))*(a_W*Ts_2[i-1,j] + a_E*Ts_2[i+1,j] + a_N*Ts_2[i,j+1] + a_S*Ts_2[i,j-1] - a_P*Ts_2[i,j] + b_P)
            else:
                Ts_2[i,j] = Ts_2[i,j] + (omega/(a_P))*(a_W*Ts_2[i-1,j] + a_E*Ts_2[i+1,j] + a_N*Ts_2[i,j+1] + a_S*Ts_2[i,j-1] - a_P*Ts_2[i,j])
    v+=1
#print(v)
#print(Ts_2)

# Example temperature matrix (replace this with your actual temperature data)
# Let's assume this is a 10x10 grid of temperatures
# temperature_matrix = np.random.rand(10, 10) * 100  # Generating random temperatures between 0 and 100

# # Create a meshgrid for the x and y coordinates
# # Assuming these temperatures are measured at 1 unit apart in both x and y directions
x = np.arange(0, Ts_2.shape[1], 1)
y = np.arange(0, Ts_2.shape[0], 1)
X, Y = np.meshgrid(x, y)

# Plotting the contour map
# plt.figure(figsize=(8, 6))
# contour = plt.contourf(X, Y, Ts_2, cmap='viridis', levels=100)
# plt.colorbar(contour)  # Add a colorbar to show the temperature scale
# plt.title('Problem 2 Symmetric Temperature Contour Plot')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.show()



###################################
#########  Problem #3 #############
###################################

N_CVs = [5,25,35,49]
sol_storage = []
Ts[:,0] = 50
Ts[ITMAX-1,:] = 50
Ts[0,:] = 100
#print(Ts)

for CVdims in N_CVs:
    
    v = 0
    conv_tol = 1
    ITCV      = CVdims
    JTCV      = ITCV
    ITMAX     = ITCV + 2
    JTMAX     = ITMAX + 2
    Ts_dim    = (ITMAX, ITMAX)
    capdelx   = L/ITCV
    delx      = capdelx/2
    capdely   = L/ITCV
    dely      = capdely/2

    Ts_3 = np.zeros(Ts_dim)
    Ts_3[:,0] = 50
    Ts_3[ITMAX-1,:] = 50
    Ts_3[0,:] = 100

    a_N = k/dely
    a_S = a_N
    a_W = k/delx
    a_E = a_W
    a_P = a_N+a_S+a_E+a_W
    while v < 200 and conv_tol > R_t:
        for j in range(1,JTCV+1):
            for i in range(1,ITCV+1):
                if i == ITCV:
                    Ts_3[i,j] = Ts_3[i,j] + (omega/(a_P))*(a_W*Ts_3[i-1,j] + a_E*Ts_3[i+1,j] + a_N*Ts_3[i,j+1] + a_S*Ts_3[i,j-1] - a_P*Ts_3[i,j] + b_P)
                else:
                    Ts_3[i,j] = Ts_3[i,j] + (omega/(a_P))*(a_W*Ts_3[i-1,j] + a_E*Ts_3[i+1,j] + a_N*Ts_3[i,j+1] + a_S*Ts_3[i,j-1] - a_P*Ts_3[i,j])
        v+=1
    sol_storage.append(Ts_3)
    #print(v)
    #print(Ts)'
    x = np.arange(0, Ts_3.shape[1], 1)
    y = np.arange(0, Ts_3.shape[0], 1)
    X, Y = np.meshgrid(x, y)

    # Plotting the contour map
    # plt.figure(figsize=(8, 6))
    # contour = plt.contourf(X, Y, Ts_3, cmap='viridis', levels=100)
    # plt.colorbar(contour)  # Add a colorbar to show the temperature scale
    # plt.title(f'Problem 3 Temperature Contour Plot for a {CVdims} by {CVdims} matrix')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.show()


###################################
#########  Problem #4 #############
###################################
centertemps = []
for mtrxs in sol_storage:
    middle = int(np.sqrt(np.size(mtrxs))/2)
    #print(middle)
    centertemps.append(mtrxs[middle,middle])
    #print(centertemps)



# plt.figure(figsize=(8, 6))
# plt.plot(N_CVs, centertemps, 'r^')
# plt.title(f'Problem 4 Plot of Center Temperatures')
# plt.xlabel('Grid Size [dim x dim]')
# plt.ylabel('Center Temperature')
# plt.show()

###################################
#########  Problem #5 #############
###################################
F_S = 3 # Safety Factor

r_1 = 35/25
e_1 = (centertemps[2]-centertemps[1])/((r_1**2)-1)
GCI_1 = 1

r_2 = 49/35
e_2 = (centertemps[2]-centertemps[1])/((r_1**2)-1)
GCI_2 = 2

print(centertemps)
#p = ln

