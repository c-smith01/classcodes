'''
Created by Coleman Smith on 1/23/24
NUEN/MEEN 644 HW1
Due 15 February 2024
'''
import numpy as np
import matplotlib.pyplot as plt

# Define constants
L     = 0.20 # cm -> m
k     = 386   # W/m*K
beta  = 100  # W/m^2*C
T_0   = 100  # deg C
T_inf = 30   # deg C
q_in  = 10E3 # W/m
omega = 1.1
R_t   = 1E-5

###################################
#########  Problem #1 #############
###################################

# Set matrix containing nodes for 5x5 CVs
ITCV = 5
JTCV = ITCV
ITMAX = ITCV + 2
JTMAX = ITMAX
Ts_dim = (ITMAX, ITMAX)
a = k/L

# Initialize matrix of zeroes to represent nodes
Ts = np.zeros(Ts_dim)

# Set boundary conditions for problem
Ts[:,0] = 50
Ts[ITMAX-1,:] = 50
Ts[0,:] = 100
#print(Ts)

k = 0
conv_tol = 1
while k < 2000 and conv_tol > R_t:
    for j in range(1,JTCV):
        for i in range(1,ITCV):
            Ts[i,j] = (omega/(4*a))*(a*Ts[i-1,j] + a*Ts[i+1,j] + a*Ts[i,j+1] + a*Ts[i,j-1])
    k+=1

print(Ts)


###################################
#########  Problem #2 #############
###################################

New_South = 100 # deg C

Ts_2 = np.zeros(Ts_dim)

Ts_2[:,0] = 50
Ts_2[ITMAX-1,:] = 100
Ts_2[0,:] = 100
#print(Ts_2)



###################################
#########  Problem #3 #############
###################################



###################################
#########  Problem #4 #############
###################################

N_CVs = [5,25,35,49]



###################################
#########  Problem #5 #############
###################################

stepsizes = []
centertemps = []

p = []
GCI = []