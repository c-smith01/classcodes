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
Ts = np.zeros(Ts_dim)

# Set boundary conditions for problem
Ts[:,0] = 50
Ts[ITMAX-1,:] = 50
Ts[0,:] = 100
print(Ts)

while np.min(np.abs()) <= R_t:
    for j in range(1,JTCV):
        for i in range(1,ITCV):
            Ts[i,j] = (Ts[i-1,j] + Ts[i+1,j] + Ts[i,j+1] + Ts[i,j-1])/4 


###################################
#########  Problem #2 #############
###################################

New_South = 100 # deg C

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