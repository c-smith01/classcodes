'''
Created by Coleman Smith on 1/23/24
NUEN/MEEN 644 HW1
Due 15 February 2024
'''
import numpy as np
import matplotlib.pyplot as plt

# Define constants
L     = 1       #m
Ru_tol = 1E-6
Rv_tol = Ru_tol
R_p_tol = 1E-5
omega = 0.5     # Reccomended relaxation factor
Re = 100        # Unitless Reynolds #
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

# Initialize matrix of zeroes to represent nodes
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

Ts_2 = np.zeros(Ts_dim)

Ts_2[:,0] = 50
Ts_2[ITMAX-1,:] = 100
Ts_2[0,:] = 100
print(Ts_2)



###################################
#########  Problem #3 #############
###################################



###################################
#########  Problem #4 #############
###################################

N_CVs = [5,8,16,64, 128, 256]



###################################
#########  Problem #5 #############
###################################

stepsizes = []
centertemps = []

p = []
GCI = []