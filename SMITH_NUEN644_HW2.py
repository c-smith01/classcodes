'''
Created by Coleman Smith on 1/23/24
NUEN/MEEN 644 HW1
Due 15 February 2024
'''
import os
import numpy as np
import matplotlib.pyplot as plt

os.system('cls') #tabula rasa

# Define constants
L     = 0.20 # cm -> m
k     = 50   # W/m*C
beta  = 100  # W/m^2*C
T_0   = 100  # deg C
T_inf = 30   # deg C
q_in  = 1E3 # W/m 

NCV = 10

# Constants formed from given constants and params
capdelx = L/NCV
delx = capdelx/2

###################################
######## Problem #1 (b) ###########
###################################

x = np.linspace(0,L,12)
T = np.zeros(NCV+2)
T[0] = T_0
TDMA_dim = (NCV, NCV)
T_TDMA = np.zeros(TDMA_dim)
bp_TDMA_dim = (NCV,1)
bp_TDMA = np.zeros(bp_TDMA_dim)

a_W = k/delx
a_E = k/delx
a_P = a_W + a_E
S_P = q_in
b_P = S_P*delx

i=0
for j in range(0,10):
        bp_TDMA[j] = b_P
        T_TDMA[i,j] = a_P
        if i!=9:
            T_TDMA[i+1,j] = a_E
        
        if i!=0:
            T_TDMA[i-1,j] = a_W
        i+=1

print(T_TDMA)

bp_TDMA[0] = b_P + a_W*T_0
bp_TDMA[9] = a_E*T_inf
print(bp_TDMA)

T_TDMA_Sol = np.linalg.solve(T_TDMA, bp_TDMA)
print(T_TDMA_Sol)

for k in range(1,NCV+1):
    T[k] = T_TDMA_Sol[k-1,0]

print(T)
    
        
# Plot the data with red triangles
plt.plot(x, T, 'r^')  # 'r^' specifies red triangles
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Temperature Profile Along X')
plt.grid(True)
plt.show()


###################################
######## Problem #2    ############
###################################

alpha_set = np.linspace(1, 1.8, 9)
print(alpha_set)
conv_tol = 1E-5
T_old = np.zeros(NCV+2)
iterlim = 100
iternum = 1

#for alpha in alpha_set:
while np.max(T_old-T) > conv_tol and iternum<iterlim:
    for P in range(1,NCV+1):
        T[P] = (a_W*T[P-1] + a_W*T[P+1] + b_P) / a_P
        iternum+=1

###################################
######## Problem #2 (a) ###########
###################################
n_conv = np.zeros(9)

# Plot the number of iterations required for convergence vs. alpha
# plt.plot(alpha_set, n_conv, 'r^')  # 'r^' specifies red triangles
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Temperature Profile Along X')
# plt.grid(True)
# plt.show()

###################################
######## Problem #2 (b) ###########
###################################
T_cent = np.zeros(9)

# Plot the centerline temperature vs. alpha
# plt.plot(alpha_set, T_cent, 'r^')  # 'r^' specifies red triangles
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Temperature Profile Along X')
# plt.grid(True)
# plt.show()