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
q_in  = 1E3  # W/m 

NCV = 10

# Constants formed from given constants and params
capdelx = (L/NCV)
delx = (capdelx/2)

###################################
######## Problem #1 (b) ###########
###################################

x = np.linspace(0,L,12)
T = np.zeros(NCV+2)
T[0] = T_0
TDMA_dim = (NCV+1, NCV+1)
T_TDMA = np.zeros(TDMA_dim)
bp_TDMA_dim = (NCV+1,1)
bp_TDMA = np.zeros(bp_TDMA_dim)

a_W = k/delx
a_E = k/delx
a_P = a_W + a_E
S_P = q_in
b_P = (S_P*delx)

i=0
for j in range(0,11):
        bp_TDMA[j] = b_P
        T_TDMA[i,j] = a_P
        if i!=10:
            T_TDMA[i+1,j] = -a_E
        
        if i!=0:
            T_TDMA[i-1,j] = -a_W
        i+=1

R_th = ((beta*k/delx)/(beta+(k/delx)))*T_inf

T_TDMA[0,1] = -k/capdelx
T_TDMA[-1,-2] =  -k/delx #-k/capdelx
T_TDMA[-1,-1] = (k/delx) + R_th

#print(T_TDMA)

bp_TDMA[0] = b_P + (a_W*T_0)
bp_TDMA[10] = b_P + R_th
#print(bp_TDMA)

T_TDMA_Sol = np.linalg.solve(T_TDMA, bp_TDMA)
#print(T_TDMA_Sol)

for k in range(1,NCV+2):
    T[k] = T_TDMA_Sol[k-1,0]

#print(T)
    
        
# # Plot the data with red triangles
# plt.plot(x, T, 'r^')  # 'r^' specifies red triangles
# plt.xlabel('X')
# plt.ylabel('T(X)')
# plt.title('Temperature Profile Along X')
# plt.grid(True)
# plt.show()


###################################
######## Problem #2    ############
###################################

alpha_set = np.linspace(1, 1.8, 9)
#print(alpha_set)
conv_tol = 1E-5
T_old = np.zeros(NCV+2)
iterlim = 100
iternum = 1
R_t = 1
T_GS_old = np.ones(NCV+1)
T_GS_new = np.ones(NCV+1)
T_GS_old[0] = 100
T_GS_new[0] = T_GS_old[0]
n_conv = []
conv_Ts = []

i=0
for alpha in alpha_set:
    while i < 100 and R_t > conv_tol:
        for P in range(1,NCV+1):
            if P == 1:
                T_GS_new[P] = ((alpha/a_P) * (b_P + a_E*T_GS_old[P-1] + a_W*T_GS_old[P+1]))
            elif P != NCV and P != 1:
                T_GS_new[P] = ((alpha/a_P) * (b_P + a_E*T_GS_old[P+1] + a_W*T_GS_new[P-1]))
            elif P == NCV:
                bp_end = R_th + b_P
                a_P_end = (k/delx) + R_th
                
                T_GS_new[P] = ((alpha/a_P_end) * (b_P + a_W*T_GS_new[P-1]))
        R_t = np.max((T_GS_new-T_GS_old)/T_GS_old)
        T_GS_old = T_GS_new
        print(R_t)
        i+=1
    conviter = i
    n_conv.append(conviter)
    i=0
    R_t = 1
    print(f'converged at {conviter} iterations yielding T profile for alpha={alpha}',f': {T_GS_old}')
    conv_Ts.append(T_GS_new)
    T_GS_old = np.ones(NCV+1)
    T_GS_new = np.ones(NCV+1)
    T_GS_old[0] = 100
    T_GS_new[0] = T_GS_old[0]
    

###################################
######## Problem #2 (a) ###########
###################################

# Plot the number of iterations required for convergence vs. alpha
plt.plot(alpha_set, n_conv, 'r^')  # 'r^' specifies red triangles
plt.xlabel('alpha')
plt.ylabel('iterations to convergence')
plt.title('iterations to convergence as a function of alpha')
plt.grid(True)
plt.show()

###################################
######## Problem #2 (b) ###########
###################################
T_cent = []

for i in range(0,9):
    centertemp = (conv_Ts[i][6]+conv_Ts[i][7])/2
    T_cent.append(centertemp)


# Plot the centerline temperature vs. alpha
plt.plot(alpha_set, T_cent, 'r^')  # 'r^' specifies red triangles
plt.xlabel('alpha')
plt.ylabel('Y-axis')
plt.title('Centerline Temperature Profile Along X')
plt.grid(True)
plt.show()