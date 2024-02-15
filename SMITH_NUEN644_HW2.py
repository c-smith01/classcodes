'''
Created by Coleman Smith on 1/23/24
NUEN/MEEN 644 HW1
Due 15 February 2024
'''
import numpy as np
import matplotlib.pyplot as plt

# Define constants
L     = 0.20 # cm -> m
k     = 50   # W/m*C
beta  = 100  # W/m^2*C
T_0   = 100  # deg C
T_inf = 30   # deg C
q_in  = 10E3 # W/m 

NCV = 10

# Constants formed from given constants and params
capdelx = L/NCV
delx = capdelx/2

###################################
######## Problem #1 (b) ###########
###################################

x = np.linspace(0,L,12)
print(x)
T = np.zeros(NCV+2)
T_old_TDMA = np.zeros(NCV+2)
T[0] = T_0

a_W = k/delx
a_E = k/delx
a_P = a_W + a_E
S_P = q_in
b_P = S_P*delx

i=0
while i < 1000:
    for P in range(1,NCV+1):
        T[P] = (a_W*T_old_TDMA[P-1] + a_W*T_old_TDMA[P+1] + b_P) / a_P

        T[NCV+1] = (a_W*T_old_TDMA[NCV] + beta*T_inf) / (a_W + beta)
    T_old_TDMA = T
    i+=1


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

# while np.max(T_old-T) > conv_tol and iternum<iterlim:
#     for P in range(1,NCV+1):
#     T[P] = (a_W*T[P-1] + a_W*T[P+1] + b_P) / a_P
#     iternum+=1

###################################
######## Problem #2 (a) ###########
###################################
n_conv = np.zeros(9)

# Plot the number of iterations required for convergence vs. alpha
plt.plot(alpha_set, n_conv, 'r^')  # 'r^' specifies red triangles
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Temperature Profile Along X')
plt.grid(True)
plt.show()

###################################
######## Problem #2 (b) ###########
###################################
T_cent = np.zeros(9)

# Plot the centerline temperature vs. alpha
plt.plot(alpha_set, T_cent, 'r^')  # 'r^' specifies red triangles
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Temperature Profile Along X')
plt.grid(True)
plt.show()


