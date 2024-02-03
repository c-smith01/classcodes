'''
Created by Coleman Smith on 1/23/24
NUEN/MEEN 644 HW1
Due 6 February 2024
'''
import numpy as np
import matplotlib.pyplot as plt


###################################
######## Problem #1 (a) ###########
###################################

# Define constants and initial conditions
l = 1 #m
m = 1 #kg
g = 9.81 #m/s^2
theta_init = [(np.pi)/12, 0]

# Initialize array of zeros for solution
theta = np.zeros(6)

def euler_expl(stepsize,ts,thetas):
    theta_n = np.zeros(len(ts))
    dot_theta_n = np.zeros(len(ts))
    theta_n[0] = thetas[0]
    dot_theta_n[0] = thetas[1]
    for i in range(1,len(ts)):
        theta_n[i] = theta_n[i-1] + dot_theta_n[i-1]*stepsize
        dot_theta_n[i] = (theta_n[i-2]-theta_n[i-1])/stepsize - (g/l)*np.sin(theta_n[i-2])*stepsize/2
    return theta_n, dot_theta_n

def euler_full_impl(t,thetas):
    theta_n = np.zeros(len(ts))
    dot_theta_n = np.zeros(len(ts))
    theta_n[0] = thetas[0]
    dot_theta_n[0] = thetas[1]
    for i in range(1,len(ts)):
        theta_n[i] = theta_n[i-1] + dot_theta_n[i-1]*stepsize - (g/l)*np.sin(theta_n[i-1])*(stepsize*stepsize)/2
        dot_theta_n[i] = (theta_n[i-2]-theta_n[i-1])/stepsize + (g/l)*np.sin(dot_theta_n[i-2])*stepsize/2
    return theta_n, dot_theta_n

def rk_scnd(t,thetas):
    theta_n = np.zeros(len(ts))
    dot_theta_n = np.zeros(len(ts))
    theta_n[0] = thetas[0]
    dot_theta_n[0] = thetas[1]
    for i in range(1,len(ts)):
        k1 = []
        k2 = []
        theta_n[i] = theta_n[i-1] + dot_theta_n[i-1]*stepsize - (g/l)*np.sin(theta_n[i-1])*(stepsize*stepsize)/2
        dot_theta_n[i] = (theta_n[i-2]-theta_n[i-1])/stepsize + (g/l)*np.sin(dot_theta_n[i-2])*stepsize/2
    return theta_n, dot_theta_n

def rk_frth(t,thetas):
    theta_n = np.zeros(len(ts))
    dot_theta_n = np.zeros(len(ts))
    theta_n[0] = thetas[0]
    dot_theta_n[0] = thetas[1]
    for i in range(1,len(ts)):
        k1 = []
        k2 = []
        k3 = []
        k4 = []
        theta_n[i] = theta_n[i-1] + dot_theta_n[i-1]*stepsize - (g/l)*np.sin(theta_n[i-1])*(stepsize*stepsize)/2
        dot_theta_n[i] = (theta_n[i-2]-theta_n[i-1])/stepsize + (g/l)*np.sin(dot_theta_n[i-2])*stepsize/2
    return theta_n, dot_theta_n

# Additional exact solution method for sanity check purposes
def exact(stepsize,ts,thetas):
    theta_n = np.zeros(len(ts))
    dot_theta_n = np.zeros(len(ts))
    theta_n[0] = thetas[0]
    dot_theta_n[0] = thetas[1]
    for i in range(1,len(ts)):
        theta_n[i] = (theta_n[0])*np.cos(np.sqrt((g/l))*i)
        dot_theta_n[i] = -(theta_n[0])*np.sqrt(g/l)*np.sin(np.sqrt(g/l)*i)
    return theta_n, dot_theta_n


###################################
######## Problem #1 (b) ###########
###################################

# Define step size
h = 0.05 #s, should be 0.05s

# Define timespan of interest
tspan = np.arange(0,2 + h,h)
#print(tspan)

# Call ODE Solver Methods
euler_esol = euler_expl(h,tspan,theta_init)
print(euler_esol)

exact_sol = exact(h,tspan,theta_init)


###################################
######## Problem #1 (c) ###########
###################################


# Create scatter plot of angular displacement vs. time
plt.plot(tspan, euler_esol[0], color='blue', label='Explicit Euler')
plt.plot(tspan, exact_sol[0], color='red', label='Exact')

# Add labels and title
plt.xlabel('Time [s]')
plt.ylabel('Angular displacment [rad]')
plt.title('Angular Displacement as a Function of Time for Various Numerical Methods')

# Display legend
plt.legend()

# Display the plot
plt.show()

###################################
######## Problem #1 (d) ###########
###################################

# Create scatter plot
plt.plot(tspan, euler_esol[1], color='blue', label='Explicit Euler')
plt.plot(tspan, exact_sol[1], color='red', label='Exact')
# Add labels and title
plt.xlabel('Time [s]')
plt.ylabel('Angular velocity [rad/s]')
plt.title('Angular Velocity as a Function of Time for Various Numerical Methods')

# Display legend
plt.legend()

# Display the plot
plt.show()
