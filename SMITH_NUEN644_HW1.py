'''
Created by Coleman Smith on 1/23/24
NUEN/MEEN 644 HW1
Due 6 February 2024
'''
import numpy as np
import matplotlib as plt


###################################
######## Problem #1 (a) ###########
###################################

# Define constants and initial conditions
l = 1 #m
m = 1 #kg
g = 9.81 #m/s^2
theta_0 = np.pi/12
theta_dot = 0

# Initialize array of zeros for solution
theta = np.zeros(6)

def euler_expl(h,ts,thetas):
    theta_n = [theta_0]
    for i in range(0,2/h):
        theta_n[i+1] = theta[i] - (g/l)*np.sin(theta[i])

def euler_full_impl(t,thetas):
    theta[i + 1] = theta -(g/l)*np.sin(t)

def rk_sncd(t,thetas):
    theta[i + 1] = theta -(g/l)*np.sin(t)

def rk_frth(t,thetas):
    theta[i + 1] = theta -(g/l)*np.sin(t)


###################################
######## Problem #1 (b) ###########
###################################

h = 0.05 #s
tspan = np.arange(0,2 + h,h)
print(tspan)


###################################
######## Problem #1 (c) ###########
###################################

# Create scatter plot of angular displacement vs. time
plt.scatter(x, y, color='blue', label='Random Data Points')

# Add labels and title
plt.xlabel('Time [s]')
plt.ylabel('Angular displacment [rad]')
plt.title('Scatter Plot of 5 Random Data Points')

# Display legend
plt.legend()

# Display the plot
plt.show()

###################################
######## Problem #1 (d) ###########
###################################

# Create scatter plot
plt.scatter(x, y, color='blue', label='Random Data Points')

# Add labels and title
plt.xlabel('Time [s]')
plt.ylabel('Angular velocity [rad/s]')
plt.title('Scatter Plot of 5 Random Data Points')

# Display legend
plt.legend()

# Display the plot
plt.show()
