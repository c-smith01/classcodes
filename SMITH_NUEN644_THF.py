'''
Created by Coleman Smith on 1/23/24
NUEN/MEEN 644 HW1
Due 03 May 2024
'''

import matplotlib.pyplot as plt
import numpy as np
import os

#os.system('cls')

# Define constants
L           = 1                         # m
omega       = 0.5                       # Reccomended relaxation factor
T_H2O       = 20                        # Deg C
Ru_tol      = 1E-6                      # Tolerance for u-vel residual
Rv_tol      = Ru_tol                    # Tolerance for v-vel residual
Rp_tol      = 1E-5                      # Tolerance for Pressure residual
Re          = 200                       # Unitless Reynolds #
rho_H2O     = 998.3                     # kg/m^3
mu_H2O      = 1.002E-3                  # N*s/m^2
u_0         = (Re*mu_H2O)/(rho_H2O*L)   # m/s
N_CVs_one   = [[10,5]]
N_CVs_two   = [[20,10], [60,20], [120,40], [160,80]]     # Dimensions of CVs