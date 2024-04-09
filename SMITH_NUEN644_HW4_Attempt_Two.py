'''
Created by Coleman Smith on 1/23/24
NUEN/MEEN 644 HW1
Due 05 April 2024
'''

import matplotlib.pyplot as plt
import numpy as np
import os

os.system('cls')
os.system('clear')


# Define constants
L       = 1                         # m
omega   = 0.5                       # Reccomended relaxation factor
T_H2O   = 20                        # Deg C
Ru_tol  = 1E-6                      # Tolerance for u-vel residual
Rv_tol  = Ru_tol                    # Tolerance for v-vel residual
Rp_tol = 1E-5                       # Tolerance for Pressure residual
Re      = 100                       # Unitless Reynolds #
rho_H2O = 998.3                     # kg/m^3
mu_H2O  = 1.002E-3                  # N*s/m^2
u_0     = (Re*mu_H2O)/(rho_H2O*L)   # m/s
N_CVs   = [5, 8, 16, 64, 128, 256]  # CVs

# General methods used in both problems

def bnd_conds_one(u_matr):
    u_matr[0] = u_0
    u_matr[-1]

def bnd_conds_two(u_matr):
    u_matr[0] = u_0

def reset(matr,dims):
    matr = np.zeros(dims)

def ucoeffs():
    
def usolve()

def vcoeffs():

def vsolve():
    
def pcoeffs():
    
def psolve():
    
def ucorrect():

def vcorrect():
    
def pcorrect():
    
def conv_check():
    
def print_res():


###################################
#########  Problem #1 #############
###################################