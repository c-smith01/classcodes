'''
GPT Example - DO NOT SUBMIT
'''

import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx = Ly = 1.0  # dimensions of the cavity
rho = 1.0  # density of water
mu = 1.0  # viscosity of water
U = 1.0  # velocity at the inlet
Re = 100  # Reynolds number
nu = mu / rho  # kinematic viscosity
dx_values = [1/5, 1/8, 1/16, 1/32, 1/64, 1/128]  # grid sizes

# Function to calculate convective and diffusive fluxes
def calculate_fluxes(phi, u, v, dx, dy):
    convective_flux_x = -np.multiply(u[:, :-1], (phi[:, 1:] - phi[:, :-1]) / dx)
    convective_flux_y = -np.multiply(v[:-1, :], (phi[1:, :] - phi[:-1, :]) / dy)
    diffusive_flux_x = nu * ((phi[:, 1:] - 2 * phi[:, :-1]) / dx ** 2)
    diffusive_flux_y = nu * ((phi[1:, :] - 2 * phi[:-1, :]) / dy ** 2)
    return convective_flux_x, convective_flux_y, diffusive_flux_x, diffusive_flux_y

# Function to calculate residuals
def calculate_residuals(u, v, p, dx, dy):
    du = np.sum(np.abs(u[:, 1:] - u[:, :-1]) / dx) + np.sum(np.abs(v[1:, :] - v[:-1, :]) / dy)
    dv = np.sum(np.abs(u[:, 1:] - u[:, :-1]) / dx) + np.sum(np.abs(v[1:, :] - v[:-1, :]) / dy)
    dp = np.sum(np.abs((p[:, 1:] - 2 * p[:, :-1]) / dx ** 2 + (p[1:, :] - 2 * p[:-1, :]) / dy ** 2))
    return du, dv, dp

# Function to solve for pressure and velocity fields using SIMPLE algorithm
def solve_simple(u, v, p, dx, dy, max_iter=1000, epsilon=1e-6):
    for iteration in range(max_iter):
        # Calculate convective and diffusive fluxes
        convective_flux_x, convective_flux_y, diffusive_flux_x, diffusive_flux_y = calculate_fluxes(p, u, v, dx, dy)

        # Calculate corrected velocities
        u_star = u[:, :-1] + dt * ((-convective_flux_x + diffusive_flux_x) / dx)
        v_star = v[:-1, :] + dt * ((-convective_flux_y + diffusive_flux_y) / dy)

        # Calculate divergence of corrected velocities
        div_u = ((u_star[:, 1:] - u_star[:, :-1]) / dx + (v_star[1:, :] - v_star[:-1, :]) / dy)

        # Solve for pressure using Poisson equation
        p[:, 1:-1] = p[:, 1:-1] + alpha * div_u

        # Correct velocities
        u[:, 1:-1] = u_star - (dt / dx) * (p[:, 1:] - p[:, :-1])
        v[1:-1, :] = v_star - (dt / dy) * (p[1:, :] - p[:-1, :])

        # Boundary conditions
        u[:, 0] = 0  # left wall
        u[:, -1] = 0  # right wall
        u[0, :] = U  # inlet
        u[-1, :] = u[-2, :]  # outlet
        v[0, :] = 0  # inlet
        v[-1, :] = 0  # outlet
        v[:, 0] = 0  # bottom wall
        v[:, -1] = 0  # top wall

        # Calculate residuals
        du, dv, dp = calculate_residuals(u, v, p, dx, dy)

        # Check for convergence
        if du < epsilon and dv < epsilon and dp < epsilon:
            print("Converged after {} iterations.".format(iteration+1))
            break
    else:
        print("Did not converge after {} iterations.".format(max_iter))

    return u, v, p

# Initialize variables
for dx_value in dx_values:
    dy = dx = dx_value
    dt = 0.01 * dx ** 2 / nu
    nx = int(Lx / dx) + 1
    ny = int(Ly / dy) + 1

    # Staggered grid
    u = np.zeros((ny, nx + 1))  # u-velocity
    v = np.zeros((ny + 1, nx))  # v-velocity
    p = np.zeros((ny, nx))  # pressure

    # Set inlet velocity
    u[:, 0] = U

    # Solve using SIMPLE algorithm
    alpha = 0.1
    u, v, p = solve_simple(u, v, p, dx, dy)

    # Plotting
    X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, p, cmap='viridis')
    plt.colorbar()
    plt.title('Pressure Field (dx={:.3f})'.format(dx))
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplot(1, 2, 2)
    plt.quiver(X, Y, u, v)
    plt.title('Velocity Field (dx={:.3f})'.format(dx))
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.tight_layout()
    plt.show()

