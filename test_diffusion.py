import numpy as np
import matplotlib.pyplot as plt
from diffusion_solver import DiffusionSolver

# Set up the simulation parameters
nz = 1000  # Number of grid points
dz = 0.1   # Grid spacing
kappa_above = 1.0  # Diffusion coefficient when theta > threshold
kappa_below = 0.1  # Diffusion coefficient when theta <= threshold
theta_threshold = 0.0  # Threshold value

# Calculate stable time step
max_kappa = max(kappa_above, kappa_below)
dt = 0.4 * (dz**2) / (2 * max_kappa)  # Using 0.4 for safety factor
print(f"Using time step dt = {dt:.6f}")

# Create solver instance
solver = DiffusionSolver(nz, dz, kappa_above, kappa_below, theta_threshold)

# Create linear temperature profile from -10 to +10
z = np.linspace(0, (nz-1)*dz, nz)
initial_theta = np.linspace(-10, 10, nz)

# Run simulation
num_steps = 1000000
theta_history = solver.run_simulation(initial_theta, dt, num_steps)

# Get initial and final profiles
initial_theta = theta_history[0]
final_theta = theta_history[-1]

# Calculate vertical averages
initial_avg = solver.vertical_average(initial_theta)
final_avg = solver.vertical_average(final_theta)

# Check for NaN values
if np.any(np.isnan(final_theta)):
    print("WARNING: NaN values detected in final solution!")
    print("This might indicate numerical instability.")
    print("Try reducing the time step further or increasing the number of grid points.")

# Create plots
plt.figure(figsize=(10, 6))

# Initial profile
plt.subplot(1, 2, 1)
plt.plot(initial_theta, z, 'b-', linewidth=2)
plt.axvline(x=initial_avg, color='k', linestyle='--', label=f'Average: {initial_avg:.2f}')
plt.xlabel('θ')
plt.ylabel('z')
plt.title('Initial Temperature Profile')
plt.grid(True)
plt.legend()

# Final profile
plt.subplot(1, 2, 2)
plt.plot(final_theta, z, 'r-', linewidth=2)
plt.axvline(x=final_avg, color='k', linestyle='--', label=f'Average: {final_avg:.2f}')
plt.xlabel('θ')
plt.ylabel('z')
plt.title('Final Temperature Profile')
plt.grid(True)
plt.legend()

# Adjust layout and save plots
plt.tight_layout()
plt.savefig('initial_profile.jpg', dpi=300, bbox_inches='tight')
plt.savefig('final_profile.jpg', dpi=300, bbox_inches='tight')
plt.close()

# Print statistics
print(f"Initial temperature range: {initial_theta.min():.2f} to {initial_theta.max():.2f}")
print(f"Final temperature range: {final_theta.min():.2f} to {final_theta.max():.2f}")
print(f"Maximum change in temperature: {np.max(np.abs(final_theta - initial_theta)):.2f}")
print(f"Initial vertical average: {initial_avg:.2f}")
print(f"Final vertical average: {final_avg:.2f}")
print(f"Change in vertical average: {final_avg - initial_avg:.2f}") 
