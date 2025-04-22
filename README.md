# One-Dimensional Diffusion Equation Solver

This project implements a numerical solver for the one-dimensional diffusion equation with a spatially varying diffusion coefficient. The solver is optimized for performance using JAX, allowing it to run efficiently on both CPUs and GPUs.

## Features

- **Flexible Diffusion Coefficient**: The diffusion coefficient (kappa) can vary with position and depends on the value of the diffused variable (theta)
- **High Performance**: Utilizes JAX for automatic GPU acceleration and JIT compilation
- **Memory Efficient**: Only stores initial and final states during simulation
- **Visualization**: Includes tools for plotting and analyzing results
- **Conservation Verification**: Tracks vertical averages to verify conservation properties

## Mathematical Formulation

The solver implements the one-dimensional diffusion equation:

∂θ/∂t = ∂/∂z (κ(θ,z) ∂θ/∂z)

where:
- θ is the diffused variable (e.g., temperature)
- z is the spatial coordinate
- t is time
- κ is the diffusion coefficient, which can depend on both θ and z

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install numpy matplotlib jax jaxlib
```

## Usage

### Basic Example

```python
from diffusion_solver import DiffusionSolver
import numpy as np

# Initialize solver
nz = 1000
dz = 0.1
kappa_above = 1.0
kappa_below = 0.1
theta_threshold = 0.0

solver = DiffusionSolver(nz, dz, kappa_above, kappa_below, theta_threshold)

# Create initial condition
initial_theta = np.linspace(-10, 10, nz)

# Run simulation
dt = 0.002  # Stable time step
num_steps = 10000
theta_history = solver.run_simulation(initial_theta, dt, num_steps)
```

### Running the Test Driver

The included test driver (`test_diffusion.py`) demonstrates the solver's capabilities:

```bash
python test_diffusion.py
```

This will:
1. Initialize a linear temperature profile
2. Run the diffusion simulation
3. Generate plots of the initial and final states
4. Print statistics about the solution

## Key Components

### `diffusion_solver.py`

Contains the core `DiffusionSolver` class with methods for:
- Time stepping the diffusion equation
- Calculating diffusion coefficients
- Computing vertical averages
- Managing simulation state

### `test_diffusion.py`

Provides a demonstration of the solver's capabilities, including:
- Setting up test cases
- Running simulations
- Visualizing results
- Computing statistics

## Performance Considerations

- The solver uses an explicit time-stepping scheme, so the time step must satisfy the stability condition:
  dt < dz² / (2 * max_kappa)
- A safety factor of 0.4 is applied to the time step calculation
- Memory usage is optimized by only storing initial and final states

## Visualization

The test driver generates two plots:
1. Initial temperature profile
2. Final temperature profile after simulation

Both plots show:
- Temperature (θ) on the x-axis
- Height (z) on the y-axis
- Vertical average as a dashed line
- Grid lines for reference

## Conservation Properties

With Neumann boundary conditions (zero flux at boundaries), the vertical average of θ should remain constant throughout the simulation. The solver includes tools to verify this conservation property. 