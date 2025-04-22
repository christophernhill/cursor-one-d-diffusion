import jax
import jax.numpy as jnp
import numpy as np

class DiffusionSolver:
    def __init__(self, nz, dz, kappa_above, kappa_below, theta_threshold):
        """
        Initialize the diffusion solver.
        
        Parameters:
        -----------
        nz : int
            Number of grid points in z direction
        dz : float
            Grid spacing in z direction
        kappa_above : float or callable
            Diffusion coefficient when theta > theta_threshold
            If callable, should take z as input and return kappa
        kappa_below : float or callable
            Diffusion coefficient when theta <= theta_threshold
            If callable, should take z as input and return kappa
        theta_threshold : float
            Threshold value for theta that determines which kappa to use
        """
        self.nz = nz
        self.dz = dz
        self.kappa_above = kappa_above
        self.kappa_below = kappa_below
        self.theta_threshold = theta_threshold
        
        # Create z grid using JAX
        self.z = jnp.linspace(0, (nz-1)*dz, nz)
        
        # JIT compile the step_forward method
        self.step_forward = jax.jit(self._step_forward)
        
    def get_kappa(self, theta, z):
        """
        Get the diffusion coefficient based on theta value and position.
        
        Parameters:
        -----------
        theta : float or array
            Value of theta at position z
        z : float or array
            Position in z
        
        Returns:
        --------
        kappa : float or array
            Diffusion coefficient
        """
        if callable(self.kappa_above):
            kappa_above_val = self.kappa_above(z)
        else:
            kappa_above_val = self.kappa_above
            
        if callable(self.kappa_below):
            kappa_below_val = self.kappa_below(z)
        else:
            kappa_below_val = self.kappa_below
            
        return jnp.where(theta > self.theta_threshold, kappa_above_val, kappa_below_val)
    
    def _step_forward(self, theta, dt):
        """
        Step the diffusion equation forward in time using explicit method.
        
        Parameters:
        -----------
        theta : array
            Current theta values on the grid
        dt : float
            Time step
        
        Returns:
        --------
        theta_new : array
            Theta values after one time step
        """
        # Calculate diffusion coefficients at each point
        kappa = self.get_kappa(theta, self.z)
        
        # Calculate second derivative using central differences
        # Note: Using Neumann boundary conditions (zero flux at boundaries)
        d2theta = jnp.zeros_like(theta)
        
        # Interior points
        d2theta = d2theta.at[1:-1].set(
            (kappa[2:] * (theta[2:] - theta[1:-1]) - 
             kappa[1:-1] * (theta[1:-1] - theta[:-2])) / (self.dz**2)
        )
        
        # Boundary points (using one-sided differences)
        d2theta = d2theta.at[0].set(
            (kappa[1] * (theta[1] - theta[0])) / (self.dz**2)
        )
        d2theta = d2theta.at[-1].set(
            (-kappa[-2] * (theta[-1] - theta[-2])) / (self.dz**2)
        )
        
        # Update theta
        theta_new = theta + dt * d2theta
        
        return theta_new
    
    def step_forward(self, theta, dt):
        """
        Wrapper for the JIT-compiled step_forward method.
        Converts numpy arrays to JAX arrays if necessary.
        """
        if isinstance(theta, np.ndarray):
            theta = jnp.array(theta)
        return self._step_forward(theta, dt)
    
    def run_simulation(self, initial_theta, dt, num_steps):
        """
        Run a simulation for multiple time steps.
        
        Parameters:
        -----------
        initial_theta : array
            Initial condition for theta
        dt : float
            Time step
        num_steps : int
            Number of time steps to run
        
        Returns:
        --------
        theta_history : array
            Array containing only the initial and final states
        """
        if isinstance(initial_theta, np.ndarray):
            theta = jnp.array(initial_theta)
        else:
            theta = initial_theta
            
        # Initialize history with just initial and final states
        theta_history = jnp.zeros((2, self.nz))
        theta_history = theta_history.at[0].set(theta)
        
        # Run the simulation
        for i in range(num_steps):
            theta = self.step_forward(theta, dt)
            
        # Store final state
        theta_history = theta_history.at[1].set(theta)
            
        return theta_history
    
    def vertical_average(self, theta):
        """
        Calculate the vertical average of theta.
        
        Parameters:
        -----------
        theta : array
            Theta values to average
        
        Returns:
        --------
        average : float
            Vertically averaged value of theta
        """
        return jnp.mean(theta) 