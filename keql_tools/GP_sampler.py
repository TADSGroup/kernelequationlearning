import jax 
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from Kernels import *
from KernelTools import *


def GP_sampler(num_samples, X, kernel, reg, seed):   
    """
        Gets samples(functions) of GP. 

        Args:
            num_samples (int): Number of samples.
            X (jnp.array): Domain to get the sample.
            kernel (fun): Kernel function from Kernels.py.
            len_scale (float): Scale to be used in kernel.
            reg (float): Regularization to invert kernel matrix.
            seed (float): Integer to fix the simulation.

        Returns:
            list: Returns the list of functions sampled from the GP.   


        Example:
            >>> # Create fine grid
            >>> x = jnp.linspace(0,1,40)
            >>> y = x
            >>> xv, yv = jnp.meshgrid(x, y)
            >>> pairs = jnp.vstack([xv.ravel(), yv.ravel()]).T
            >>> u1, u2, u3 = GPsampler2D(num_samples = 3,
                                        X = pairs, 
                                        kernel = 'rbf',
                                        len_scale = 1., 
                                        reg = 1e-12,
                                        seed = 2025
                                    )
            >>> u1(jnp.array([0.1,0.4]))
            Array(-0.30628616, dtype=float64)
            >>> jax.vmap(u1)(pairs)
            Array([-0.81486278, -0.69548977, -0.39381869, ...,  0.79765953,
                    0.76227927,  0.51719524], dtype=float64)

    """
    N = len(X)
    k = vectorize_kfunc(kernel)
    # Build kernel matrix
    kernel_matrix = k(X,X) + reg*jnp.eye(N)
    chol_factor = jnp.linalg.cholesky(kernel_matrix)
    PRNG_key = jax.random.PRNGKey(seed)
    # Sample standard normal
    normal_samples = jax.random.normal(key = PRNG_key, shape=(N,num_samples))
    # Compute sample
    sample = jnp.dot(chol_factor,normal_samples)    

    # Build interpolants
    def get_interpolant(pairs,values):
      coeffs = jnp.linalg.solve(kernel_matrix,values)
      k_vec = jax.vmap(kernel,in_axes=(0,None))

      def interp(x):
        return jnp.dot(k_vec(X,x),coeffs)
      return interp
    
    interps = [get_interpolant(X,S) for S in sample.T]
    
    return interps 






