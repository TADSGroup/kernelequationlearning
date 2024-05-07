import jax 
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from Kernels import *
from KernelTools import *


def GP_sampler(num_samples, X, kernel, len_scale, reg, seed, **kwargs):   
    """
        Gets samples(functions) of GP. 

        Args:
            num_samples (int): Number of samples.
            X (jnp.array): Domain to get the sample.
            kernel (str): Name of kernel.
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
            >>> u1(pairs)
            Array([-0.81486408, -0.80513906, -0.79336064, ...,  1.33616814,
                    1.35864323,  1.38001376], dtype=float64)
    """
    N = len(X)
    # Choose kernel
    if kernel == 'rbf':
      k = get_gaussianRBF(len_scale)
    elif kernel == 'matern':
      k = get_gaussianRBF(len_scale)
    k = vectorize_kfunc(k)
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
      def interp(x):
        return jnp.dot(k(x,X),coeffs)
      return interp
    
    interps = []
    for i in range(num_samples):
      interps.append(get_interpolant(X,sample[:,i]))
    
    return interps






