import jax.numpy as jnp
import numpy as np
from kernels import *

# Pendulum equation

def Kphiphi_assembly(X, k = 'Gaussian', k_param = 1.):

  # Set kernel to use
  if k == 'Gaussian':
    kernel = Gaussian

  # No. of collocation points
  N = X.shape[0]
  # Initialize K(phi,phi)
  Theta = jnp.zeros((3*N,3*N))

  # Block 1,1
  val = K(kernel, X,X,k_param)
  Theta = Theta.at[:N, :N].set(jnp.reshape(val, (N, N)))

  # Block 1,2
  val = K_dot(kernel, X, X, k_param, 1)
  Theta = Theta.at[:N, N:2*N].set(jnp.reshape(val, (N, N)))
  Theta = Theta.at[N:2*N, :N].set(jnp.transpose(np.reshape(val, (N, N)))) # Block 2,1

  # Block 1,3
  val = K_2dot(kernel, X, X, k_param, 1, 1)
  Theta = Theta.at[:N, 2*N:].set(jnp.reshape(val, (N, N)))
  Theta = Theta.at[2*N:, :N].set(jnp.transpose(np.reshape(val, (N, N)))) # Block 3,1

  # Block 2,2
  val = K_2dot(kernel, X, X, k_param, 0, 1)
  Theta = Theta.at[N:2*N, N:2*N].set(jnp.reshape(val, (N, N)))
  
  # Block 2,3
  val = K_3dot(kernel, X, X, k_param, 0, 1, 1)
  Theta = Theta.at[N:2*N, 2*N:].set(jnp.reshape(val, (N, N)))
  Theta = Theta.at[2*N:, N:2*N].set(jnp.transpose(np.reshape(val, (N, N)))) # Block 3,2
 
  # Block 3,3
  val = K_4dot(kernel, X , X, k_param, 0, 0, 1, 1)
  Theta = Theta.at[2*N:, 2*N:].set(jnp.reshape(val, (N, N)))

  return Theta


### Burgers equation

def Kphiphi_Burgers_assembly(X, k = 'Gaussian2D', k_param = 1.):

  # Set kernel to use
  if k == 'Gaussian2D':
    kernel = Gaussian2D

  # No. of collocation points
  N = X.shape[0]
  # No. of functions
  m = 1
  # Initialize K(phi,phi)
  Theta = jnp.zeros((m*N,m*N))

  # Block 1,1
  val = K_2D(kernel, X,X,k_param)
  Theta = Theta.at[:N, :N].set(jnp.reshape(val, (N, N)))

  # Block 1,2
  val = K_dot2D(kernel, X, X, k_param, 1)
  Theta = Theta.at[:N, N:2*N].set(jnp.reshape(val, (N, N)))
  Theta = Theta.at[N:2*N, :N].set(jnp.transpose(np.reshape(val, (N, N)))) # Block 2,1

  # Block 1,3
  val = K_2dot2D(kernel, X, X, k_param, 1, 1)
  Theta = Theta.at[:N, 2*N:].set(jnp.reshape(val, (N, N)))
  Theta = Theta.at[2*N:, :N].set(jnp.transpose(np.reshape(val, (N, N)))) # Block 3,1

  # Block 2,2
  val = K_2dot2D(kernel, X, X, k_param, 0, 1)
  Theta = Theta.at[N:2*N, N:2*N].set(jnp.reshape(val, (N, N)))
  
  # Block 2,3
  val = K_3dot2D(kernel, X, X, k_param, 0, 1, 1)
  Theta = Theta.at[N:2*N, 2*N:].set(jnp.reshape(val, (N, N)))
  Theta = Theta.at[2*N:, N:2*N].set(jnp.transpose(np.reshape(val, (N, N)))) # Block 3,2
 
  # Block 3,3
  val = K_4dot2D(kernel, X , X, k_param, 0, 0, 1, 1)
  Theta = Theta.at[2*N:, 2*N:].set(jnp.reshape(val, (N, N)))

  return Theta
