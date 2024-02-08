import jax.numpy as jnp
from jax import grad, jacfwd, vmap, jit
import numpy as np

#	1D 

# Gaussian Kernel

def Gaussian(t,t_,params):
	sigma = params
	r2 = jnp.dot(t-t_,t-t_)
	return jnp.exp(-r2/(2*jnp.power(sigma,2)))

# Polynomial Kernel

def Polynomial(t,t_,params):
	c0,d = params
	return (jnp.dot(t,t_)+c0)**d

# Matern Kernel

def Matern_Kernel_0(t,t_,params):
	rho, sigma = params
	d = jnp.sqrt(jnp.dot(t-t_,t-t_))
	coef = sigma**2
	return coef * jnp.exp(-d/rho)

def Matern_Kernel_1(t,t_,params):
	rho, sigma = params
	d = jnp.sqrt(jnp.dot(t-t_,t-t_))
	coef = sigma**2 * (1 + (jnp.sqrt(3)*d/rho))
	return coef * jnp.exp(-jnp.sqrt(3)*d/rho)

def Matern_Kernel_52(t,t_,params):
	rho, sigma = params
	d = jnp.sqrt(jnp.dot(t-t_,t-t_))
	coef = sigma**2 * (1 + (jnp.sqrt(5)*d/rho) + (5*d**2/3*rho**2))
	return coef * jnp.exp(-jnp.sqrt(5)*d/rho)

# Kernel Matrices

def K(kernel, T, T_, params):
	return vmap(lambda t: vmap(lambda t_: kernel(t,t_, params))(T_))(T)

def K_dot(kernel, T ,T_, params, arg):
	K_Dot = grad(kernel,arg)
	return vmap(lambda t: vmap(lambda t_: K_Dot(t, t_, params))(T_))(T)

def K_2dot(kernel, T ,T_, params, arg1, arg2):
	K_2Dot = grad(grad(kernel,arg1),arg2)
	return vmap(lambda t: vmap(lambda t_: K_2Dot(t ,t_, params))(T_))(T)

def K_3dot(kernel, T ,T_, params, arg1, arg2, arg3):
	K_3Dot = grad(grad(grad(kernel,arg1),arg2),arg3)
	return vmap(lambda t: vmap(lambda t_: K_3Dot(t ,t_, params))(T_))(T)
 
def K_4dot(kernel, T ,T_, params, arg1, arg2, arg3, arg4):
	K_4Dot = grad(grad(grad(grad(kernel,arg1),arg2),arg3),arg4)
	return vmap(lambda t: vmap(lambda t_: K_4Dot(t ,t_, params))(T_))(T)

#	2D

# Gaussian Kernel

def Gaussian2D(x1,x2,y1,y2,params):
  sigma = params
  r2 = ((x1-y1)**2 + 10*(x2-y2)**2)
  return jnp.exp(-r2/(2*sigma**2))

def Anisotropic_Gaussian_2D(x1,x2,y1,y2,params):
	scale_t = params[0]
	scale_x = params[1]
	r = ((x1-y1)/scale_t)**2+((x2-y2)/scale_x)**2
	return jnp.exp(-r)

# @jit
# def Matern_Kernel_52_2D(x1,x2,y1,y2,params):
# 	rho, sigma = params
# 	d = jnp.sqrt(((x1-y1)**2 + (x2-y2)**2) + 1e-8)
# 	coef = sigma**2 * (1 + (jnp.sqrt(5)*d/rho) + (5*d**2/3*rho**2))
# 	return coef * jnp.exp(-jnp.sqrt(5)*d/rho)

@jit
def Matern_Kernel_52_2D(x1,x2,y1,y2,params):
	rho = params
	d = jnp.sqrt(((x1-y1)**2 + 10*(x2-y2)**2) + 1e-8)
	coef = (1 + (jnp.sqrt(5)*d/rho) + (5*d**2/(3*rho**2)))
	return coef * jnp.exp(-jnp.sqrt(5)*d/rho) 

@jit
def Matern_Kernel_112_2D(x1,x2,y1,y2,params):
	rho = params
	d = jnp.sqrt(((x1-y1)**2 + 10*(x2-y2)**2) + 1e-8)
	coef = (1 + (jnp.sqrt(11)*d)/(rho) + (4620*(d**2))/((945)*rho**2) + (1155*jnp.sqrt(11)*(d**3))/((945)*rho**3) + (1815*(d**4))/((945)*rho**4) + (121*jnp.sqrt(11)*(d**5))/((945)*rho**5))
	return coef * jnp.exp(-jnp.sqrt(11)*d/rho) 

@jit
def Polynomial_2D(x1,x2,y1,y2,params):
	c0, d = params
	return (x1*y1 + x2*y2 +c0)**d

@jit
def Polynomial_2D(x1,x2,y1,y2,params):
	d = params
	return (x1*y1 + x2*y2)**d

# ND

def polynomial_kernel2(x, y, degree=2, constant=1.0):
    """
    Polynomial kernel function.

    Parameters:
    - x: Input vector x.
    - y: Input vector y.
    - degree: Degree of the polynomial kernel.
    - constant: Constant term in the polynomial kernel.

    Returns:
    - Result of the polynomial kernel applied to x and y.
    """
    return (jnp.dot(x, y) + constant)**degree


# Kernel matrices
def K_2D(kernel, T,T_, params):
  return vmap(lambda t: vmap(lambda t_: kernel(t[0],t[1], t_[0],t_[1], params))(T_))(T)

def K_dot2D(kernel, T ,T_, params, arg):
  K_Dot = jit(grad(kernel,arg))
  return vmap(lambda t: vmap(lambda t_: K_Dot(t[0],t[1], t_[0],t_[1], params))(T_))(T)

def K_2dot2D(kernel, T ,T_, params, arg1, arg2):
  K_Dot = jit(grad(grad(kernel,arg1),arg2))
  return vmap(lambda t: vmap(lambda t_: K_Dot(t[0],t[1], t_[0],t_[1], params))(T_))(T)
 
def K_3dot2D(kernel, T ,T_, params, arg1, arg2, arg3):
  K_Dot = jit(grad(grad(grad(kernel,arg1),arg2),arg3))
  return vmap(lambda t: vmap(lambda t_: K_Dot(t[0],t[1], t_[0],t_[1], params))(T_))(T)
 
def K_4dot2D(kernel, T ,T_, params, arg1, arg2, arg3, arg4):
  K_Dot = jit(grad(grad(grad(grad(kernel,arg1),arg2),arg3),arg4))
  return vmap(lambda t: vmap(lambda t_: K_Dot(t[0],t[1], t_[0],t_[1], params))(T_))(T)
