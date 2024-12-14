import jax.numpy as jnp

def gradient_descent():
  return 0


# To choose step size at every step of gradient descent.
def armijo_step(s, beta, sigma, dk,grad_f_xk, xk, f):
  '''
  s : Positive parameter.
  beta: Parameter between 0 and 1.
  sigma: Parameter between 0 and 1.
  dk: Descent direction.
  grad_f_xk: grad(f)(xk).
  xk: Point at where to find best step.
  f: Function to optimize.
  '''
  m = 0 
  # Check Armijo condition
  while True:
    alpha = (beta**m) * sigma
    f_xk = f(xk)
    f_xknext = f(xk + alpha*dk)
    cond = f_xk >= f_xknext - alpha*jnp.dot(grad_f_xk,dk)
    if cond == 1:
      mk = m
      break
    else:
      m += 1

  alpha = (beta**mk) * sigma
  
  return alpha
