import jax.numpy as jnp
from KernelTools import diagpart

def get_gaussianRBF(gamma):
    def f(x,y):
        return jnp.exp(-jnp.sum((x-y)**2)/gamma)
    return f

def get_anisotropic_gaussianRBF(gamma,A):
    def f(x,y):
        diff = x-y
        return jnp.exp(-jnp.dot(diff,A@diff)/gamma)
    return f


def get_matern_five_half(rho):
    random_constant = 1.2345678e-100
    def k(x,y):
        d=jnp.sqrt(jnp.sum((x-y+random_constant)**2))
        return (1+jnp.sqrt(5)*d/rho+5*d**2/(3 * rho**2))*jnp.exp(-jnp.sqrt(5)*d/rho)
    return k

def linear_kernel(x,y):
    return jnp.dot(x,y)

def get_poly_kernel(deg,c=1):
    def k(x,y):
        return (jnp.dot(x,y)+c)**deg
    return k

def get_anisotropic_poly_kernel(deg,A,c=1,shift = 0.):
    def k(x,y):
        return (jnp.dot(x-shift,A@(y-shift))+c)**deg
    return k

def get_shift_scale(X,scaling = 'diagonal',eps = 1e-7):
    cov = jnp.cov(X.T)
    if scaling == 'diagonal':
        A = jnp.linalg.inv(diagpart(cov))
    elif scaling == 'full':
        A = jnp.linalg.inv(cov + eps * jnp.eye(len(cov)))
    else:
        raise NotImplementedError("Only diagonal and full scaling are available")
    shift = jnp.mean(X,axis=0)
    return shift,A


def get_centered_scaled_poly_kernel(deg,X_train,c = 1,scaling = 'diagonal',eps = 1e-7):
    shift,A = get_shift_scale(X_train,scaling,eps)
    return get_anisotropic_poly_kernel(deg,A,c,shift=shift)

def get_centered_scaled_linear_kernel(X_train,c = 1,scaling = 'diagonal',eps = 1e-7):
    shift,A = get_shift_scale(X_train,scaling,eps)
    def k(x,y):
        return jnp.dot(x-shift,A@(y-shift)) + c
    return k

def get_sum_of_kernels(kernels,coefficients):
    def k(x,y):
        return jnp.sum(jnp.array([c * ki(x,y) for c,ki in zip(coefficients,kernels)]))
    return k
