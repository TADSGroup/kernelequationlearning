import jax.numpy as jnp
from KernelTools import diagpart, vectorize_kfunc
from scipy.special import factorial
from jaxopt import GradientDescent,LBFGS
from functools import partial
from jax import jit,value_and_grad

# Kernels

def get_gaussianRBF(gamma):
    """
        Builds an RBF kernel function.

        Args:
            gamma (double): Length scale of the RBF kernel.

        Returns:
            function: This function returns the RBF kernel with fixed parameter gamma.   


        Example:
            >>> gamma1 = 1.
            >>> k = get_gaussianRBF(gamma1)
            >>> k(1.,2.)
            Array(0.60653066, dtype=float64)
            >>> gamma2 = 2.
            >>> k = get_gaussianRBF(gamma2)
            >>> k_vectorized = vectorize_kfunc(k)
            >>> x = jnp.array([1.,2.,3.])
            >>> y = jnp.array([4.,5.])
            >>> k(x,y)
            Array([[0.32465247, 0.13533528],
                   [0.60653066, 0.32465247],
                   [0.8824969 , 0.60653066]], dtype=float64)           
    """
    def k(x,y):
        return jnp.exp(-jnp.sum((x-y)**2)/(2*gamma**2))
    return k

def get_anisotropic_gaussianRBF(gamma,A):
    def k(x,y):
        diff = x-y
        return jnp.exp(-jnp.dot(diff,A@diff)/(2*gamma**2))
    return k

def get_matern_5_2(rho):
    def k(x,y):
        # True
        d2 = jnp.sum((x-y)**2)/rho**2
        d = jnp.sqrt(d2+1e-100)
        coeff = (1 + (jnp.sqrt(5)*d) + (5*d**2/3))
        true = coeff*jnp.exp(-jnp.sqrt(5)*d)
        # Taylor
        nu = 5/2
        taylor = 1 + (nu/(2*(1-nu))*d2 + nu**2/(8*(2-3*nu+nu**2))*(d2)**2) 
        return jnp.where(d2<1e-8,taylor,true)
    return k

def get_matern_11_2(rho):
    def k(x,y):
        # True 
        d2 = jnp.sum((x-y)**2)/rho**2
        d = jnp.sqrt(d2+1e-100)
        coeff = (1 + jnp.sqrt(11)*d + (4620/945)*d**2 + (1155*jnp.sqrt(11)/945)*d**3 
                 + (1815/945)*d**4 + (121*jnp.sqrt(11)/945) *d**5)
        true = coeff*jnp.exp(-jnp.sqrt(11)*d) 
        #  Taylor
        nu = 11/2
        taylor = 1 + (nu/(2*(1-nu))*d2 + nu**2/(8*(2-3*nu+nu**2))*(d2)**2)
        return jnp.where(d2<1e-8, taylor, true)
    return k

def get_matern(p,rho):
    """
        Builds a Matern kernel function.

        Args:
            p (int): As used in smoothness nu = p + 1/2.
            rho (double): Length scale of the Matern kernel.

        Returns:
            function: This function returns the Matern kernel with smoothness p+1/2 and lengthscale rho.   


        Example:
            >>> rho1 = 1.
            >>> k_matern_5_2 = get_matern(2,rho1)
            >>> k_matern_5_2(1.,2.)
            Array(0.52399411, dtype=float64)
            >>> rho2 = 2.
            >>> k_matern_5_2 = get_matern(2,rho2)
            >>> k_matern_5_2_vectorized = vectorize_kfunc(k_matern_5_2)
            >>> x = jnp.array([1.,2.,3.])
            >>> y = jnp.array([4.,5.])
            >>> k_matern_5_2_vectorized(x,y)
            Array([[0.28316327, 0.13866022],
                   [0.52399411, 0.28316327],
                   [0.82864914, 0.52399411]], dtype=float64)           
    """
    exp_multiplier = -jnp.sqrt(2 * p + 1)
    coefficients = jnp.array([factorial(p + i) / (factorial(i) * factorial(p - i)) * (jnp.sqrt(8 * p + 4))**(p - i) for i in range(p + 1)])
    powers = jnp.arange(p,-1,-1)
    norm_cons = factorial(p)/factorial(2*p)
    def k(x,y):
        # True 
        d2 = jnp.sum((x-y)**2)/rho**2
        d = jnp.sqrt(d2+1e-100)
        true =  norm_cons*jnp.sum(coefficients * jnp.power(d,powers))*jnp.exp(exp_multiplier * d)
        #  Taylor
        nu = p + 1/2
        taylor = 1 + (nu/(2*(1-nu))*d2 + nu**2/(8*(2-3*nu+nu**2))*(d2)**2)
        return jnp.where(d2<1e-12, taylor, true)
    return k

def get_matern_13_2(rho):
    def k(x,y):
        # True 
        d2 = jnp.sum((x-y)**2)/rho**2
        d = jnp.sqrt(d2+1e-100)
        coeff = (1 + jnp.sqrt(13)*d + (65/11)*d**2 + (52*jnp.sqrt(13)/33)*d**3 
                 + (338/99)*d**4 + (169*jnp.sqrt(13)/495)*d**5 + (2197/10395)*d**6)
        true = coeff*jnp.exp(-jnp.sqrt(13)*d) 
        #  Taylor
        nu = 13/2
        taylor = 1 + (nu/(2*(1-nu))*d2 + nu**2/(8*(2-3*nu+nu**2))*(d2)**2)
        return jnp.where(d2<1e-8, taylor, true)
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


# Kernel transformations

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

def log1pexp(x):
    return jnp.log(jnp.exp(-x)+1) + x

def inv_log1pexp(y):
    return jnp.log(jnp.exp(y)-1)

def fit_kernel_params(parametrized_kernel,X,y,init_params,nugget = 1e-7):
    
    @jit
    @value_and_grad
    def marginal_like(params):
        vmapped_kfunc = vectorize_kfunc(partial(parametrized_kernel,params = params))
        K = vmapped_kfunc(X,X)
        K = K + nugget * diagpart(K)
        return (1/2) * y.T@jnp.linalg.inv(K)@y + (1/2) * jnp.linalg.slogdet(K).logabsdet
    solver = GradientDescent(
        marginal_like,value_and_grad=True,
        jit = True,tol = 1e-6)
    result = solver.run(init_params)
    optimized_params = result.params
    final_val,final_grad = marginal_like(optimized_params)

    return optimized_params,final_val#,partial(parametrized_kernel,params = optimized_params)


import sympy as sym
from sympy import factorial
from sympy.series.series import series

def setup_matern(p,eps = 1e-8):
    exp_multiplier = -sym.sqrt(2 * p + 1)
    coefficients = [
        (factorial(p)/factorial(2*p)) * (factorial(p + i) / (factorial(i) * factorial(p - i)))
        * (sym.sqrt(8 * p + 4))**(p - i) 
        for i in range(p + 1)]
    powers = list(range(p,-1,-1))

    jax_coefficients = jnp.array(list(map(float,coefficients)))
    jax_powers = jnp.array(powers)
    jax_exp_multiplier = float(exp_multiplier)

    d = sym.symbols('d')
    matern = sum([c * (d**power) for c,power in zip(coefficients,powers)])*sym.exp(exp_multiplier * d)
    #S = series(sym.log(matern),d,0,2*p+1).removeO()
    S = series(matern,d,0,2*p+1).removeO()
    polyS = sym.Poly(S,d)
    asy_coeffs = polyS.coeffs()
    asy_coeffs = jnp.array(list(map(float,asy_coeffs)))

    asy_powers = polyS.monoms()
    half_asy_powers = jnp.array(asy_powers)[:,0]//2

    def matern_p_factory(rho):
        def matern_func(x,y):
            d2 = jnp.sum((x-y)**2)/(rho**2) + 1e-16# + 1e-36
            d = jnp.sqrt(jnp.maximum(d2,1e-36))
            true = jnp.sum(jax_coefficients*jnp.power(d,jax_powers))*jnp.exp(jax_exp_multiplier * d)
            #asymptotic = jnp.exp(jnp.sum(asy_coeffs * jnp.power(d2,half_asy_powers)))
            asymptotic = jnp.sum(asy_coeffs * jnp.power(d2,half_asy_powers))
            return jnp.where(d2<eps, asymptotic, true)
        return matern_func
    return matern_p_factory


def get_rq_kernel(rho):
    def k(x,y):
        d2 = jnp.sum((x-y)**2)/(rho**2)
        return 1/(1+d2)
    return k