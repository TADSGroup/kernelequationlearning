import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold  
import jax.numpy as jnp
from kernels import *
import scipy.spatial.distance as dist
from grid_search import *

# Kernel parameters - 1D - RBF
def kernel_parameters(X_train, U_train, e):
    '''
    X_train: N x d array with collocation points.
    U_train: N x m array with values of u at X_train.
    e: Number of observed values.
    '''
    m = U_train.shape[1] # Number of functions
    N = len(X_train)
    optim_sgm  = np.zeros(m)
    optim_lmbd = np.zeros(m)
    alphas     = np.zeros((e,m))
    for i in range(m):
        optim_sgm[i],optim_lmbd[i] = grid_search_RBF(X_train[e*i:e*(i+1)],U_train[:,i].reshape(-1,1))
        G = K(Gaussian,X_train[e*i:e*(i+1)], X_train[e*i:e*(i+1)], optim_sgm[i]) 
        M = (G + optim_lmbd[i]*jnp.eye(e))
        alphas[:,i] = jnp.linalg.solve(M, U_train[:,i])
    return optim_sgm, alphas, optim_lmbd



def kernel_parameters_Gaussian_2D(X_train, U_train, e):
    '''
    X_train: N_train x 2 array with collocation points.
    U_train: N_train x m array with values of u at X_train.
    e: Number of observed values.
    '''
    m = U_train.shape[1] # Number of functions
    N = len(X_train)
    optim_sgm  = np.zeros(m)
    optim_lmbd = np.zeros(m)
    alphas     = np.zeros((e,m))
    for i in range(m):
        optim_sgm[i],optim_lmbd[i] = grid_search_RBF_2D(X_train[e*i:e*(i+1)],U_train[:,i].reshape(-1,1))
        G = K_2D(Gaussian2D,X_train[e*i:e*(i+1)], X_train[e*i:e*(i+1)], optim_sgm[i]) 
        M = (G + optim_lmbd[i]*jnp.eye(e))
        alphas[:,i] = jnp.linalg.solve(M, U_train[:,i])
    return optim_sgm, alphas, optim_lmbd


# Kernel parameters - 2D - Anisotropic RBF 
def kernel_parameters_Anisotropic_RBF_2D(X_train, U_train, e):
    '''
    Parameters
    ----------
    X_train: N x d array with collocation points.
    U_train: N x m array with values of u at X_train.
    e: Number of observed values.
    Returns
    -------
    optim_sgm: 2 x m array with the scale_t and scale_x per function.
    alphas: e x m array with dual coefficients for each kernel interpolant. 
    optim_lmbd: (m, 1) array with optimal regularization per function.

    '''
    m = U_train.shape[1] # Number of functions

    optim_sgm  = np.zeros((m,2))
    optim_lmbd = np.zeros(m)
    alphas     = np.zeros((e,m))
    
    for i in range(m):
        optim_sgm[i], optim_lmbd[i] = grid_search_Anisotropic_Gaussian_2D(X_train[e*i:e*(i+1)], U_train[:,i].reshape(-1,1))
        G = K_2D(Anisotropic_Gaussian_2D, X_train[e*i:e*(i+1)],X_train[e*i:e*(i+1)],optim_sgm[i]) 
        M = (G + optim_lmbd[i]*jnp.eye(e))
        alphas[:,i] = jnp.linalg.solve(M,U_train[:,i])
    return optim_sgm, alphas, optim_lmbd

# Kernel parameters - 2D - Matern52 
def kernel_parameters_Matern52_2D(X_train, U_train, e):
    '''
    Parameters
    ----------
    X_train: N x d array with collocation points.
    U_train: N x m array with values of u at X_train.
    e: Number of observed values.
    Returns
    -------
    optim_rho: m array with the scale_t per function. The other scale is proportional to scale_t.
    alphas: e x m array with dual coefficients for each kernel interpolant. 
    optim_lmbd: (m, 1) array with optimal regularization per function.

    '''
    m = U_train.shape[1] # Number of functions

    optim_rho  = np.zeros(m)
    optim_lmbd = np.zeros(m)
    alphas     = np.zeros((e,m))
    
    for i in range(m):
        optim_rho[i], optim_lmbd[i] = grid_search_Matern52_2D(X_train[e*i:e*(i+1)], U_train[:,i].reshape(-1,1))
        G = K_2D(Matern_Kernel_52_2D, X_train[e*i:e*(i+1)],X_train[e*i:e*(i+1)],optim_rho[i]) 
        #M = (G + optim_lmbd[i]*jnp.eye(e)) # Using nugget
        M = G # No nugget
        alphas[:,i] = jnp.linalg.solve(M,U_train[:,i])
    return optim_rho, alphas, optim_lmbd

# Kernel parameters - 2D - Matern112 
def kernel_parameters_Matern112_2D(X_train, U_train, e):
    '''
    Parameters
    ----------
    X_train: N x d array with collocation points.
    U_train: N x m array with values of u at X_train.
    e: Number of observed values.
    Returns
    -------
    optim_rho: m array with the scale_t per function. The other scale is proportional to scale_t.
    alphas: e x m array with dual coefficients for each kernel interpolant. 
    optim_lmbd: (m, 1) array with optimal regularization per function.

    '''
    m = U_train.shape[1] # Number of functions

    optim_rho  = np.zeros(m)
    optim_lmbd = np.zeros(m)
    alphas     = np.zeros((e,m))
    
    for i in range(m):
        optim_rho[i], optim_lmbd[i] = grid_search_Matern112_2D(X_train[e*i:e*(i+1)], U_train[:,i].reshape(-1,1))
        #optim_rho[i], optim_lmbd[i] = 0.11, 0.0 # User input
        G = K_2D(Matern_Kernel_112_2D, X_train[e*i:e*(i+1)],X_train[e*i:e*(i+1)],optim_rho[i]) 
        #M = (G + optim_lmbd[i]*jnp.eye(e)) # Using nugget
        M = G # No nugget
        alphas[:,i] = jnp.linalg.solve(M,U_train[:,i])
    return optim_rho, alphas, optim_lmbd

# Kernel parameters - 2D - Polynomial_2D 
def kernel_parameters_Polynomial_2D(X_train, U_train, e):
    '''
    Parameters
    ----------
    X_train: N x d array with collocation points.
    U_train: N x m array with values of u at X_train.
    e: Number of observed values.
    Returns
    -------
    optim_rho: 2 x m array with the scale_t and scale_x per function.
    alphas: e x m array with dual coefficients for each kernel interpolant. 
    optim_lmbd: (m, 1) array with optimal regularization per function.

    '''
    m = U_train.shape[1] # Number of functions

    optim_d  = np.zeros(m)
    optim_lmbd = np.zeros(m)
    alphas     = np.zeros((e,m))
    
    for i in range(m):
        optim_d[i], optim_lmbd[i] = 2, 0.0 # User input
        G = K_2D(Polynomial_2D, X_train[e*i:e*(i+1)],X_train[e*i:e*(i+1)], optim_d[i]) 
        #M = (G + optim_lmbd[i]*jnp.eye(e)) # Using nugget
        M = G # No nugget
        alphas[:,i] = jnp.linalg.solve(M,U_train[:,i])
    return optim_d, alphas, optim_lmbd