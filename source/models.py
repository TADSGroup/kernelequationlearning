import numpy as np
import jax.numpy as jnp
from jax import grad, jacfwd, vmap, jit
from kernels import *

##########################    PENDULUM    ##########################

def ODE_solutions(X, e, k, d, c, m = 3):
	
    N = len(X)
    
    u    = np.zeros((e,m))
    u[:,0] = (np.sin(k*np.pi*X[:e]))
    u[:,1] = (X[e:2*e]**d + 1.0)
    u[:,2] = (c*np.exp(X[2*e:3*e]))
    
    u_dot  = np.zeros((e,m))
    u_dot[:,0] = (k*np.pi*np.cos(k*np.pi*X[:e]))
    u_dot[:,1] = (d*X[e:2*e]**(d-1))
    u_dot[:,2] = (c*np.exp(X[2*e:3*e]))
    
    u_ddot = np.zeros((e,m))
    u_ddot[:,0] = (-(k**2)*(np.pi**2)*np.sin(k*np.pi*X[:e]))
    u_ddot[:,1] = (d*(d-1)*X[e:2*e]**(d-2))
    u_ddot[:,2] = (c*np.exp(X[2*e:3*e]))
    
    return u, u_dot, u_ddot

def predictions_ode(X, X_train, kernel, optim_sgm, alphas, e_train, e_test):
    m = len(optim_sgm)
    N = len(X)
    u_pred      = np.zeros((e_test,m))
    u_dot_pred  = np.zeros((e_test,m))
    u_ddot_pred = np.zeros((e_test,m))
    for i in range(m):
        u_pred[:,i]      = np.dot(K(kernel, X[e_test*i:e_test*(i+1)], X_train[e_train*i:e_train*(i+1)], optim_sgm[i]), alphas[:,i])
        u_dot_pred[:,i]  = np.dot(K_dot(kernel, X[e_test*i:e_test*(i+1)], X_train[e_train*i:e_train*(i+1)], optim_sgm[i], 0), alphas[:,i])
        u_ddot_pred[:,i] = np.dot(K_2dot(kernel, X[e_test*i:e_test*(i+1)], X_train[e_train*i:e_train*(i+1)], optim_sgm[i], 0, 0), alphas[:,i])
    
    return u_pred, u_dot_pred, u_ddot_pred

##########################     DARCY      ##########################

def predictions_darcy(X, X_train, kernel, optim_sgm, alphas):
    m = len(optim_sgm)
    N = len(X)
    u_pred    = np.zeros((N,m))
    u_x_pred  = np.zeros((N,m))
    u_y_pred  = np.zeros((N,m))
    u_xx_pred = np.zeros((N,m))
    u_yy_pred = np.zeros((N,m))
    for i in range(m):
        u_pred[:,i]    = np.dot(K_2D(kernel, X, X_train, optim_sgm[i]), alphas[:,i])
        u_x_pred[:,i]  = np.dot(K_dot2D(kernel, X, X_train, optim_sgm[i], 0), alphas[:,i])
        u_y_pred[:,i]  = np.dot(K_dot2D(kernel, X, X_train, optim_sgm[i], 1), alphas[:,i])
        u_xx_pred[:,i] = np.dot(K_2dot2D(kernel, X, X_train, optim_sgm[i], 0, 0), alphas[:,i])
        u_yy_pred[:,i] = np.dot(K_2dot2D(kernel, X, X_train, optim_sgm[i], 1, 1), alphas[:,i])
    return u_pred, u_x_pred, u_y_pred, u_xx_pred, u_yy_pred


def u1(X0,X1,k):
    return jnp.sin(k*jnp.pi*X0 + k*jnp.pi*X1)

def u2(X0,X1,d):
    return jnp.array((X0-0.5)**d + (X1-0.5)**d)

def u3(X0,X1,c):
    return c*jnp.exp(X0+ X1)

def a(X0,X1,k=1):
    t1 = jnp.exp(jnp.sin(jnp.pi*X0)+jnp.sin(jnp.pi*X1))
    t2 = jnp.exp(-jnp.sin(jnp.pi*X0)-jnp.sin(jnp.pi*X1))
    return t1 + t2

def u_dot(u, T, params, arg):
	u_Dot = jit(grad(u,arg))
	return vmap(lambda t : u_Dot(t[0], t[1], params))(T)

def u_ddot(u, T, params, arg1, arg2):
	u_2Dot = jit(grad(grad(u,arg1),arg2))
	return vmap(lambda t : u_2Dot(t[0], t[1], params))(T)

def u_dddot(u, T, params, arg1, arg2, arg3):
	u_3Dot = jit(grad(grad(grad(u,arg1),arg2),arg3))
	return vmap(lambda t : u_3Dot(t[0], t[1], params))(T)

def u_ddddot(u, T, params, arg1, arg2, arg3, arg4):
	u_4Dot = jit(grad(grad(grad(grad(u,arg1),arg2),arg3),arg4))
	return vmap(lambda t : u_4Dot(t[0], t[1], params))(T)


def darcy_solutions(X, k, d, c, m=3):
    
    N = len(X)
    
    # u
    u = np.zeros((N,m))
    u[:,0] = u1(X[:,0], X[:,1], k)
    u[:,1] = u2(X[:,0], X[:,1], d)
    u[:,2] = u3(X[:,0], X[:,1], c)
    
    # u_x
    u_x = np.zeros((N,m))
    u_x[:,0] = u_dot(u1, X, k, 0)
    u_x[:,1] = u_dot(u2, X, d, 0)
    u_x[:,2] = u_dot(u3, X, c, 0)
    
    # u_y
    u_y = np.zeros((N,m))
    u_y[:,0] = u_dot(u1, X, k, 1)
    u_y[:,1] = u_dot(u2, X, d, 1)
    u_y[:,2] = u_dot(u3, X, c, 1)

    # u_xx
    u_xx = np.zeros((N,m))
    u_xx[:,0] = u_ddot(u1, X, k, 0, 0)
    u_xx[:,1] = u_ddot(u2, X, d, 0, 0)
    u_xx[:,2] = u_ddot(u3, X, c, 0, 0)

    # u_yy
    u_yy = np.zeros((N,m))
    u_yy[:,0] = u_ddot(u1, X, k, 1, 1)
    u_yy[:,1] = u_ddot(u2, X, d, 1, 1)
    u_yy[:,2] = u_ddot(u3, X, c, 1, 1)

    # a
    a_vals = a(X[:,0], X[:,1])

    # a_x
    a_x = u_dot(a, X, 2, 0)
    
    # a_y
    a_y = u_dot(a, X, 2, 1)

    return u, u_x, u_y, u_xx, u_yy, a_vals, a_x, a_y

##########################     KS      ##########################

def predictions_KS(X, X_train, kernel, optim_rho, alphas, e_train, e_test):
    m = len(optim_rho)
    N = len(X)
    u_pred    = np.zeros((N,m))
    u_x_pred  = np.zeros((N,m))
    u_t_pred  = np.zeros((N,m))
    u_xx_pred = np.zeros((N,m))
    u_xxxx_pred = np.zeros((N,m))
    for i in range(m):
        u_pred[:,i]    = jnp.dot(K_2D(kernel, X, X_train, optim_rho[i]), alphas[:,i])
        u_t_pred[:,i]  = jnp.dot(K_dot2D(kernel, X, X_train, optim_rho[i], 0), alphas[:,i])
        u_x_pred[:,i]  = jnp.dot(K_dot2D(kernel, X, X_train, optim_rho[i], 1), alphas[:,i])
        u_xx_pred[:,i] = jnp.dot(K_2dot2D(kernel, X, X_train, optim_rho[i], 1, 1), alphas[:,i])
        u_xxxx_pred[:,i] = jnp.dot(K_4dot2D(kernel, X, X_train, optim_rho[i], 1, 1, 1, 1), alphas[:,i])
    return u_pred, u_t_pred, u_x_pred ,u_xx_pred ,u_xxxx_pred

##########################     KS      ##########################

def predictions_Burgers(X, X_train, kernel, optim_rho, alphas, e_train, e_test):
    m = len(optim_rho)
    N = len(X)
    u_pred    = np.zeros((N,m))
    u_x_pred  = np.zeros((N,m))
    u_t_pred  = np.zeros((N,m))
    u_xx_pred = np.zeros((N,m))
    for i in range(m):
        u_pred[:,i]    = jnp.dot(K_2D(kernel, X, X_train, optim_rho[i]), alphas[:,i])
        u_t_pred[:,i]  = jnp.dot(K_dot2D(kernel, X, X_train, optim_rho[i], 0), alphas[:,i])
        u_x_pred[:,i]  = jnp.dot(K_dot2D(kernel, X, X_train, optim_rho[i], 1), alphas[:,i])
        u_xx_pred[:,i] = jnp.dot(K_2dot2D(kernel, X, X_train, optim_rho[i], 1, 1), alphas[:,i])
    
    return u_pred, u_t_pred, u_x_pred, u_xx_pred 