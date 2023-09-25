import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold  
import jax.numpy as jnp
from kernels import *
import scipy.spatial.distance as dist

# 1D

# Grid search
def grid_search_RBF(x_train,u_train, print_MSE = False):
  '''
    x_train: N x d array with collocation points.
    u_train: N x 1 values of u at x_train.
    print_MSE: Bool. Output the value of the loss.
  '''

  k1 = 10 # size of grid for sigma
  k2 = 20 # size of grid for regularization
  n_splits = 3
  

  k = np.linspace(10**-3, 2 , num=k1)
  distances = dist.pdist(x_train) # pairwise distances
  beta = np.median(distances) # median of the pairwise distances
  # Search space for sigma
  sgm = beta*k
  
  # Search space for lambda 
  lmbd = 10**np.linspace(-14, -8, k2)

  scores_rbf = np.zeros((k1, k2))
  scores_std_rbf = np.zeros((k1, k2))

  mses = []
  
  for i in range(k1):
    sigma = sgm[i]

    for j in range(k2):
      alpha = lmbd[j]

      kf = KFold(n_splits = n_splits) 
      mse = 0.

      for l, (train_index, test_index) in enumerate(kf.split(x_train)):
        #print(f"Fold {l}:")
        #print(f"  Train: index={train_index}")
        xtrain, ytrain = x_train[train_index,:], u_train[train_index]
        #print(f"  Test:  index={test_index}")
        xtest, ytest = x_train[test_index,:], u_train[test_index] 
        # Train here 
        G = K(Gaussian,xtrain,xtrain,sigma) 
        M = (G + alpha*jnp.eye(xtrain.shape[0]))
        alphas_lu = jnp.linalg.solve(M,ytrain)
         
        # Predict on test data
        k_test_train = K(Gaussian,xtest,xtrain,sigma)
        y_pred = np.dot(k_test_train, alphas_lu)

        mse += jnp.mean((y_pred - ytest)**2)
      
      scores_rbf[i,j] = mse/n_splits

  if print_MSE:
    print('NegMSEs are for every pair of indices: \n {}'.format(np.round(scores_rbf,1)))

  
  ij_min_rbf = np.array( np.where( scores_rbf == np.nanmin(scores_rbf) ), dtype=int).flatten()
  optim_sgm = sgm[ij_min_rbf[0]]
  optim_lmbd = lmbd[ij_min_rbf[1]]
  
  return optim_sgm, optim_lmbd


# Grid search - JAX version
def grid_search_RBF_JAX(x_train,u_train, print_MSE = False):
  '''
    x_train: N x d array with collocation points.
    u_train: N x 1 values of u at x_train. 
    print_MSE: Bool. Output the value of the loss.
  '''

  k1 = 10 # size of grid for sigma
  k2 = 20 # size of grid for regularization
  n_splits = 3
  

  k = jnp.linspace(10**-3, 2 , num=k1)
  distances = jnp.abs(x_train - x_train[:, None]) # pairwise distances
  beta = jnp.median(distances) # median of the pairwise distances
  # Search space for sigma
  sgm = beta*k
  
  # Search space for lambda 
  lmbd = 10**jnp.linspace(-14, -8, k2)

  scores_rbf = jnp.zeros((k1, k2))
  scores_std_rbf = np.zeros((k1, k2))

  mses = []
  
  for i in range(k1):
    sigma = sgm[i]

    for j in range(k2):
      alpha = lmbd[j]

      kf = KFold(n_splits = n_splits) 
      mse = 0.

      for l, (train_index, test_index) in enumerate(kf.split(x_train)):
        #print(f"Fold {l}:")
        #print(f"  Train: index={train_index}")
        xtrain, ytrain = x_train[train_index,:], u_train[train_index]
        #print(f"  Test:  index={test_index}")
        xtest, ytest = x_train[test_index,:], u_train[test_index] 
        # Train here 
        G = K(Gaussian,xtrain,xtrain,sigma) 
        M = (G + alpha*jnp.eye(xtrain.shape[0]))
        alphas_lu = jnp.linalg.solve(M,ytrain)
         
        # Predict on test data
        k_test_train = K(Gaussian,xtest,xtrain,sigma)
        y_pred = jnp.dot(k_test_train, alphas_lu)

        mse += jnp.mean((y_pred - ytest)**2)
      
      scores_rbf = scores_rbf.at[i,j].set(mse/n_splits)

  if print_MSE:
    print('NegMSEs are for every pair of indices: \n {}'.format(np.round(scores_rbf,1)))

  
  #ij_min_rbf = np.array( np.where( scores_rbf == jnp.nanmin(scores_rbf) ), dtype=int).flatten()
  min_index = jnp.argmin(scores_rbf)
  min_row_index = min_index // scores_rbf.shape[1]
  min_col_index = min_index % scores_rbf.shape[1]
  #optim_sgm = sgm[ij_min_rbf[0]]
  #optim_lmbd = lmbd[ij_min_rbf[1]]
  optim_sgm = sgm[min_row_index].astype(float)
  optim_lmbd = jnp.array(lmbd[min_col_index])

  return optim_sgm, optim_lmbd


# 2D

# Grid search
def grid_search_Anisotropic_Gaussian_2D(x_train,u_train, print_MSE = False):
  '''
    x_train: N x d array with collocation points.
    u_train: N x 1 values of u at x_train.
    print_MSE: Bool. Output the value of the loss.
  '''

  k1 = 10 # size of grid for scale_t
  k2 = 10 # size of grid for scale_x
  k3 = 20 # size of grid for regularization
  
  n_splits = 3
  
  # scale_t
  kk1 = np.linspace(10**-3, 2 , num=k1)
  distances_t = dist.pdist(x_train[:,0][:,np.newaxis]) # pairwise distances
  beta_t = np.median(distances_t) # median of the pairwise distances
  # Search space for sigma_t
  sgm_t = beta_t*kk1
  
  # scale_x
  kk2 = np.linspace(10**-3, 2 , num=k2)
  distances_x = dist.pdist(x_train[:,1][:,np.newaxis]) # pairwise distances
  beta_x = np.median(distances_x) # median of the pairwise distances
  # Search space for sigma_t
  sgm_x = beta_x*kk2


  # Search space for lambda 
  lmbd = 10**np.linspace(-14, -8, k3)

  # Arrays to store MSEs
  scores_rbf = np.zeros((k1, k2, k3))
  scores_std_rbf = np.zeros((k1, k2, k3))

  mses = []
  
  for i in range(k1):
    sigma_t = sgm_t[i]

    for j in range(k2):
      sigma_x = sgm_x[i]

      for k in range(k3):
        alpha = lmbd[k]

        kf = KFold(n_splits = n_splits) 
        mse = 0.

        for l, (train_index, test_index) in enumerate(kf.split(x_train)):
          #print(f"Fold {l}:")
          #print(f"  Train: index={train_index}")
          xtrain, ytrain = x_train[train_index,:], u_train[train_index]
          #print(f"  Test:  index={test_index}")
          xtest, ytest = x_train[test_index,:], u_train[test_index] 
          # Train here 
          G = K_2D(Anisotropic_Gaussian_2D, xtrain, xtrain, np.array([sigma_t, sigma_x])) 
          M = (G + alpha*jnp.eye(xtrain.shape[0]))
          alphas_lu = jnp.linalg.solve(M,ytrain)
          
          # Predict on test data
          k_test_train = K_2D(Anisotropic_Gaussian_2D, xtest, xtrain, np.array([sigma_t, sigma_x]))
          y_pred = np.dot(k_test_train, alphas_lu)

          mse += jnp.mean((y_pred - ytest)**2)
        
        scores_rbf[i,j,k] = mse/n_splits

  if print_MSE:
    print('NegMSEs are for every pair of indices: \n {}'.format(np.round(scores_rbf,1)))

  
  ijk_min_rbf = np.array( np.where( scores_rbf == np.nanmin(scores_rbf) ), dtype=int).flatten()
  optim_sgm_t = sgm_t[ijk_min_rbf[0]]
  optim_sgm_x = sgm_x[ijk_min_rbf[1]]
  optim_sgm = np.array([optim_sgm_t, optim_sgm_x])
  optim_lmbd = lmbd[ijk_min_rbf[2]]

  
  return optim_sgm, optim_lmbd



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