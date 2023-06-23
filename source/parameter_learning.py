import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold  
import jax.numpy as jnp
from kernels import *
import scipy.spatial.distance as dist

# Cross validation - Grid search
def grid_search_RBF(x_train,u_train, grid = False):
  '''
    x_train: N x d array with collocation points.
    u_train: N x 1 values of u at x_train.
    kernel: Kernel to learn its parameters. 
    grid: Bool. Output the value of the loss.
  '''

  k1 = 10 # size of grid for sigma
  k2 = 20 # size of grid for regularization
  n_splits = 5
  

  k = np.linspace(10**-3, 2 , num=k1)
  distances = dist.pdist(x_train) # pairwise distances
  beta = np.median(distances) # median of the pairwise distances
  # Search space for sigma
  sgm = beta*k
  
  # Search space for lambda 
  lmbd = 10**np.linspace(-12, 0, k2)

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

  if grid:
    print('The grid with the loss values is:')
    print('NegMSEs are for every pair of indices: \n {}'.format(np.round(scores_rbf,1)))

  
  ij_min_rbf = np.array( np.where( scores_rbf == np.nanmin(scores_rbf) ), dtype=int).flatten()
  optim_sgm = sgm[ij_min_rbf[0]]
  optim_lmbd = lmbd[ij_min_rbf[1]]
  
  return optim_sgm, optim_lmbd


# Cross validation - Grid search - Jax version
def grid_search_RBF_JAX(x_train,u_train, grid = False):
  '''
    x_train: N x d array with collocation points.
    u_train: N x 1 values of u at x_train.
    kernel: Kernel to learn its parameters. 
    grid: Bool. Output the value of the loss.
  '''

  k1 = 10 # size of grid for sigma
  k2 = 20 # size of grid for regularization
  n_splits = 5
  

  k = jnp.linspace(10**-3, 2 , num=k1)
  distances = jnp.abs(x_train - x_train[:, None]) # pairwise distances
  beta = jnp.median(distances) # median of the pairwise distances
  # Search space for sigma
  sgm = beta*k
  
  # Search space for lambda 
  lmbd = 10**jnp.linspace(-12, 0, k2)

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

  if grid:
    print('The grid with the loss values is:')
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


# kernel parameters
def kernel_parameters(X_train,U_train):
    '''
    X_train: N x d array with collocation points.
    U_train: N x m array with values of u at X_train.
    '''
    m = U_train.shape[1] # Number of functions
    N = len(X_train)
    optim_sgm  = np.zeros(m)
    optim_lmbd = np.zeros(m)
    alphas     = np.zeros((N,m))
    for i in range(m):
        optim_sgm[i],optim_lmbd[i] = grid_search_RBF(X_train,U_train[:,i].reshape(-1,1))
        G = K(Gaussian,X_train,X_train,optim_sgm[i]) 
        M = (G + optim_lmbd[i]*jnp.eye(N))
        alphas[:,i] = jnp.linalg.solve(M,U_train[:,i])
    return optim_sgm, alphas, optim_lmbd