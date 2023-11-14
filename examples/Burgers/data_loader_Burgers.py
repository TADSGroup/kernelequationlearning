import numpy as np
from scipy.io import loadmat
import pysindy as ps
from models import *


data = loadmat('/home/juanfelipe/Desktop/research/keql/examples/Burgers/data/burgers.mat')

np.random.seed(9)

# Number of training points
N = 1000
N_train = 20
N_test = N

# Number of functions
m = 1

# t
t = data['t']
t = np.ravel(t)
dt = t[1] - t[0] # dt

# x 
x = data['x'].T
x = np.ravel(x)
dx = x[1] - x[0] # dx

# u
u = np.real(data['usol']).reshape(-1,1)
u = u.reshape(256, 101)

# u_t, u_x, u_xx
u_t = ps.FiniteDifference(axis=1)._differentiate(u, t=dt)
u_x = ps.FiniteDifference(axis=0, order=4)._differentiate(u, t=dx)
u_xx = ps.FiniteDifference(axis=0, order=4, d=2)._differentiate(u, t=dx)


# (t,x)-meshgrid
T, X = np.meshgrid(t,x)
# N x 2 of all collocation points 
all_pairs = np.vstack([T.ravel(), X.ravel()]) 
# N values of u at all collocation points 
u_all_flat = u.flatten() 
# N values of u_t at all collocation points
u_t_all_flat = u_t.flatten()
# N values of u_x at all collocation points
u_x_all_flat = u_x.flatten()
# N values of u_xx at all collocation points
u_xx_all_flat = u_xx.flatten()

# Stack all triples
u_all_ = np.vstack([all_pairs,u_all_flat]).T # N x 3 of collocation pts and u values 
u_t_all_ = np.vstack([all_pairs,u_t_all_flat]).T # N x 3 of collocation pts and u_t values
u_x_all_ = np.vstack([all_pairs,u_x_all_flat]).T # N x 3 of collocation pts and u_x values
u_xx_all_ = np.vstack([all_pairs,u_xx_all_flat]).T # N x 3 of collocation pts and u_xx values


# Get random indices
idx_test = np.random.randint(len(u_all_), size = N_test)
idx_train = np.random.choice(idx_test, size = N_train, replace = False)


## Get training and testing points from triples

# u
u_train_ = u_all_[idx_train,:]
u_test_ = u_all_[idx_test,:]
t_train, x_train, u_train = u_train_[:,0], u_train_[:,1], u_train_[:,2]
t_test, x_test, u_test = u_test_[:,0], u_test_[:,1], u_test_[:,2]
# u_t
u_t_train_ = u_t_all_[idx_train,:]
u_t_test_ = u_t_all_[idx_test,:]
t_train, x_train, u_t_train = u_t_train_[:,0], u_t_train_[:,1], u_t_train_[:,2]
t_test, x_test, u_t_test = u_t_test_[:,0], u_t_test_[:,1], u_t_test_[:,2]
# u_x
u_x_train_ = u_x_all_[idx_train,:]
u_x_test_ = u_x_all_[idx_test,:]
t_train, x_train, u_x_train = u_x_train_[:,0], u_x_train_[:,1], u_x_train_[:,2]
t_test, x_test, u_x_test = u_x_test_[:,0], u_x_test_[:,1], u_x_test_[:,2]
# u_xx
u_xx_train_ = u_xx_all_[idx_train,:]
u_xx_test_ = u_xx_all_[idx_test,:]
t_train, x_train, u_xx_train = u_xx_train_[:,0], u_xx_train_[:,1], u_xx_train_[:,2]
t_test, x_test, u_xx_test = u_xx_test_[:,0], u_xx_test_[:,1], u_xx_test_[:,2]


X_train = np.vstack([t_train, x_train]).T
X_test = np.vstack([t_test, x_test]).T

# # Create a boolean array of size N initialized with False
M = np.zeros((m, N_test), dtype=bool)
# # Set the values at specified indices to True
# idx = []
# for i in range(m):
#     index = np.where(np.in1d(X_test,X_train[i]))[0]
#     idx.append(index)
#     M[i][index] = True