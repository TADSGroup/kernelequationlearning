import numpy as np
from scipy.io import loadmat
import pysindy as ps
from models import *


data = loadmat('/home/juanfelipe/Desktop/research/keql/examples/Burgers/data/burgers.mat')

# Number of functions
m = 1

# Scaling and get u and its gradients
# t scaler
#t_scaler = MinMaxScaler()
t = data['t']
t = np.ravel(t)
#t = np.ravel(t_scaler.fit_transform(t))
dt = t[1] - t[0]

# x scaler
#x_scaler = MinMaxScaler()
x = data['x'].T
x = np.ravel(x)
#x = np.ravel(x_scaler.fit_transform(x))
dx = x[1] - x[0]

# u scaler
#u_scaler = MinMaxScaler()
u = np.real(data['usol']).reshape(-1,1)
#u = u_scaler.fit_transform(u)
u = u.reshape(256, 101)

# Plot u and u_dot

u_t = ps.FiniteDifference(axis=1)._differentiate(u, t=dt)
u_x = ps.FiniteDifference(axis=0, order=4)._differentiate(u, t=dx)
u_xx = ps.FiniteDifference(axis=0, order=4, d=2)._differentiate(u, t=dx)


# Data loader
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
np.random.seed(9)
idx_train = np.random.randint(len(u_all_), size = 2000)
idx_test = np.random.randint(len(u_all_), size = int(1e4))


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


e, e_test = len(u_train), len(u_test)
X_train = np.vstack([t_train, x_train]).T
X_test = np.vstack([t_test, x_test]).T


###################
# np.random.seed(9)
# # Pendulum
# N = 10 # Number of collocation points
# #N = 4 # Number of collocation points
# m = 3 # Number of functions
# e = 3 # Number of non-zero elements for training
# e_test = N # Number of non-zero elements for test
# # N equidistant points in [0,1] to sample the training set
# x_train = np.linspace(start=0.0, stop=1.0, num=N)
# # N equidistant points in [0,1] to test P_pred
# x_test = x_train
# # Choose m times e points randomly in x_train
# x_train1 = np.sort(np.random.choice(x_train, size=e, replace=False))
# x_train2 = np.sort(np.random.choice(x_train, size=e, replace=False))
# x_train3 = np.sort(np.random.choice(x_train, size=e, replace=False))
# # Get indices of the e points selected 
# idx_1 = np.where(np.in1d(x_train,x_train1))[0]
# idx_2 = np.where(np.in1d(x_train,x_train2))[0]
# idx_3 = np.where(np.in1d(x_train,x_train3))[0]
# # Create a boolean array of size N initialized with False
# M1 = np.zeros(N, dtype=bool)
# M2 = np.zeros(N, dtype=bool)
# M3 = np.zeros(N, dtype=bool)
# # Set the values at specified indices to True
# M1[idx_1] = True
# M2[idx_2] = True
# M3[idx_3] = True
# # Concatenate the observed indices in one array
# M = np.array(np.concatenate((M1,M2,M3)), dtype=bool)
# x_train = np.concatenate([x_train1,x_train2,x_train3])
# x_test = np.concatenate([x_test,x_test,x_test])
# #x_test = np.linspace(start = 0.0, stop = 1.0, num = N//3)

# u_train, u_x_train, u_xx_train = ODE_solutions(x_train, e, k=2,d=3,c=4)
# u_test,  u_x_test,  u_xx_test  = ODE_solutions(x_test,e_test, k=2,d=3,c=4)

# x_train_all = np.concatenate([x_train1,x_train2,x_train3]).reshape(-1,1) # 30 * 1
# x_test_all = x_test.reshape(-1,1) # 30 * 1

# def f_Train(model):
#     f_train = np.zeros((e,3))
#     for i in range(3):
#         if model == 'pendulum':
#             f_train[:,i] = u_xx_train[:,i] + np.sin(u_train[:,i])
#     f_train = f_train.T.flatten()  # 30 * 1
#     return f_train

# def f_Test(model,e_test):
#     f_test = np.zeros((e_test,3))
#     for i in range(3):
#         if model == 'pendulum':
#             f_test[:,i] = u_xx_test[:,i] + np.sin(u_test[:,i])
#     f_test = f_test.T.flatten()  # 30 * 1
#     return f_test

# def f_true_Test(model, s_test):
#     if model == 'pendulum':
#         return s_test[:,3] + np.sin(s_test[:,1])