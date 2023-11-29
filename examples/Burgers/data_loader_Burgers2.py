import numpy as np
from scipy.io import loadmat
import pysindy as ps
#from models import *


data = np.load('/home/juanfelipe/Desktop/research/keql/examples/Burgers/gen_data/sols_burgers.npy')
print(data[0,:,:].shape)
np.random.seed(9)

# Number of functions
m = 10

# Number of points to be sampled per function: 150 ghost points
N = 200
N_train = 150
N_test = N

# t
L_t = 8
dt = 0.025
N_t = int(L_t/dt)
t = np.linspace(0, L_t, N_t)
# x
L_x = 10 
dx = 0.1 
N_x = int(L_x/dx)
x = np.linspace(0,L_x,N_x)
# (t,x)-meshgrid
T, X = np.meshgrid(t,x)
# 32000 x 2  
all_pairs = np.vstack([T.ravel(), X.ravel()]) 

# At the grid points
U = []
U_t = []
U_x = []
U_xx = []
# At collocation points
U_train = []
U_t_train = []
U_x_train = []
U_xx_train = []
X_train = []
X_test = []

for i in range(m):

    # u
    u = data[i,:,:]
    U.append(u)
    # u_t, u_x, u_xx
    u_t = ps.FiniteDifference(axis=1)._differentiate(u, t=dt)
    U_t.append(u_t)
    u_x = ps.FiniteDifference(axis=0, order=4)._differentiate(u, t=dx)
    U_x.append(u_x)
    u_xx = ps.FiniteDifference(axis=0, order=4, d=2)._differentiate(u, t=dx)
    U_xx.append(u_xx)

    # Stack all triples (t,x,u), (t,x,u_t), (t,x,u_x), (t,x,u_xx). Get 32000 x 3 arrays
    u_all = np.vstack([all_pairs, u.flatten()]).T 
    u_t_all = np.vstack([all_pairs, u_t.flatten()]).T 
    u_x_all = np.vstack([all_pairs, u_x.flatten()]).T 
    u_xx_all = np.vstack([all_pairs, u_xx.flatten()]).T 


    # Get random indices
    idx_test = np.random.randint(len(u_all), size = N_test)
    idx_train = np.random.choice(idx_test, size = N_train, replace = False)


    # Get training and testing points from triples

    # u
    u_train_ = u_all[idx_train,:]
    u_test_ = u_all[idx_test,:]
    t_train, x_train, u_train = u_train_[:,0], u_train_[:,1], u_train_[:,2]
    t_test, x_test, u_test = u_test_[:,0], u_test_[:,1], u_test_[:,2]
    U_train.append(u_train)
    # u_t
    u_t_train_ = u_t_all[idx_train,:]
    u_t_test_ = u_t_all[idx_test,:]
    t_train, x_train, u_t_train = u_t_train_[:,0], u_t_train_[:,1], u_t_train_[:,2]
    t_test, x_test, u_t_test = u_t_test_[:,0], u_t_test_[:,1], u_t_test_[:,2]
    U_t_train.append(u_t_train)
    # u_x
    u_x_train_ = u_x_all[idx_train,:]
    u_x_test_ = u_x_all[idx_test,:]
    t_train, x_train, u_x_train = u_x_train_[:,0], u_x_train_[:,1], u_x_train_[:,2]
    t_test, x_test, u_x_test = u_x_test_[:,0], u_x_test_[:,1], u_x_test_[:,2]
    U_x_train.append(u_x_train)
    # u_xx
    u_xx_train_ = u_xx_all[idx_train,:]
    u_xx_test_ = u_xx_all[idx_test,:]
    t_train, x_train, u_xx_train = u_xx_train_[:,0], u_xx_train_[:,1], u_xx_train_[:,2]
    t_test, x_test, u_xx_test = u_xx_test_[:,0], u_xx_test_[:,1], u_xx_test_[:,2]
    U_xx_train.append(u_xx_train)

    # Training
    X_train_c = np.vstack([t_train, x_train]).T
    X_train.append(X_train_c)
    # Testing
    X_test_c = np.vstack([t_test, x_test]).T
    X_test.append(X_test_c)


U_train = np.vstack(U_train).reshape(-1,1) # m*N_train x 1
U_t_train = np.vstack(U_t_train).reshape(-1,1) # m*N_train x 1
U_x_train = np.vstack(U_x_train).reshape(-1,1) # m*N_train x 1
U_xx_train = np.vstack(U_xx_train).reshape(-1,1) # m*N_train x 1
X_train = np.vstack(X_train) # m*N_train x 2
X_test = np.vstack(X_test) # m*N_test x 2



# # Create a boolean array of size N initialized with False
M = np.zeros((m, N_test), dtype=bool)
# # Set the values at specified indices to True
# idx = []
# for i in range(m):
#     index = np.where(np.in1d(X_test,X_train[i]))[0]
#     idx.append(index)
#     M[i][index] = True