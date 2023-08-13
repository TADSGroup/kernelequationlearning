import numpy as np
from models import *

np.random.seed(9)
# Pendulum
N = 10 # Number of collocation points
#N = 4 # Number of collocation points
m = 3 # Number of functions
e = 3 # Number of non-zero elements for training
e_test = N # Number of non-zero elements for test
# N equidistant points in [0,1] to sample the training set
x_train = np.linspace(start=0.0, stop=1.0, num=N)
# N equidistant points in [0,1] to test P_pred
x_test = x_train
# Choose m times e points randomly in x_train
x_train1 = np.sort(np.random.choice(x_train, size=e, replace=False))
x_train2 = np.sort(np.random.choice(x_train, size=e, replace=False))
x_train3 = np.sort(np.random.choice(x_train, size=e, replace=False))
# Get indices of the e points selected 
idx_1 = np.where(np.in1d(x_train,x_train1))[0]
idx_2 = np.where(np.in1d(x_train,x_train2))[0]
idx_3 = np.where(np.in1d(x_train,x_train3))[0]
# Create a boolean array of size N initialized with False
M1 = np.zeros(N, dtype=bool)
M2 = np.zeros(N, dtype=bool)
M3 = np.zeros(N, dtype=bool)
# Set the values at specified indices to True
M1[idx_1] = True
M2[idx_1] = True
M3[idx_3] = True
# Concatenate the observed indices in one array
M = np.array(np.concatenate((M1,M2,M3)), dtype=bool)
x_train = np.concatenate([x_train1,x_train2,x_train3])
x_test = np.concatenate([x_test,x_test,x_test])
#x_test = np.linspace(start = 0.0, stop = 1.0, num = N//3)

u_train, u_x_train, u_xx_train = ODE_solutions(x_train, e, k=2,d=3,c=4)
u_test,  u_x_test,  u_xx_test  = ODE_solutions(x_test,e_test, k=2,d=3,c=4)

x_train_all = np.concatenate([x_train1,x_train2,x_train3]).reshape(-1,1) # 30 * 1
x_test_all = x_test.reshape(-1,1) # 30 * 1

def f_Train(model):
    f_train = np.zeros((e,3))
    for i in range(3):
        if model == 'pendulum':
            f_train[:,i] = u_xx_train[:,i] + np.sin(u_train[:,i])
    f_train = f_train.T.flatten()  # 30 * 1
    return f_train

def f_Test(model,e_test):
    f_test = np.zeros((e_test,3))
    for i in range(3):
        if model == 'pendulum':
            f_test[:,i] = u_xx_test[:,i] + np.sin(u_test[:,i])
    f_test = f_test.T.flatten()  # 30 * 1
    return f_test

def f_true_Test(model, s_test):
    if model == 'pendulum':
        return s_test[:,3] + np.sin(s_test[:,1])