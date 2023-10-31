import numpy as np
from models import *

##########################    PENDULUM    ##########################

np.random.seed(9)

N = 10
m = 3
N_train = 3 
N_test = N

x_test  = np.linspace(start=0.0, stop=1.0, num=N)
x_train1 = np.sort(np.random.choice(x_test, size=N_train, replace=False))
x_train2 = np.sort(np.random.choice(x_test, size=N_train, replace=False))
x_train3 = np.sort(np.random.choice(x_test, size=N_train, replace=False))

# Get indices of the e points selected 
idx_1 = np.where(np.in1d(x_test,x_train1))[0]
idx_2 = np.where(np.in1d(x_test,x_train2))[0]
idx_3 = np.where(np.in1d(x_test,x_train3))[0]

# Create a boolean array of size N initialized with False
M1 = np.zeros(N, dtype=bool)
M2 = np.zeros(N, dtype=bool)
M3 = np.zeros(N, dtype=bool)

# Set the values at specified indices to True
M1[idx_1] = True
M2[idx_2] = True
M3[idx_3] = True

# Concatenate the observed indices in one array
M = np.array(np.concatenate((M1,M2,M3)), dtype=bool)

x_train = np.concatenate([x_train1,x_train2,x_train3]) # 9  * 1
x_test =  np.concatenate([x_test,   x_test,   x_test]) # 30 * 1

u_train, u_x_train, u_xx_train = ODE_solutions(x_train, N_train, k=2, d=3, c=4)
u_test,  u_x_test,  u_xx_test  = ODE_solutions(x_test,  N_test,  k=2 ,d=3, c=4)

def f_Train(model):
    f_train = np.zeros((N_train,3))
    for i in range(3):
        if model == 'pendulum':
            f_train[:,i] = u_xx_train[:,i] + np.sin(u_train[:,i])
    f_train = f_train.T.flatten()  # 9 * 1
    return f_train

def f_Test(model):
    f_test = np.zeros((N_test,3))
    for i in range(3):
        if model == 'pendulum':
            f_test[:,i] = u_xx_test[:,i] + np.sin(u_test[:,i])
    f_test = f_test.T.flatten()   # 30 * 1
    return f_test

def f_true_Test(model, s_test):
    if model == 'pendulum':
        return s_test[:,3] + np.sin(s_test[:,1])