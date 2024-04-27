import numpy as np
from models import *
		
np.random.seed(9)
N = 10
m = 3
N_train = 3 
N_test = N

x_test  = np.linspace(start=0.0, stop=1.0, num=N)

x_train = np.zeros((m,N_train)) # m * N_train
for i in range(m):
    x_train[i] = np.sort(np.random.choice(x_test, size=N_train, replace=False))

# Create a boolean array of size N initialized with False
M = np.zeros((m,N), dtype=bool)

# Get indices of the N_train points selected and set the values at specified indices to True
idx = []
for i in range(m):
    index = np.where(np.in1d(x_test,x_train[i]))[0]
    idx.append(index)
    M[i][index] = True

idx = np.array(idx)

x_test =  np.concatenate([x_test,   x_test,   x_test]) # m N * 1
x_train = x_train.flatten() # m N_train  * 1

u_train, u_x_train, u_xx_train = ODE_solutions(x_train, N_train, k=2, d=3, c=4)
u_test,  u_x_test,  u_xx_test  = ODE_solutions(x_test,  N_test,  k=2 ,d=3, c=4)

def f_Train(model):
    f_train = np.zeros((N_train,m))
    for i in range(m):
        if model == 'pendulum':
            f_train[:,i] = u_xx_train[:,i] + np.sin(u_train[:,i])
    f_train = f_train.T.flatten()  # 9 * 1
    return f_train

def f_Test(model):
    f_test = np.zeros((N_test,m))
    for i in range(m):
        if model == 'pendulum':
            f_test[:,i] = u_xx_test[:,i] + np.sin(u_test[:,i])
    f_test = f_test.T.flatten()   # 30 * 1
    return f_test

def f_true_Test(model, s_test):
    if model == 'pendulum':
        return s_test[:,3] + np.sin(s_test[:,1])