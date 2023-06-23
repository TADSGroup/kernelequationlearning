import numpy as np
from models import *

# Pendulum
N = 10
# Locations known per function
M1 = np.random.randint(2, size=10)
M2 = np.random.randint(2, size=10)
M3 = np.random.randint(2, size=10)
M = np.array(np.concatenate((M1,M2,M3)), dtype=bool)

m = 3
np.random.seed(2023)
x_train = np.sort(np.random.uniform(low = 0.0, high = 1.0, size = N))
x_test = np.linspace(start = 0.0, stop = 1.0, num = N//3)

u_train, u_x_train, u_xx_train = ODE_solutions(x_train,k=2,d=3,c=4)
u_test,  u_x_test,  u_xx_test  = ODE_solutions(x_test, k=2,d=3,c=4)



x_train_all = np.concatenate([x_train,x_train,x_train]).reshape(-1,1) # 300 * 1

def f_Train(model):
    f_train = np.zeros((N,3))
    for i in range(3):
        if model == 'pendulum':
            f_train[:,i] = u_xx_train[:,i] + np.sin(u_train[:,i])
    f_train = f_train.T.flatten()  # 300 * 1
    return f_train

def f_true_Test(model, s_test):
    if model == 'pendulum':
        return s_test[:,3] + np.sin(s_test[:,1])