import numpy as np
from scipy.io import loadmat
import pysindy as ps
from models import *


data = np.load('/home/juanfelipe/Desktop/research/keql/examples/Burgers/gen_data/sols_burgers.npy')
#print(data[0,:,:].shape)

np.random.seed(9)

# Number of functions
m = 2
# Ghost_training (gh_tr): Uniform grid. Same per function.
N_t_gh_tr, N_x_gh_tr = 40, 40
N_gh_tr = N_t_gh_tr*N_x_gh_tr
# Training (tr): Randomly sampled from Ghost_training. Different per function.
N_t_tr, N_x_tr = 10, 10
N_tr = N_t_tr*N_x_tr
# Testing (te): Randomly sampled from Supergrid \ Ghost_training. Same per function
N_t_te, N_x_te = 20, 20
N_te = N_t_te*N_x_te


# t
L_t = 8
dt = 0.025
N_t = int(L_t/dt)
t = np.linspace(0, L_t, N_t) # (320,)
idx_t_gh_tr = np.round(np.linspace(0,len(t)-1, int(N_t_gh_tr))).astype(int)
t_gh_tr = t[idx_t_gh_tr] # (N_t_gh_tr,)
idx_t_te = np.round(np.linspace(0,len(t)-1, int(N_t_te))).astype(int)
t_te = t[idx_t_te] # (N_t_te,)

# x
L_x = 10 
dx = 0.1 
N_x = int(L_x/dx)
x = np.linspace(0,L_x,N_x) # (100,)
idx_x_gh_tr = np.round(np.linspace(0,len(x)-1, int(N_x_gh_tr))).astype(int)
x_gh_tr = x[idx_x_gh_tr] # (N_x_gh_tr,)
idx_x_te = np.round(np.linspace(0,len(x)-1, int(N_x_te))).astype(int)
x_te = x[idx_x_te] # (N_x_te,)

# Size of the grid
N = N_t*N_x

# (t,x)- full meshgrid
TT, XX = np.meshgrid(t,x) # (100,320) , (100,320)
pairs = np.vstack([TT.ravel(), XX.ravel()]) # (2, 32000)
# (t,x)- gh_tr meshgrid
TT_gh_tr, XX_gh_tr = np.meshgrid(t_gh_tr,x_gh_tr) # (N_t_gh_tr, N_x_gh_tr) , (N_t_gh_tr, N_x_gh_tr)
pairs_gh_tr = np.vstack([TT_gh_tr.ravel(), XX_gh_tr.ravel()]) # (2, N_gh_tr)
# (t,x)- te meshgrid
TT_te, XX_te = np.meshgrid(t_te,x_te) # (N_t_te, N_x_te) , (N_t_te, N_x_te)
pairs_te = np.vstack([TT_te.ravel(), XX_te.ravel()]) # (2, N_te)

# Initialize arrays
# U
U = []
U_gh_tr = []
U_tr = []
U_te = []
# U_t
U_t = []
U_t_gh_tr = []
U_t_tr = []
U_t_te = []
# U_x
U_x = []
U_x_gh_tr = []
U_x_tr = []
U_x_te = []
# U_xx
U_xx = []
U_xx_gh_tr = []
U_xx_tr = []
U_xx_te = []
# X
X = []
X_gh_tr = []
X_tr = []
X_te = []

# M
M_gh_tr = []

for i in range(1,m+1):

    # Training(collocation) points
    # (t,x)- (2, N_tr)
    idx_tr = np.random.choice(np.arange(N_gh_tr), size = N_tr, replace = False)




    # u - (100, 320)
    u = data[i,:,:]
    U.append(u.flatten())
    # u_gh_tr - (N_t_gh_tr, N_x_gh_tr) 
    u_gh_tr = u[np.ix_(idx_x_gh_tr, idx_t_gh_tr)]
    triples_gh_tr = np.vstack([pairs_gh_tr,u_gh_tr.flatten()]).T # (N_gh_tr, 3)
    U_gh_tr.append(u_gh_tr.flatten())
    # u_tr - (N_t_tr, N_x_tr) 
    u_tr = triples_gh_tr[idx_tr,:][:,-1].reshape(N_t_tr, N_x_tr)
    U_tr.append(u_tr.flatten())
    # u_te - (N_t_te, N_x_te) 
    u_te = u[np.ix_(idx_x_te, idx_t_te)]
    U_te.append(u_te.flatten())

    # u_t - (100, 320)
    u_t = ps.FiniteDifference(axis=1, order=15)._differentiate(u, t=dt)
    U_t.append(u_t.flatten())
    # u_t_gh_tr - (N_t_gh_tr, N_x_gh_tr)
    u_t_gh_tr = u_t[np.ix_(idx_x_gh_tr, idx_t_gh_tr)]
    triples_t_gh_tr = np.vstack([pairs_gh_tr,u_t_gh_tr.flatten()]).T # (N_gh_tr, 3)
    U_t_gh_tr.append(u_t_gh_tr.flatten())
    # u_t_tr - (N_t_tr, N_x_tr) 
    u_t_tr = triples_t_gh_tr[idx_tr,:][:,-1].reshape(N_t_tr, N_x_tr)    
    U_t_tr.append(u_t_tr.flatten())
    # u_t_te - (N_t_te, N_x_te) 
    u_t_te = u_t[np.ix_(idx_x_te, idx_t_te)]
    U_t_te.append(u_t_te.flatten())

    # u_x - (100, 320)
    u_x = ps.FiniteDifference(axis=0, order=4)._differentiate(u, t=dx)
    U_x.append(u_x.flatten())
    # u_x_gh_tr - (N_t_gh_tr, N_x_gh_tr)
    u_x_gh_tr = u_x[np.ix_(idx_x_gh_tr, idx_t_gh_tr)]
    triples_x_gh_tr = np.vstack([pairs_gh_tr,u_x_gh_tr.flatten()]).T # (N_gh_tr, 3)
    U_x_gh_tr.append(u_x_gh_tr.flatten())
    # u_x_tr - (N_t_tr, N_x_tr) 
    u_x_tr = triples_x_gh_tr[idx_tr,:][:,-1].reshape(N_t_tr, N_x_tr)    
    U_x_tr.append(u_x_tr.flatten())
    # u_x_te - (N_t_te, N_x_te) 
    u_x_te = u_x[np.ix_(idx_x_te, idx_t_te)]
    U_x_te.append(u_x_te.flatten())

    # u_xx - (100, 320)
    u_xx = ps.FiniteDifference(axis=0, order=4, d=2)._differentiate(u, t=dx)
    U_xx.append(u_xx.flatten())
    # u_xx_gh_tr - (N_t_gh_tr, N_x_gh_tr)
    u_xx_gh_tr = u_xx[np.ix_(idx_x_gh_tr, idx_t_gh_tr)]
    triples_xx_gh_tr = np.vstack([pairs_gh_tr,u_xx_gh_tr.flatten()]).T # (N_gh_tr, 3)
    U_xx_gh_tr.append(u_xx_gh_tr.flatten())
    # u_xx_tr - (N_t_tr, N_x_tr) 
    u_xx_tr = triples_xx_gh_tr[idx_tr,:][:,-1].reshape(N_t_tr, N_x_tr)    
    U_xx_tr.append(u_xx_tr.flatten())
    # u_xx_te - (N_t_te, N_x_te) 
    u_xx_te = u_xx[np.ix_(idx_x_te, idx_t_te)]
    U_xx_te.append(u_xx_te.flatten())


    # X
    X.append(pairs.T)
    # X_gh_tr
    X_gh_tr.append(pairs_gh_tr.T)
    # X_tr
    pairs_tr = pairs_gh_tr.T[idx_tr] #(N_t_tr*N_x_tr, 2)
    X_tr.append(pairs_tr)
    # X_te
    X_te.append(pairs_te.T)

    
    # M
    m_gh_tr = np.zeros(N_gh_tr, dtype=bool)
    m_gh_tr[idx_tr] = True
    M_gh_tr.append(m_gh_tr)

# Stack everything on top of each other
# U
U = np.vstack(U).T # (32000, m)
U_gh_tr = np.vstack(U_gh_tr).T # (N_gh_tr, m)
U_tr = np.vstack(U_tr).T # (N_tr, m)
U_te = np.vstack(U_te).T # (N_te, m)
# U_t
U_t = np.vstack(U_t).T # (32000, m)
U_t_gh_tr = np.vstack(U_t_gh_tr).T # (N_gh_tr, m)
U_t_tr = np.vstack(U_t_tr).T # (N_tr, m)
U_t_te = np.vstack(U_t_te).T # (N_te, m)
# U_x
U_x = np.vstack(U_x).T # (32000, m)
U_x_gh_tr = np.vstack(U_x_gh_tr).T # (N_gh_tr, m)
U_x_tr = np.vstack(U_x_tr).T # (N_tr, m)
U_x_te = np.vstack(U_x_te).T # (N_te, m)
# U_xx
U_xx = np.vstack(U_xx).T # (32000, m)
U_x_gh_tr = np.vstack(U_xx_gh_tr).T # (N_gh_tr, m)
U_xx_tr = np.vstack(U_xx_tr).T # (N_tr, m)
U_xx_te = np.vstack(U_xx_te).T # (N_te, m)
# X
X = np.vstack(X) # (m*32000,2)
X_gh_tr = np.vstack(X_gh_tr) # (m*N_gh_tr,2)
X_tr = np.vstack(X_tr) # (m*N_tr,2)
X_te = np.vstack(X_te) # (m*N_te,2)
# M
M_gh_tr = np.hstack(M_gh_tr)




