import numpy as np
from scipy.io import loadmat
import pysindy as ps
from models import *


data = np.load('/home/juanfelipe/Desktop/research/keql/examples/Burgers/gen_data/sols_burgers.npy')
print(data[0,:,:].shape)
np.random.seed(9)

# Number of functions
m = 2

# Number of points to be sampled per function

# Ghost_training (gh_tr): Uniform grid. Same per function.
N_t_gh_tr, N_x_gh_tr = 40, 40
N_gh_tr = N_t_gh_tr*N_x_gh_tr
# Training (tr): Randomly sampled from Ghost_training. Different per function.
N_t_tr, N_x_tr = 30, 30
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
idx_t_te = np.random.choice(np.setdiff1d(np.arange(t.size),idx_t_gh_tr), size = N_t_te, replace = False)
t_te = np.sort(t[idx_t_te]) # (N_t_te,)

# x
L_x = 10 
dx = 0.1 
N_x = int(L_x/dx)
x = np.linspace(0,L_x,N_x) # (100,)
idx_x_gh_tr = np.round(np.linspace(0,len(x)-1, int(N_x_gh_tr))).astype(int)
x_gh_tr = x[idx_x_gh_tr] # (N_x_gh_tr,)
idx_x_te = np.random.choice(np.setdiff1d(np.arange(x.size),idx_x_gh_tr), size = N_x_te, replace = False)
x_te = np.sort(x[idx_x_te]) # (N_x_te,)

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

for i in range(m):

    # Training(collocation) points
    # t
    idx_t_tr = np.random.choice(idx_t_gh_tr, size = N_t_tr, replace = False)
    t_tr = t[idx_t_tr] # (N_t_tr,)
    # x
    idx_x_tr = np.random.choice(idx_x_gh_tr, size = N_x_tr, replace = False)
    x_tr = x[idx_x_tr] # (N_x_tr,)
    # (t,x)- (2, N_tr)
    idx_tr = np.random.choice(np.arange(N_gh_tr), size = N_tr, replace = False)
    

    # (t,x)- tr meshgrid 
    #TT_tr, XX_tr = np.meshgrid(t_tr,x_tr) # (N_t_tr, N_x_tr) , (N_t_tr, N_x_tr)




    # u - (100, 320)
    u = data[i,:,:]
    U.append(u.flatten())
    # u_gh_tr - (N_t_gh_tr, N_x_gh_tr) 
    u_gh_tr = u[np.ix_(idx_x_gh_tr, idx_t_gh_tr)]
    triples_gh_tr = np.vstack([pairs_gh_tr,u_gh_tr.flatten()]).T # (N_gh_tr, 3)
    U_gh_tr.append(u_gh_tr.flatten())
    # u_tr - (N_t_tr, N_x_tr) 
    #u_tr = u[np.ix_(idx_x_tr, idx_t_tr)]
    u_tr = triples_gh_tr[idx_tr,:][:,-1].reshape(N_t_tr, N_x_tr)
    U_tr.append(u_tr)
    # u_te - (N_t_te, N_x_te) 
    u_te = u[np.ix_(idx_x_te, idx_t_te)]
    U_te.append(u_te.flatten())

    # u_t - (100, 320)
    u_t = ps.FiniteDifference(axis=1, order=10)._differentiate(u, t=dt)
    U_t.append(u_t.flatten())
    # u_t_gh_tr - (N_t_gh_tr, N_x_gh_tr)
    u_t_gh_tr = u_t[np.ix_(idx_x_gh_tr, idx_t_gh_tr)]
    U_t_gh_tr.append(u_t_gh_tr.flatten())
    # u_t_tr - (N_t_tr, N_x_tr) 
    u_t_tr = u_t[np.ix_(idx_x_tr, idx_t_tr)]    
    U_t_tr.append(u_t_tr.flatten())
    # u_t_te - (N_t_te, N_x_te) 
    u_t_te = u_t[np.ix_(idx_x_te, idx_t_te)]
    U_t_te.append(u_t_te.flatten())

    # u_x - (100, 320)
    u_x = ps.FiniteDifference(axis=0, order=4)._differentiate(u, t=dx)
    U_x.append(u_x.flatten())
    # u_x_gh_tr - (N_t_gh_tr, N_x_gh_tr)
    u_x_gh_tr = u_x[np.ix_(idx_x_gh_tr, idx_t_gh_tr)]
    U_x_gh_tr.append(u_x_gh_tr.flatten())
    # u_x_tr - (N_t_tr, N_x_tr) 
    u_x_tr = u_x[np.ix_(idx_x_tr, idx_t_tr)]    
    U_x_tr.append(u_x_tr.flatten())
    # u_x_te - (N_t_te, N_x_te) 
    u_x_te = u_x[np.ix_(idx_x_te, idx_t_te)]
    U_x_te.append(u_x_te.flatten())

    # u_xx - (100, 320)
    u_xx = ps.FiniteDifference(axis=0, order=4, d=2)._differentiate(u, t=dx)
    U_xx.append(u_xx.flatten())
    # u_xx_gh_tr - (N_t_gh_tr, N_x_gh_tr)
    u_xx_gh_tr = u_xx[np.ix_(idx_x_gh_tr, idx_t_gh_tr)]
    U_xx_gh_tr.append(u_xx_gh_tr.flatten())
    # u_xx_tr - (N_t_tr, N_x_tr) 
    u_xx_tr = u_xx[np.ix_(idx_x_tr, idx_t_tr)]    
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



# # Create a boolean array of size N initialized with False
M = np.zeros((m, N_te), dtype=bool)
# # Set the values at specified indices to True
# idx = []
# for i in range(m):
#     index = np.where(np.in1d(X_test,X_train[i]))[0]
#     idx.append(index)
#     M[i][index] = True




# all_pairs = np.vstack([T.ravel(), X.ravel()])
# # (256, 2)
# all_pairs_ghost = np.vstack([T_ghost.ravel(), X_ghost.ravel()]).T

# # At the grid points
# U = []
# U_t = []
# U_x = []
# U_xx = []
# # U_ghost
# U_ghost = []
# U_t_ghost = []
# U_x_ghost = []
# U_xx_ghost = []
# # At collocation points
# U_train = []
# U_t_train = []
# U_x_train = []
# U_xx_train = []
# X_train = []
# # At testing points
# U_test = []
# U_t_test = []
# U_x_test = []
# U_xx_test = []
# X_test = []


# #idx_test = np.random.randint(len(all_pairs[0]), size = N_test)
# idx_test = np.random.randint(len(all_pairs[0]), size = N_test)

# for i in range(m):

#     # u - (100, 320)
#     u = data[i,:,:]
#     U.append(u)
#     # u_ghost - (16,16)
#     u_ghost = u[np.ix_(idx_x_ghost, idx_t_ghost)]
#     U_ghost.append(u_ghost)
#     # u_t, u_x, u_xx - (100, 320)
#     u_t = ps.FiniteDifference(axis=1, order=4)._differentiate(u, t=dt)
#     U_t.append(u_t)
#     u_x = ps.FiniteDifference(axis=0, order=4)._differentiate(u, t=dx)
#     U_x.append(u_x)
#     u_xx = ps.FiniteDifference(axis=0, order=4, d=2)._differentiate(u, t=dx)
#     U_xx.append(u_xx)
#     # u_t_ghost, u_x_ghost, u_xx_ghost - (16, 16)
#     u_t_ghost = u_t[np.ix_(idx_x_ghost, idx_t_ghost)]
#     U_t_ghost.append(u_t_ghost)
#     u_x_ghost = u_x[np.ix_(idx_x_ghost, idx_t_ghost)]
#     U_x_ghost.append(u_x_ghost)
#     u_xx_ghost = u_xx[np.ix_(idx_x_ghost, idx_t_ghost)]
#     U_xx_ghost.append(u_xx_ghost)

#     # Stack all triples (t,x,u), (t,x,u_t), (t,x,u_x), (t,x,u_xx) - (32000, 3)
#     u_all = np.vstack([all_pairs, u.flatten()]).T 
#     u_t_all = np.vstack([all_pairs, u_t.flatten()]).T 
#     u_x_all = np.vstack([all_pairs, u_x.flatten()]).T 
#     u_xx_all = np.vstack([all_pairs, u_xx.flatten()]).T 

#     # Stack all ghost triples (t,x,u), (t,x,u_t), (t,x,u_x), (t,x,u_xx) - (32000, 3)
#     u_all_ghost = np.vstack([all_pairs_ghost, u_ghost.flatten()]).T 
#     u_t_all_ghost = np.vstack([all_pairs_ghost, u_t_ghost.flatten()]).T 
#     u_x_all_ghost = np.vstack([all_pairs_ghost, u_x_ghost.flatten()]).T 
#     u_xx_all_ghost = np.vstack([all_pairs_ghost, u_xx_ghost.flatten()]).T 

#     # Get random indices
#     idx_train = np.random.choice(np.arange(u_all_ghost.size), size = N_train, replace = False)


#     # Get training and testing points from triples

#     # u
#     u_train_ = u_all_ghost[idx_train,:]
#     u_test_ = u_all[idx_test,:]
#     t_train, x_train, u_train = u_train_[:,0], u_train_[:,1], u_train_[:,2]
#     t_test, x_test, u_test = u_test_[:,0], u_test_[:,1], u_test_[:,2]
#     U_train.append(u_train)
#     U_test.append(u_test)
#     # u_t
#     u_t_train_ = u_t_all[idx_train,:]
#     u_t_test_ = u_t_all[idx_test,:]
#     t_train, x_train, u_t_train = u_t_train_[:,0], u_t_train_[:,1], u_t_train_[:,2]
#     t_test, x_test, u_t_test = u_t_test_[:,0], u_t_test_[:,1], u_t_test_[:,2]
#     U_t_train.append(u_t_train)
#     U_t_test.append(u_t_test)
#     # u_x
#     u_x_train_ = u_x_all[idx_train,:]
#     u_x_test_ = u_x_all[idx_test,:]
#     t_train, x_train, u_x_train = u_x_train_[:,0], u_x_train_[:,1], u_x_train_[:,2]
#     t_test, x_test, u_x_test = u_x_test_[:,0], u_x_test_[:,1], u_x_test_[:,2]
#     U_x_train.append(u_x_train)
#     U_x_test.append(u_x_test)
#     # u_xx
#     u_xx_train_ = u_xx_all[idx_train,:]
#     u_xx_test_ = u_xx_all[idx_test,:]
#     t_train, x_train, u_xx_train = u_xx_train_[:,0], u_xx_train_[:,1], u_xx_train_[:,2]
#     t_test, x_test, u_xx_test = u_xx_test_[:,0], u_xx_test_[:,1], u_xx_test_[:,2]
#     U_xx_train.append(u_xx_train)
#     U_xx_test.append(u_xx_test)
#     # Training
#     X_train_c = np.vstack([t_train, x_train]).T
#     X_train.append(X_train_c)
#     # Testing
#     X_test_c = np.vstack([t_test, x_test]).T
#     X_test.append(X_test_c)


# # At collocation points
# U_train = np.vstack(U_train).T # (N_train, m)
# U_t_train = np.vstack(U_t_train).T # (N_train, m)
# U_x_train = np.vstack(U_x_train).T # (N_train, m)
# U_xx_train = np.vstack(U_xx_train).T # (N_train, m)
# X_train = np.vstack(X_train) # (m*N_train, 2)
# # At ghost + collocation points
# U_test = np.vstack(U_test).T # (N_train, m)
# U_t_test = np.vstack(U_t_test).T # (N_train, m)
# U_x_test = np.vstack(U_x_test).T # (N_train, m)
# U_xx_test = np.vstack(U_xx_test).T # (N_train, m)
# X_test = np.vstack(X_test) # (m*N_test, 2)



