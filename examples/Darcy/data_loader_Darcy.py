import numpy as np
import jax.numpy as jnp
from jax import jacfwd, grad, jit, vmap
from scipy.io import loadmat
import pysindy as ps
from models import *

# True solution: Scalar field u(x1, x2)
def u(x1,x2,beta):
    return jnp.exp(jnp.sin(beta*(jnp.cos(x1) + jnp.cos(x2))))
# True solution: Scalar field u_x1(x1,x2)
def u_x1(x1,x2,beta):
    grad0 = grad(u, 0)
    return grad0(x1,x2,beta)

# True solution: Scalar field u_x2(x1,x2)
def u_x1x1(x1,x2,beta):
    gradgrad2 = grad(grad(u, 0),0)
    return gradgrad2(x1,x2,beta)

# True solution: Scalar field u_x2(x1,x2)
def u_x2(x1,x2,beta):
    grad1 = grad(u, 1)
    return grad1(x1,x2,beta)
# True solution: Scalar field u_x2(x1,x2)
def u_x2x2(x1,x2,beta):
    gradgrad1 = grad(grad(u, 1),1)
    return gradgrad1(x1,x2,beta)

# Vector field A(x1,x2)
def A(x1,x2):
    # return jnp.exp(jnp.sin(jnp.cos(x1) + jnp.cos(x2)))
    return 2.
# A*grad(u_star)
def A_times_grad_u(x1,x2,beta):
    grad_u = jacfwd(u, argnums=[0,1])
    return A(x1,x2)*jnp.array(grad_u(x1,x2,beta))
# rhs = -div(A*grad(u_star))
def rhs(x1,x2,beta):
    div = jacfwd(A_times_grad_u, argnums=[0,1])
    return -jnp.trace(jnp.array(div(x1,x2,beta)))

#data = np.load('/home/juanfelipe/Desktop/research/keql/examples/Burgers/gen_data/sols_burgers.npy')
#print(data[0,:,:].shape)

np.random.seed(9)

# Number of functions
m = 3
# Ghost_training (gh_tr): Uniform grid. Same per function.
N_x1_gh_tr, N_x2_gh_tr = 20, 20
N_gh_tr = N_x1_gh_tr*N_x2_gh_tr
# Training (tr): Randomly sampled from Ghost_training. Different per function.
N_x1_tr, N_x2_tr = 3, 3
N_tr = N_x1_tr*N_x2_tr
# Testing (te): Randomly sampled from Supergrid \ Ghost_training. Same per function
N_x1_te, N_x2_te = 10, 10
N_te = N_x1_te*N_x2_te


# x1
L_x1 = 1
dx1 = 0.001
N_x1 = int(L_x1/dx1)
x1 = np.linspace(0, L_x1, N_x1) # (1000,)
idx_x1_gh_tr = np.round(np.linspace(0,len(x1)-1, int(N_x1_gh_tr))).astype(int)
x1_gh_tr = x1[idx_x1_gh_tr] # (N_x1_gh_tr,)
idx_x1_te = np.round(np.linspace(0,len(x1)-1, int(N_x1_te))).astype(int)
x1_te = x1[idx_x1_te] # (N_x1_te,)

# x2
L_x2 = 1
dx2 = 0.001 
N_x2 = int(L_x2/dx2)
x2 = np.linspace(0,L_x2,N_x2) # (1000,)
idx_x2_gh_tr = np.round(np.linspace(0,len(x2)-1, int(N_x2_gh_tr))).astype(int)
x2_gh_tr = x2[idx_x2_gh_tr] # (N_x2_gh_tr,)
idx_x2_te = np.round(np.linspace(0,len(x2)-1, int(N_x2_te))).astype(int)
x2_te = x2[idx_x2_te] # (N_x2_te,)

# Size of the grid
N = N_x1*N_x2

# (x1,x2)- full meshgrid
X1X1, X2X2 = np.meshgrid(x1,x2) # (1000,1000) , (1000,1000)
pairs = np.vstack([X1X1.ravel(), X2X2.ravel()]).T # (1 000 000, 2)
# (x1,x2)- gh_tr meshgrid
X1X1_gh_tr, X2X2_gh_tr = np.meshgrid(x1_gh_tr,x2_gh_tr) # (N_x1_gh_tr, N_x2_gh_tr) , (N_x1_gh_tr, N_x2_gh_tr)
pairs_gh_tr = np.vstack([X1X1_gh_tr.ravel(), X2X2_gh_tr.ravel()]).T # (N_gh_tr, 2)
# (x1,x2)- te meshgrid
X1X1_te, X2X2_te = np.meshgrid(x1_te,x2_te) # (N_x1_te, N_x2_te) , (N_x1_te, N_x2_te)
pairs_te = np.vstack([X1X1_te.ravel(), X2X2_te.ravel()]).T # (N_te, 2)

# Initialize arrays
# U
U = []
U_gh_tr = []
U_tr = []
U_te = []
# U_x1
U_x1 = []
U_x1_gh_tr = []
U_x1_tr = []
U_x1_te = []
# U_x1x1
U_x1x1 = []
U_x1x1_gh_tr = []
U_x1x1_tr = []
U_x1x1_te = []
# U_x2
U_x2 = []
U_x2_gh_tr = []
U_x2_tr = []
U_x2_te = []
# U_x2x2
U_x2x2 = []
U_x2x2_gh_tr = []
U_x2x2_tr = []
U_x2x2_te = []
# F
F = []
F_gh_tr = []
F_tr = []
F_te = []
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


    # u - (1000, 1000)
    utrue = np.array(vmap(lambda t: u(t[0], t[1], i))(pairs))
    U.append(utrue.flatten())
    # u_gh_tr - (N_x1_gh_tr, N_x2_gh_tr) 
    u_gh_tr = np.array(vmap(lambda t: u(t[0], t[1], i))(pairs_gh_tr))
    # triples_gh_tr = np.vstack([pairs_gh_tr,u_gh_tr.flatten()]).T # (N_gh_tr, 3)
    U_gh_tr.append(u_gh_tr.flatten())
    # u_tr - (N_t_tr, N_x_tr) 
    pairs_tr = pairs_gh_tr[idx_tr] #(N_tr, 2)
    u_tr = np.array(vmap(lambda t: u(t[0], t[1], i))(pairs_tr))
    U_tr.append(u_tr.flatten())
    # u_te - (N_x1_te, N_x2_te) 
    u_te = np.array(vmap(lambda t: u(t[0], t[1], i))(pairs_te))
    U_te.append(u_te.flatten())

    # u_x1 - (1000, 1000)
    utrue_x1 = np.array(vmap(lambda t: u_x1(t[0], t[1], i))(pairs))
    U_x1.append(utrue_x1.flatten())
    # u_x1_gh_tr - (N_t_gh_tr, N_x_gh_tr)
    utrue_x1_gh_tr = np.array(vmap(lambda t: u_x1(t[0], t[1], i))(pairs_gh_tr))
    U_x1_gh_tr.append(utrue_x1_gh_tr.flatten())
    # u_x1_tr - (N_x1_tr, N_x2_tr) 
    utrue_x1_tr = np.array(vmap(lambda t: u_x1(t[0], t[1], i))(pairs_tr))
    U_x1_tr.append(utrue_x1_tr.flatten())
    # u_x1_te - (N_x1_te, N_x2_te) 
    utrue_x1_te = np.array(vmap(lambda t: u_x1(t[0], t[1], i))(pairs_te))
    U_x1_te.append(utrue_x1_te.flatten())

    # u_x1x1 - (1000, 1000)
    utrue_x1x1 = np.array(vmap(lambda t: u_x1x1(t[0], t[1], i))(pairs))
    U_x1x1.append(utrue_x1x1.flatten())
    # u_x1x1_gh_tr - (N_t_gh_tr, N_x_gh_tr)
    utrue_x1x1_gh_tr = np.array(vmap(lambda t: u_x1x1(t[0], t[1], i))(pairs_gh_tr))
    U_x1x1_gh_tr.append(utrue_x1x1_gh_tr.flatten())
    # u_x1x1_tr - (N_x1x1_tr, N_x1x1_tr) 
    utrue_x1x1_tr = np.array(vmap(lambda t: u_x1x1(t[0], t[1], i))(pairs_tr))
    U_x1x1_tr.append(utrue_x1x1_tr.flatten())
    # u_x1x1_te - (N_x1x1_te, N_x1x1_te) 
    utrue_x1x1_te = np.array(vmap(lambda t: u_x1x1(t[0], t[1], i))(pairs_te))
    U_x1x1_te.append(utrue_x1x1_te.flatten())


    # u_x2 - (1000, 1000)
    utrue_x2 = np.array(vmap(lambda t: u_x2(t[0], t[1], i))(pairs))
    U_x2.append(utrue_x2.flatten())
    # u_x2_gh_tr - (N_t_gh_tr, N_x_gh_tr)
    utrue_x2_gh_tr = np.array(vmap(lambda t: u_x2(t[0], t[1], i))(pairs_gh_tr))
    U_x2_gh_tr.append(utrue_x2_gh_tr.flatten())
    # u_x2_tr - (N_x2_tr, N_x2_tr) 
    utrue_x2_tr = np.array(vmap(lambda t: u_x2(t[0], t[1], i))(pairs_tr))
    U_x2_tr.append(utrue_x2_tr.flatten())
    # u_x2_te - (N_x2_te, N_x2_te) 
    utrue_x2_te = np.array(vmap(lambda t: u_x2(t[0], t[1], i))(pairs_te))
    U_x2_te.append(utrue_x2_te.flatten())

    # u_x2x2 - (1000, 1000)
    utrue_x2x2 = np.array(vmap(lambda t: u_x2x2(t[0], t[1], i))(pairs))
    U_x2x2.append(utrue_x2x2.flatten())
    # u_x2x2_gh_tr - (N_t_gh_tr, N_x_gh_tr)
    utrue_x2x2_gh_tr = np.array(vmap(lambda t: u_x2x2(t[0], t[1], i))(pairs_gh_tr))
    U_x2x2_gh_tr.append(utrue_x2x2_gh_tr.flatten())
    # u_x2x2_tr - (N_x2x2_tr, N_x2x2_tr) 
    utrue_x2x2_tr = np.array(vmap(lambda t: u_x2x2(t[0], t[1], i))(pairs_tr))
    U_x2x2_tr.append(utrue_x2x2_tr.flatten())
    # u_x2x2_te - (N_x2x2_te, N_x2x2_te) 
    utrue_x2x2_te = np.array(vmap(lambda t: u_x2x2(t[0], t[1], i))(pairs_te))
    U_x2x2_te.append(utrue_x2x2_te.flatten())


    # f - (1000, 1000)
    ftrue = np.array(vmap(lambda t: rhs(t[0], t[1], i))(pairs))
    F.append(ftrue.flatten())
    # f_gh_tr - (N_x1_gh_tr, N_x2_gh_tr) 
    ftrue_gh_tr = np.array(vmap(lambda t: rhs(t[0], t[1], i))(pairs_gh_tr))
    F_gh_tr.append(ftrue_gh_tr.flatten())
    # f_tr - (N_t_tr, N_x_tr) 
    pairs_tr = pairs_gh_tr[idx_tr] #(N_tr, 2)
    ftrue_tr = np.array(vmap(lambda t: rhs(t[0], t[1], i))(pairs_tr))
    F_tr.append(ftrue_tr.flatten())
    # f_te - (N_x1_te, N_x2_te) 
    ftrue_te = np.array(vmap(lambda t: rhs(t[0], t[1], i))(pairs_te))
    F_te.append(ftrue_te.flatten())

    # X
    X.append(pairs)
    # X_gh_tr
    X_gh_tr.append(pairs_gh_tr)
    # X_tr
    X_tr.append(pairs_tr)
    # X_te
    X_te.append(pairs_te)

    
    # M
    m_gh_tr = np.zeros(N_gh_tr, dtype=bool)
    m_gh_tr[idx_tr] = True
    M_gh_tr.append(m_gh_tr)

# Stack everything on top of each other
# U
U = np.vstack(U).T # (1 000 000, m)
U_gh_tr = np.vstack(U_gh_tr).T # (N_gh_tr, m)
U_tr = np.vstack(U_tr).T # (N_tr, m)
U_te = np.vstack(U_te).T # (N_te, m)
# U_x1
U_x1 = np.vstack(U_x1).T # (1 000 000, m)
U_x1_gh_tr = np.vstack(U_x1_gh_tr).T # (N_gh_tr, m)
U_x1_tr = np.vstack(U_x1_tr).T # (N_tr, m)
U_x1_te = np.vstack(U_x1_te).T # (N_te, m)
# U_x1x1
U_x1x1 = np.vstack(U_x1x1).T # (1 000 000, m)
U_x1x1_gh_tr = np.vstack(U_x1x1_gh_tr).T # (N_gh_tr, m)
U_x1x1_tr = np.vstack(U_x1x1_tr).T # (N_tr, m)
U_x1x1_te = np.vstack(U_x1x1_te).T # (N_te, m)
# U_x2
U_x2 = np.vstack(U_x2).T # (1 000 000, m)
U_x2_gh_tr = np.vstack(U_x2_gh_tr).T # (N_gh_tr, m)
U_x2_tr = np.vstack(U_x2_tr).T # (N_tr, m)
U_x2_te = np.vstack(U_x2_te).T # (N_te, m)
# U_x2x2
U_x2x2 = np.vstack(U_x2x2).T # (1 000 000, m)
U_x2x2_gh_tr = np.vstack(U_x2x2_gh_tr).T # (N_gh_tr, m)
U_x2x2_tr = np.vstack(U_x2x2_tr).T # (N_tr, m)
U_x2x2_te = np.vstack(U_x2x2_te).T # (N_te, m)
# F
F = np.vstack(F).T # (1 000 000, m)
F_gh_tr = np.vstack(F_gh_tr).T # (N_gh_tr, m)
F_tr = np.vstack(F_tr).T # (N_tr, m)
F_te = np.vstack(F_te).T # (N_te, m)
# X
X = np.vstack(X) # (m*1 000 000,2)
X_gh_tr = np.vstack(X_gh_tr) # (m*N_gh_tr,2)
X_tr = np.vstack(X_tr) # (m*N_tr,2)
X_te = np.vstack(X_te) # (m*N_te,2)
# M
M_gh_tr = np.hstack(M_gh_tr)




