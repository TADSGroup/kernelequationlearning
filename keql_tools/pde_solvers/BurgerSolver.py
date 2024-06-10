import numpy as np
from scipy.sparse import diags_array
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.sparse.linalg import spsolve
from tqdm.auto import tqdm
from scipy.sparse import identity
from scipy.sparse import eye_array


def get_burger_solver(alpha,kappa,k = 1e-4,n = 1000):
    h = 1/(n+1)


    L = diags_array([np.ones(n-1),-2 * np.ones(n),np.ones(n-1)],offsets = [-1,0,1])/(h**2)
    D = diags_array([-np.ones(n-1),np.ones(n-1)],offsets = [-1,1])/(2*h)

    xgrid = np.linspace(0,1,n+2)
    def heat_step(k,kappa,n):
        B = L * kappa * k
        I = eye_array(n)
        L1 = I - B/4
        R1 = I + B/4
        L2 = I - B/3
        def trbdf2_heat(u):
            #u1 = spsolve(L2,R1@u)
            u1 = spsolve(L1,R1@u)
            return spsolve(L2,(1/3) * (4 * u1 - u))
        return trbdf2_heat

    def rk2_burger(k,alpha):
        def Fval(u):
            return alpha * (D@u)*u
        def rk2_step(u):
            u1 = u + (k/2)* Fval(u)
            return u + k * Fval(u1)
        return rk2_step
    
    half_heat = heat_step(k = k/2,kappa=kappa,n=n)
    burgers = rk2_burger(k = k,alpha = alpha)

    def get_ss_step(half_op1,op2):
        def strang_split_timestep(u):
            return half_op1(op2(half_op1(u)))
        return strang_split_timestep

    burger_split_step = get_ss_step(half_heat,burgers)

    def solve_burgers(u0,final_t = 1.):
        num_timesteps = final_t//k
        u_vals = [u0]
        u = u0
        for i in tqdm(range(int(num_timesteps))):
            u = burger_split_step(u)
            u_vals+=[u]
        return np.array(u_vals),np.arange(0,final_t,k)
    return xgrid,solve_burgers