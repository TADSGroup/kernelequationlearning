import jax 
import jax.numpy as jnp
from pde_solvers.BurgerSolver import get_burger_solver
import numpy as np
from scipy.interpolate import RectBivariateSpline

def build_tx_grid(t_range,x_range,num_grid_t,num_grid_x):
    full_t_grid = jnp.linspace(t_range[0],t_range[1],num_grid_t)
    full_x_grid = jnp.linspace(x_range[0],x_range[1],num_grid_x)
    x_interior = full_x_grid[1:-1]
    x_boundary = full_x_grid[jnp.array([0,-1])]

    #Get interior points
    t,x = jnp.meshgrid(full_t_grid,x_interior)
    tx_interior = jnp.vstack([t.flatten(),x.flatten()]).T

    #I'm doing this with meshgrid to handle more general higher dimensions, ie circular domain in 2d
    #Get boundary points
    t,x = jnp.meshgrid(full_t_grid,x_boundary)
    tx_boundary = jnp.vstack([t.flatten(),x.flatten()]).T
    return tx_interior,tx_boundary

def build_tx_grid_chebyshev(t_range,x_range,num_grid_t,num_grid_x,alpha = 0.5):
    full_t_grid = jnp.linspace(t_range[0],t_range[1],num_grid_t)
    nodes = jnp.cos(jnp.pi*(2*jnp.arange(0,num_grid_x,1)[::-1])/(2*(num_grid_x-1)))
    reg_grid = jnp.linspace(-1,1,num_grid_x)
    nodes = alpha * nodes + (1-alpha) * reg_grid
    full_x_grid = (nodes  + (x_range[0]+1))/(2*(x_range[1]-x_range[0]))
    x_interior = full_x_grid[1:-1]
    x_boundary = full_x_grid[jnp.array([0,-1])]

    #Get interior points
    t,x = jnp.meshgrid(full_t_grid,x_interior)
    tx_interior = jnp.vstack([t.flatten(),x.flatten()]).T

    #I'm doing this with meshgrid to handle more general higher dimensions, ie circular domain in 2d
    #Get boundary points
    t,x = jnp.meshgrid(full_t_grid,x_boundary)
    tx_boundary = jnp.vstack([t.flatten(),x.flatten()]).T
    return tx_interior,tx_boundary

def build_burgers_data(
    func_u0,
    kappa,
    alpha,
    k_timestep = 1e-4,
    n_finite_diff = 1999,
    final_time = 1.01
):
    grid,solver = get_burger_solver(alpha,kappa,k_timestep,n = n_finite_diff)
    u0 = func_u0(grid)[1:-1]
    sols,tvals = solver(u0,final_time)
    sols = np.hstack([np.zeros((len(sols),1)),sols,np.zeros((len(sols),1))])
    interp = RectBivariateSpline(tvals,grid,sols)

    def u_true_function(x):
        return interp(x[:,0],x[:,1],grid = False)

    def ut_true_function(x):
        ut_interp = interp.partial_derivative(1,0)
        return ut_interp(x[:,0],x[:,1],grid = False)
    return u_true_function,ut_true_function,interp,tvals,sols

def setup_problem_data(
    tx_int,
    tx_bdy,
    num_obs_to_sample,
    prng_key,
    tx_obs_to_include = None,
    times_to_observe = (0,),
):
    if tx_obs_to_include is None:
        included_obs = jnp.zeros((0,tx_int.shape[1]))
    else:
        included_obs = tx_obs_to_include
    time_full_obs_inds = jnp.where(jnp.isin(tx_int[:,0],jnp.array(times_to_observe)))[0]

    #All points are boundary, interior collocation, and observations
    tx_all = jnp.vstack([tx_bdy,tx_int,included_obs])

    remaining_inds = jnp.delete(jnp.arange(len(tx_int)),time_full_obs_inds)
    sampled_inds = jax.random.choice(prng_key,remaining_inds,(num_obs_to_sample,),replace = False)

    tx_obs = jnp.vstack([
        tx_bdy,
        tx_int[time_full_obs_inds],
        tx_int[sampled_inds],
        included_obs
    ])
    return tx_all,tx_obs
