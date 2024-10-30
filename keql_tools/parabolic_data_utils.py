import jax 
import jax.numpy as jnp
from pde_solvers.BurgerSolver import get_burger_solver,get_burger_solver_periodic
from KernelTools import vectorize_kfunc
import numpy as np
from scipy.interpolate import RectBivariateSpline


def GP_Sampler_1D_Pinned(num_samples, X, smooth, kernel, reg, seed): 
    """
        Gets samples(functions) of GP whose values are zero at {0,1}. 

        Args:
            num_samples (int): Number of samples.
            X (jnp.array): Domain to get the sample.
            smooth (float): Regularity level, the bigger the smoother. 
            kernel (fun): Kernel function from Kernels.py.
            reg (float): Regularization to invert kernel matrix.
            seed (float): Integer to fix the simulation.

        Returns:
            list: Returns the list of functions sampled from the GP.   


        Example:
            >>> # Create fine grid
            >>> X = np.linspace(0,1,100)
            >>> kernel = get_gaussianRBF(0.2)
            >>> u1, u2, u3 = GP_Sampler_1D_Pinned(num_samples = 3,
                                        X = pairs, 
                                        smooth = 5,
                                        kernel = kernel, 
                                        reg = 1e-12,
                                        seed = 2025
                                    )

    """ 
    N = len(X)
    pred = []
    key = jax.random.key(seed)
    for _ in range(num_samples):
        key, subkey = jax.random.split(key)
        phis = jax.random.normal(key = subkey, shape=(N,))
        lmbds = np.array([1/(j**smooth) for j in range(1,N+1)])
        coefs = phis*lmbds
        sine_matrix = np.array([np.sin(j*np.pi*X) for j in range(N)])
        y_pred_test = sine_matrix.T @ coefs
        pred.append(y_pred_test)    
    sample = jnp.array(pred)

    # Build interpolants
    k = vectorize_kfunc(kernel)
    kernel_matrix = k(X,X) + reg*jnp.eye(N)

    def get_interpolant(values):
      coeffs = jnp.linalg.solve(kernel_matrix,values)
      k_vec = jax.vmap(kernel,in_axes=(0,None))

      def interp(x):
        return jnp.dot(k_vec(X,x),coeffs)
      return interp
    
    interps = [get_interpolant(S) for S in sample]

    return interps

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

def build_alpha_chebyshev(x_range,num_grid,alpha):
    nodes = jnp.cos(jnp.pi*(2*jnp.arange(0,num_grid,1)[::-1])/(2*(num_grid-1)))
    reg_grid = jnp.linspace(-1,1,num_grid)
    nodes = alpha * nodes + (1-alpha) * reg_grid
    full_x_grid = (nodes  + (x_range[0]+1))/(2*(x_range[1]-x_range[0]))
    return full_x_grid

def build_tx_grid_chebyshev(t_range,x_range,num_grid_t,num_grid_x,alpha = 0.5):
    full_t_grid = jnp.linspace(t_range[0],t_range[1],num_grid_t)
    full_x_grid = build_alpha_chebyshev(x_range,num_grid_x,alpha)
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
    #TODO: move the appending of boundary conditions into what the solver itself returns 
    #So that it's congruent with the grid
    sols = np.hstack([np.zeros((len(sols),1)),sols,np.zeros((len(sols),1))])
    interp = RectBivariateSpline(tvals,grid,sols)

    def u_true_function(x):
        return interp(x[:,0],x[:,1],grid = False)

    def ut_true_function(x):
        ut_interp = interp.partial_derivative(1,0)
        return ut_interp(x[:,0],x[:,1],grid = False)
    return u_true_function,ut_true_function,interp,tvals,sols

def build_burgers_data_periodic(
    func_u0,
    kappa,
    alpha,
    k_timestep = 1e-4,
    n_finite_diff = 1999,
    final_time = 1.01
):
    grid,solver = get_burger_solver_periodic(alpha,kappa,k_timestep,n = n_finite_diff)
    u0 = func_u0(grid)[:-1]
    sols,tvals = solver(u0,final_time)
    #TODO: Do this adjustment inside the solver
    sols = np.hstack([sols,sols[:,:1]])
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


def build_u_obs_single_burgers(num_obs, tx_int, tx_bdy,vmapped_u_true_function,key):
    """
        Samples points out of those being modeled as observations, and compute u_true those points
            Include xy_bdy as observations for all
        Computes rhs at all interior points being modeled

        Args:
            num_obs (int): Number of observed points on the function.
            xy_int (jnp.array): Interior domain to get the sample.
            xy_bdy (jnp.array): Boundary domain.
            vmapped_u_true_function (callable): Vmapped u true functions.
            vmapped_rhs (callable): Vmapped f true functions.
            seed (int): Integer to fix the simulation.

        Returns:
            list: List of lists xy_obs, u_obs, f_obs.   
    """

    xy_sample_inds = jax.random.choice(key,jnp.arange(len(tx_int)),(num_obs,),replace = False)
    xy_obs = jnp.vstack([tx_bdy,tx_int[xy_sample_inds]])
    u_obs = vmapped_u_true_function(xy_obs)
    return xy_obs, u_obs
