import jax 
import jax.numpy as jnp
from KernelTools import vectorize_kfunc
from scipy.stats import qmc
import numpy as np

def get_xy_grid_pairs(n = 25,x_lower = 0,x_upper =1,y_lower = 0,y_upper = 1):
    # Define grid to get samples from GP
    x_grid=jnp.linspace(x_lower,x_upper,n)
    y_grid=jnp.linspace(y_lower,y_upper,n)
    X,Y=jnp.meshgrid(x_grid,y_grid)
    pairs = jnp.vstack([X.ravel(), Y.ravel()]).T
    return pairs

def GP_sampler(num_samples, X, kernel, reg, seed):   
    """
        Gets samples(functions) of GP. 

        Args:
            num_samples (int): Number of samples.
            X (jnp.array): Domain to get the sample.
            kernel (fun): Kernel function from Kernels.py.
            len_scale (float): Scale to be used in kernel.
            reg (float): Regularization to invert kernel matrix.
            seed (float): Integer to fix the simulation.

        Returns:
            list: Returns the list of functions sampled from the GP.   


        Example:
            >>> # Create fine grid
            >>> x = jnp.linspace(0,1,40)
            >>> kernel_GP = get_gaussianRBF(0.5)
            >>> y = x
            >>> xv, yv = jnp.meshgrid(x, y)
            >>> pairs = jnp.vstack([xv.ravel(), yv.ravel()]).T
            >>> u1, u2, u3 = GPsampler2D(num_samples = 3,
                                        X = pairs, 
                                        kernel = kernel_GP, 
                                        reg = 1e-12,
                                        seed = 2025
                                    )
            >>> u1(jnp.array([0.1,0.4]))
            Array(-0.30628616, dtype=float64)
            >>> jax.vmap(u1)(pairs)
            Array([-0.81486278, -0.69548977, -0.39381869, ...,  0.79765953,
                    0.76227927,  0.51719524], dtype=float64)

    """
    N = len(X)
    k = vectorize_kfunc(kernel)
    # Build kernel matrix
    kernel_matrix = k(X,X) + reg*jnp.eye(N)
    chol_factor = jnp.linalg.cholesky(kernel_matrix)
    PRNG_key = jax.random.PRNGKey(seed)
    # Sample standard normal
    normal_samples = jax.random.normal(key = PRNG_key, shape=(N,num_samples))
    # Compute sample
    sample = jnp.dot(chol_factor,normal_samples)    

    # Build interpolants
    def get_interpolant(pairs,values):
      coeffs = jnp.linalg.solve(kernel_matrix,values)
      k_vec = jax.vmap(kernel,in_axes=(0,None))

      def interp(x):
        return jnp.dot(k_vec(X,x),coeffs)
      return interp
    
    interps = [get_interpolant(X,S) for S in sample.T]
    
    return interps

# Build interior and boundary grids 
def build_xy_grid(x_range,y_range,num_grid_x,num_grid_y):
    full_x_grid = jnp.linspace(x_range[0],x_range[1],num_grid_x)
    full_y_grid = jnp.linspace(y_range[0],y_range[1],num_grid_y)

    x_interior = full_x_grid[1:-1]
    y_interior = full_y_grid[1:-1]
    
    x_boundary = full_x_grid[jnp.array([0,-1])]
    y_boundary = full_y_grid[jnp.array([0,-1])]

    #Get interior points
    x,y = jnp.meshgrid(x_interior,y_interior)
    xy_interior = jnp.vstack([x.flatten(),y.flatten()]).T

    #I'm doing this with meshgrid to handle more general higher dimensions, ie circular domain in 2d
    #Get boundary points
    x,y = jnp.meshgrid(x_interior,y_boundary)
    xy_boundary_1 = jnp.vstack([x.flatten(),y.flatten()]).T

    x,y = jnp.meshgrid(x_boundary,y_interior)
    xy_boundary_2 = jnp.vstack([x.flatten(),y.flatten()]).T

    x,y = jnp.meshgrid(x_boundary,y_boundary)
    xy_boundary_3 = jnp.vstack([x.flatten(),y.flatten()]).T

    xy_boundary = jnp.vstack([xy_boundary_1,xy_boundary_2,xy_boundary_3])
    return xy_interior,xy_boundary

# Build interior and boundary grids
def sample_xy_grid_latin(
    num_interior,
    x_range,
    y_range,
    num_grid_x_bdy,
    num_grid_y_bdy,
    seed = None
    ):
    """
    Samples interior points with latin hypercube sampling,
    Gives boundary points with explicit grid
    """

    #Build grid for boundary
    full_x_grid = jnp.linspace(x_range[0],x_range[1],num_grid_x_bdy)
    full_y_grid = jnp.linspace(y_range[0],y_range[1],num_grid_y_bdy)

    x_interior = full_x_grid[1:-1]
    y_interior = full_y_grid[1:-1]

    x_boundary = full_x_grid[jnp.array([0,-1])]
    y_boundary = full_y_grid[jnp.array([0,-1])]

    #Get interior points
    x,y = jnp.meshgrid(x_interior,y_interior)
    xy_interior = jnp.vstack([x.flatten(),y.flatten()]).T

    #I'm doing this with meshgrid to handle more general higher dimensions, ie circular domain in 2d
    #Get boundary points
    x,y = jnp.meshgrid(x_interior,y_boundary)
    xy_boundary_1 = jnp.vstack([x.flatten(),y.flatten()]).T

    x,y = jnp.meshgrid(x_boundary,y_interior)
    xy_boundary_2 = jnp.vstack([x.flatten(),y.flatten()]).T

    x,y = jnp.meshgrid(x_boundary,y_boundary)
    xy_boundary_3 = jnp.vstack([x.flatten(),y.flatten()]).T
    xy_boundary = jnp.vstack([xy_boundary_1,xy_boundary_2,xy_boundary_3])

    # Use latic hypercube sampling for interior
    sampler = qmc.LatinHypercube(d=2,seed = seed)
    sample = sampler.random(num_interior)
    interior_sample = qmc.scale(
       sample,[x_range[0],y_range[0]],[x_range[1],y_range[1]]
       )

    return interior_sample,xy_boundary

def sample_multiple_xy_grids_latin(
    num_functions,
    num_interior,
    x_range,
    y_range,
    num_grid_x_bdy,
    num_grid_y_bdy,
    key,
):
    scipy_seeds = jax.random.randint(
        key,(num_functions,),0,10000
    )
    interior_obs,xy_boundaries = zip(*[
       sample_xy_grid_latin(num_interior,x_range,y_range,num_grid_x_bdy,num_grid_y_bdy,seed.item())
       for seed in scipy_seeds])
    return interior_obs,xy_boundaries

#TODO: Use jax random functions instead of numpy
def build_input_observations(num_obs, xy_int, xy_bdy,vmapped_u_true_functions, vmapped_rhs, seed):
    """
        Samples points out of those being modeled as observations, and compute u_true those points
            Include xy_bdy as observations for all
        Computes rhs at all interior points being modeled

        Args:
            num_obs (list): List of integers of number of observed points per function.
            xy_int (jnp.array): Interior domain to get the sample.
            xy_bdy (jnp.array): Boundary domain.
            vmapped_u_true_functions (list): List of vmapped u true functions.
            vmapped_rhs (list): List of vmapped f true functions.
            seed (int): Integer to fix the simulation.

        Returns:
            list: List of lists xy_obs, u_obs, f_obs.   
    """

    if len(num_obs) != len(vmapped_u_true_functions):
        raise Exception("len of list of provided observed points not same as len of list of provided functions.")
    
    num_f = len(vmapped_rhs)
    key = jax.random.PRNGKey(seed)
    seed = jax.random.randint(key=key, shape=(num_f,), minval=1, maxval=40)
    xy_obs, u_obs, f_obs =  [], [], []
    for i, num_ob in enumerate(num_obs):
        np.random.seed(seed[i])
        xy_sample_indsi = np.random.choice(list(np.arange(len(xy_int))),num_ob,replace = False)
        xy_obsi = jnp.vstack([xy_bdy,xy_int[xy_sample_indsi]])
        xy_obs.append(xy_obsi)
        u_obs.append(vmapped_u_true_functions[i](xy_obsi))
        f_obs.append(vmapped_rhs[i](xy_obsi))
    return xy_obs, u_obs, f_obs


def build_u_obs_single(num_obs, xy_int, xy_bdy,vmapped_u_true_function,key):
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

    xy_sample_inds = jax.random.choice(key,jnp.arange(len(xy_int)),(num_obs,),replace = False)
    xy_obs = jnp.vstack([xy_bdy,xy_int[xy_sample_inds]])
    u_obs = vmapped_u_true_function(xy_obs)
    return xy_obs, u_obs

def build_u_obs_all(
    observation_counts,
    xy_ints,
    xy_bdys,
    vec_u_true_funcs,
    key
):
    split_keys = jax.random.split(key,len(xy_ints))
    return zip(*[
        build_u_obs_single(*args) 
        for args in zip(
            observation_counts,xy_ints,xy_bdys,vec_u_true_funcs,split_keys
            )
        ])


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit_transform(self, X):
        X_min = jnp.min(X, axis=0)
        X_max = jnp.max(X, axis=0)
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (X_max - X_min)
        self.min_ = self.feature_range[0] - X_min * self.scale_
        return self.transform(X)

    def transform(self, X):
        return X * self.scale_ + self.min_

    def inverse_transform(self, X_scaled):
        return (X_scaled - self.min_) / self.scale_