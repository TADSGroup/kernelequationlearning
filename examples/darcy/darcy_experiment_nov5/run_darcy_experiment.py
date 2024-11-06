device_num = 3
num_functions = 50
samples_per_function = 10
collocation_grid_n = 16
a_matern_lengthscale = 0.3
a_exponent = 0.25
random_seed = 13

settings = {
    'num_functions':num_functions,
    'samples_per_functions':samples_per_function,
    'num_col_grid':collocation_grid_n,
    'a_matern_lengthscale':16,
    'a_exponent':0.25,
    'random_seed':13,
    'run_name':f"seed_{random_seed}_{num_functions}fun_{samples_per_function}obs"
}

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices()[device_num])

import jax.numpy as jnp
from KernelTools import make_block,eval_k,vectorize_kfunc,diagpart,get_selected_grad
from Kernels import get_gaussianRBF,get_centered_scaled_poly_kernel,get_matern
from data_utils import build_xy_grid
from darcy_data import get_darcy_solver,sample_gp_function
from jax.random import PRNGKey as pkey
import matplotlib.pyplot as plt
from EquationModel import CholInducedRKHS,SharedOperatorPDEModel,InducedOperatorModel,AltCholInducedRKHS
from tqdm.auto import tqdm
from parabolic_data_utils import build_alpha_chebyshev
from data_utils import make_grids
from pickle_save import save

obs_key,rhs_key,a_key = jax.random.split(pkey(random_seed))
# obs_key = pkey(32)
# rhs_key = pkey(10)
# a_key = pkey(124)

obs_random_keys = jax.random.split(obs_key,num_functions)
xy_obs = tuple(jax.random.uniform(key,(samples_per_function,2),minval = 0.,maxval = 1.) for key in obs_random_keys)


kernel_f = get_gaussianRBF(0.15)
rhs_keys = jax.random.split(rhs_key,num_functions)
rhs_functions = tuple(
    sample_gp_function(subkey,kernel_f) for subkey in rhs_keys
)

grid = jnp.linspace(0,1,100)
x,y = jnp.meshgrid(grid,grid)
fine_grid = jnp.vstack([x.flatten(),y.flatten()]).T
loga = sample_gp_function(a_key,get_matern(2,a_matern_lengthscale))
def a(x):
    return jnp.exp(a_exponent * loga(x))

# def a(x):
#     return 0.1 * jnp.exp(0.5 * jnp.sin(4*jnp.pi*x[0]*x[1]))

darcy_solve = get_darcy_solver(a,num_grid = 50,k_u = get_gaussianRBF(0.1))
u_true_functions = tuple([darcy_solve(f) for f in rhs_functions])

single_grid = build_alpha_chebyshev([0,1],collocation_grid_n,1.)
xy_int,xy_bdy = make_grids(single_grid,single_grid)
xy_all = jnp.vstack([xy_int,xy_bdy])

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title('a(x)')
plt.tricontourf(fine_grid[:,0],fine_grid[:,1],jax.vmap(a)(fine_grid),100)
plt.colorbar()

plt.subplot(1,2,2)
plt.title('u')
plt.tricontourf(fine_grid[:,0],fine_grid[:,1],jax.vmap(u_true_functions[0])(fine_grid),100)
plt.colorbar()
plt.scatter(xy_int[:,0],xy_int[:,1],c = 'black',s = 3)
plt.scatter(xy_obs[0][:,0],xy_obs[0][:,1],c = 'red',s = 50,alpha = 0.8)
# plt.scatter(xy_bdy[:,0],xy_bdy[:,1],c = 'red',s = 50,alpha = 0.8)
plt.xlim(-0.1,1.1)
plt.ylim(-0.1,1.1)
plt.savefig(settings['run_name']+'/example_a_u.png')
plt.close()



def diff_x_op(k,index):
    return get_selected_grad(k,index,0)

def diff_xx_op(k,index):
    return get_selected_grad(get_selected_grad(k,index,0),index,0)

def diff_y_op(k,index):
    return get_selected_grad(k,index,1)

def diff_yy_op(k,index):
    return get_selected_grad(get_selected_grad(k,index,1),index,1)

def diff_xy_op(k,index):
    return get_selected_grad(get_selected_grad(k,index,0),index,1)

feature_operators = tuple([eval_k,diff_x_op,diff_xx_op,diff_y_op,diff_yy_op,diff_xy_op])
basis_operators = tuple([eval_k,diff_xx_op,diff_yy_op])
basis_operators = feature_operators

k_u = get_gaussianRBF(0.25)
u_model = CholInducedRKHS(
    xy_all,
    basis_operators,
    k_u
)

observation_points = tuple(jnp.vstack([xy_bdy,obs]) for obs in xy_obs)
observation_values = tuple(jax.vmap(u)(obs_loc) for u,obs_loc in zip(u_true_functions,observation_points))
collocation_points = (xy_int,)*num_functions

rhs_values = tuple(jax.vmap(rhs_func)(int_points) for rhs_func,int_points in zip(rhs_functions,collocation_points))

all_u_params_init = tuple([
    u_model.get_fitted_params(obs_loc,obs_val) 
    for obs_loc,obs_val in zip(observation_points,observation_values)
    ]
    )

grid_features_u_init = [(
    u_model.evaluate_operators(feature_operators,xy_int,model_params)).reshape(
            len(xy_int),
            len(feature_operators),
            order = 'F'
        ) for model_params in all_u_params_init
        ]
grid_features_u_init = jnp.vstack([jnp.hstack([xy_int,features]) for features in grid_features_u_init])

num_P_inducing = 250
input_feature_sample = jax.random.choice(pkey(320),grid_features_u_init,(num_P_inducing,))
k_P_u_part = get_centered_scaled_poly_kernel(1,grid_features_u_init[:,2:],c=1)
k_P_x_part = get_matern(2,0.25)

def k_P(x,y):
    return k_P_x_part(x[:2],y[:2]) * k_P_u_part(x[2:],y[2:])

P_model = InducedOperatorModel(input_feature_sample,k_P)
P_init = P_model.get_fitted_params(grid_features_u_init,jnp.hstack(rhs_values),lam = 1e-4)

params_init = jnp.hstack(list(all_u_params_init)+[P_init])
u_init = jnp.stack(all_u_params_init)
EqnModel  = SharedOperatorPDEModel(
    P_model,
    u_model,
    observation_points,
    observation_values,
    collocation_points,
    feature_operators,
    rhs_values,
    datafit_weight = 10.,
    num_P_operator_params=num_P_inducing
)
beta_reg = 1e-12

from Optimizers import BlockArrowLM,LMParams
lm_params = LMParams(max_iter = 501,init_alpha = 1e-1,min_alpha = 1e-8)
u_sol,P_sol,arrow_conv = BlockArrowLM(
    u_init,P_init,EqnModel,beta_reg,beta_reg,
    optParams=lm_params
    )

grid = jnp.linspace(0.,1.,50)
x,y = jnp.meshgrid(grid,grid)
fine_grid_int = jnp.vstack([x.flatten(),y.flatten()]).T

def get_percent_errors(u_params):
    #Have to loop since u_true_functions are just callables
    percent_errors = []
    for i in range(num_functions):
        u_vals = u_model.point_evaluate(fine_grid_int,u_params[i])
        u_true_vals = jax.vmap(u_true_functions[i])(fine_grid_int)
        percent_errors.append(jnp.linalg.norm(u_vals - u_true_vals)/jnp.linalg.norm(u_true_vals))
    percent_errors = jnp.array(percent_errors)
    return percent_errors

onestep_errors = get_percent_errors(u_sol)
interp_errors = get_percent_errors(u_init)
print("Onestep ",jnp.mean(onestep_errors))
print("Interpolant ",jnp.mean(interp_errors))
print("Average Ratio: ",jnp.mean(interp_errors/onestep_errors))