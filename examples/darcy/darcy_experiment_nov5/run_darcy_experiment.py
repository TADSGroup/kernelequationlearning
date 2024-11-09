import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run the Python script with parameters')
parser.add_argument('--device_num', type=int, default=3, help='Device number for jax')
parser.add_argument('--num_functions', type=int, default=50, help='Number of functions')
parser.add_argument('--samples_per_function', type=int, default=10, help='Samples per function')
parser.add_argument('--collocation_grid_n', type=int, default=16, help='Number of collocation grid points')
parser.add_argument('--a_matern_lengthscale', type=float, default=0.3, help='Matern lengthscale parameter')
parser.add_argument('--a_exponent', type=float, default=0.25, help='Exponent parameter')
parser.add_argument('--random_seed', type=int, default=13, help='Random seed')
parser.add_argument('--experiment_extra_name', type=str, default='', help='Extra name data')


args = parser.parse_args()

# Assign parameters
device_num = args.device_num
num_functions = args.num_functions
samples_per_function = args.samples_per_function
collocation_grid_n = args.collocation_grid_n
a_matern_lengthscale = args.a_matern_lengthscale
a_exponent = args.a_exponent
random_seed = args.random_seed
extra_name = args.experiment_extra_name

# Settings dictionary
settings = {
    'num_functions': num_functions,
    'samples_per_functions': samples_per_function,
    'num_col_grid': collocation_grid_n,
    'a_matern_lengthscale': a_matern_lengthscale,
    'a_exponent': a_exponent,
    'random_seed': random_seed,
    'run_name': f"{extra_name}_seed{random_seed}_{num_functions}fun_{samples_per_function}obs"
}

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices()[device_num])

import jax.numpy as jnp
from KernelTools import eval_k,vectorize_kfunc,get_selected_grad
from Kernels import get_gaussianRBF,get_centered_scaled_poly_kernel,get_matern
from darcy_data import get_darcy_solver,sample_gp_function
from jax.random import PRNGKey as pkey
import matplotlib.pyplot as plt
from EquationModel import CholInducedRKHS,SharedOperatorPDEModel,InducedOperatorModel
from tqdm.auto import tqdm
from parabolic_data_utils import build_alpha_chebyshev
from data_utils import make_grids
from pickle_save import save
import time

start_time = time.time()

save(settings,settings['run_name'] + "/settings")


obs_key,rhs_key,a_key = jax.random.split(pkey(random_seed),3)
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

result_dict = {
    '1step_u_params':u_sol,
    '1step_P_params':P_sol,
    '2step_u_params':u_init,
    '2step_P_params':P_sol
               }

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
result_dict['u_error_onestep'] = onestep_errors
result_dict['u_error_interp'] = interp_errors

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(15, 4 * num_functions))
gs = gridspec.GridSpec(num_functions, 4, width_ratios=[1, 1, 1, 0.05])  # Last column for colorbar

for i in range(num_functions):
    # Get the subplots for this row
    ax1 = plt.subplot(gs[i, 0])
    ax2 = plt.subplot(gs[i, 1])
    ax3 = plt.subplot(gs[i, 2])
    cbar_ax = plt.subplot(gs[i, 3])  # Colorbar axis

    # Compute the values
    u_vals_1step = u_model.point_evaluate(fine_grid_int, u_sol[i])
    u_vals_interp = u_model.point_evaluate(fine_grid_int, u_init[i])
    u_true_vals = jax.vmap(u_true_functions[i])(fine_grid_int)

    # Determine the levels for consistent color mapping across subplots in the same row
    lower = jnp.min(jnp.vstack([u_vals_1step, u_vals_interp, u_true_vals]))
    upper = jnp.max(jnp.vstack([u_vals_1step, u_vals_interp, u_true_vals]))
    levels = jnp.linspace(lower, upper + 1e-4, 200)

    # First subplot: Truth
    ax1.set_title("Truth")
    contour_truth = ax1.tricontourf(fine_grid_int[:, 0], fine_grid_int[:, 1], u_true_vals, levels=levels)
    ax1.scatter(xy_obs[i][:, 0], xy_obs[i][:, 1], c='black', s=25)
    ax1.scatter(xy_bdy[:, 0], xy_bdy[:, 1], c='red', s=25)

    # Second subplot: 1 Step Solution
    ax2.set_title(f"1 Step, {100 * onestep_errors[i]:.2f}% Error")
    contour_1step = ax2.tricontourf(fine_grid_int[:, 0], fine_grid_int[:, 1], u_vals_1step, levels=levels)
    ax2.scatter(xy_obs[i][:, 0], xy_obs[i][:, 1], c='black', s=25)
    ax2.scatter(xy_bdy[:, 0], xy_bdy[:, 1], c='red', s=25)

    # Third subplot: Basic Kernel Interpolant
    ax3.set_title(f"Basic Kernel Interpolant, {100 * interp_errors[i]:.2f}% Error")
    contour_interp = ax3.tricontourf(fine_grid_int[:, 0], fine_grid_int[:, 1], u_vals_interp, levels=levels)
    ax3.scatter(xy_obs[i][:, 0], xy_obs[i][:, 1], c='black', s=25)
    ax3.scatter(xy_bdy[:, 0], xy_bdy[:, 1], c='red', s=25)

    # Add colorbar for this row
    cbar = fig.colorbar(contour_interp, cax=cbar_ax)
    cbar_ax.set_ylabel('Intensity')  # Label for colorbar

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig(settings['run_name']+'/u_results.png')
plt.close()

import pde_solvers.kernel_elliptic
import Optimizers.solvers_base
from importlib import reload
reload(Optimizers.solvers_base)
reload(pde_solvers.kernel_elliptic)
from pde_solvers.kernel_elliptic import EllipticPDEModel

evaluation_key = pkey(10)
num_evaluation_functions = 10
kernel_f = get_gaussianRBF(0.15)
keys = jax.random.split(evaluation_key,num_evaluation_functions)

rhs_functions_eval = tuple(
    sample_gp_function(subkey,kernel_f) for subkey in keys
)
darcy_solve = get_darcy_solver(a,num_grid = 50,k_u = get_gaussianRBF(0.2))
u_true_fine = tuple([jax.vmap(darcy_solve(f))(fine_grid) for f in tqdm(rhs_functions_eval)])

k_u = get_gaussianRBF(0.2)
solver_params = LMParams(max_iter = 501,init_alpha = 1e-4,min_alpha = 1e-8,use_jit = False,show_progress=False)


onestep_P = lambda x:P_model.predict(x,P_sol)
onestep_solver = EllipticPDEModel(
    get_gaussianRBF(0.2),onestep_P,
    feature_operators,num_grid = 30,solverParams=solver_params
    )

twostep_P = lambda x:P_model.predict(x,P_init)
twostep_solver = EllipticPDEModel(
    get_gaussianRBF(0.2),twostep_P,
    feature_operators,num_grid = 30,
    solverParams=solver_params
    )

def solve_evaluate(solver,rhs_func,grid):
    pde_u,pde_params,conv = solver.solve(rhs_func)
    return pde_u.point_evaluate(grid,pde_params)

onestep_solutions = [solve_evaluate(onestep_solver,rhs_f,fine_grid) for rhs_f in rhs_functions_eval]
twostep_solutions = [solve_evaluate(twostep_solver,rhs_f,fine_grid) for rhs_f in rhs_functions_eval]

onestep_operator_errors = jnp.array([jnp.linalg.norm(uu - u_one)/jnp.linalg.norm(uu) for uu,u_one in zip(u_true_fine,onestep_solutions)])
twostep_operator_errors = jnp.array([jnp.linalg.norm(uu - u_two)/jnp.linalg.norm(uu) for uu,u_two in zip(u_true_fine,twostep_solutions)])

result_dict['1step_op_errors']=onestep_operator_errors
result_dict['2step_op_errors']=twostep_operator_errors


onestep_forward_errors = []
twostep_forward_errors = []

for i in range(num_evaluation_functions):
    sol = darcy_solve(rhs_functions_eval[i])
    rhs_values = jax.vmap(rhs_functions_eval[i])(fine_grid_int)
    true_input_features = jnp.vstack([fine_grid_int.T] + [jax.vmap(op(sol,0))(fine_grid_int) for op in feature_operators]).T
    onestep_forward_errors.append(jnp.linalg.norm(rhs_values - onestep_P(true_input_features))/jnp.linalg.norm(rhs_values))
    twostep_forward_errors.append(jnp.linalg.norm(rhs_values - twostep_P(true_input_features))/jnp.linalg.norm(rhs_values))

onestep_forward_errors = jnp.array(onestep_forward_errors)
twostep_forward_errors = jnp.array(twostep_forward_errors)

result_dict['1step_forward_errors']=onestep_forward_errors
result_dict['2step_forward_errors']=twostep_forward_errors

save(result_dict,settings['run_name'] + '/error_results')

import pandas as pd
result_summary_dict = {
    "mean u rmse across in sample functions":{
        'one step':jnp.mean(onestep_errors),
        'two step':jnp.mean(interp_errors),
    },
    'operator rmse OOS':{
        'one step':jnp.mean(onestep_operator_errors),
        'two step':jnp.mean(twostep_operator_errors),
    },
    'P forward rmse OOS':{
        'one step':jnp.mean(onestep_forward_errors),
        'two step':jnp.mean(twostep_forward_errors),
    },
    "total experiment time":{
        'one step':time.time() - start_time,
        'two step':time.time() - start_time,
    }
}
pd.DataFrame(result_summary_dict).to_csv(settings['run_name'] + '/error_summary.csv')


