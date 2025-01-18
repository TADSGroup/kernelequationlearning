# imports
import jax
jax.config.update("jax_default_device",jax.devices()[2])
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit,grad,jacfwd,jacrev,vmap
from jax.random import PRNGKey as pkey
from jax.scipy.linalg import solve
# Other libraries
import numpy as np
import matplotlib.pyplot as plt
from labellines import labelLines
from matplotlib.lines import Line2D
plt.style.use('default')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    'font.size': 20
})
from tqdm.auto import tqdm
# plt.style.use("ggplot")
from importlib import reload

# Our libraries
import KernelTools
reload(KernelTools)
from KernelTools import *
from EquationModel import OperatorModel, OperatorPDEModel,CholInducedRKHS,InducedOperatorModel
from plotting import plot_obs,plot_compare_error
from evaluation_metrics import get_nrmse,table_u_errors
from data_utils import (
    get_xy_grid_pairs,
    GP_sampler,
    build_xy_grid,
    sample_xy_grid_latin,
    build_input_observations,
    build_u_obs_single,
    build_u_obs_all,
    sample_multiple_xy_grids_latin
)

from Kernels import (
    get_gaussianRBF,
    get_centered_scaled_poly_kernel
)

import Optimizers
import importlib
importlib.reload(Optimizers)
from Optimizers import CholeskyLM,SVD_LM
from Optimizers.solvers_base import *


# Out-of-distribution error
def run_exp_o_dis_err(m,obs_pts,run):
    '''
    Computes out-of-distribution error for 1 step and 2 step methods.

    Args:
        m (int): Number of functions.
        obs_pts (int): Number of observed points.
        run (int): seed.
    
    Returns:
        i_opt_1_5 (float): Error for Phat (1 step) for in-sample functions.
        i_opt_2 (float): Error for Phat (2 step) for in-sample functions.

    '''
    # Sample m training functions from a GP
    kernel_GP = get_gaussianRBF(0.5)
    xy_pairs = get_xy_grid_pairs(50,0,1,0,1) # Pairs to build interpolants
    u_true_functions = tuple(GP_sampler(num_samples = m,
                                            X = xy_pairs, 
                                            kernel = kernel_GP,
                                            reg = 1e-12,
                                            seed = 2024
                                        )
                                        )
    # Permeability field A
    def A(xy):
        x = xy[0]
        y = xy[1]
        return jnp.exp(jnp.sin(jnp.cos(x) + jnp.cos(y)))

    # Compute f = Pu for a given u
    def get_rhs_darcy(u):
        def Agradu(xy):
            return A(xy)*jax.grad(u)(xy)
        def Pu(xy):
            return jnp.trace(jax.jacfwd(Agradu)(xy))
        return Pu

    # Lists of m true u's and f's
    vmapped_u_true_functions = tuple([jax.vmap(u) for u in u_true_functions]) # vmap'ed
    rhs_functions = tuple([jax.vmap(get_rhs_darcy(u)) for u in u_true_functions]) #vmap'ed

    # Define the num of ghost points for each u
    num_grid_points = 10
    num_interior_points = 50

    # # Sample collocation points for f using random points different for every function
    # xy_ints,xy_bdys = sample_multiple_xy_grids_latin(
    #         num_functions = m,
    #         num_interior = num_interior_points,
    #         x_range = [0,1],
    #         y_range = [0,1],
    #         num_grid_x_bdy = num_grid_points,
    #         num_grid_y_bdy = num_grid_points,
    #         key = pkey(23)
    #     )
    
    # Sample collocation points for f using same uniform grid for every function
    xy_ints = tuple(build_xy_grid([0,1],[0,1],7,7)[0] for m in range(m))
    xy_bdys = tuple(build_xy_grid([0,1],[0,1],7,7)[1] for m in range(m))

    xy_all = tuple(jnp.vstack([xy_int,xy_bdy]) for xy_int,xy_bdy in zip(xy_ints,xy_bdys))

    # List of number of observation points per u
    num_obs = [obs_pts]*m

    # Get (X^m, u^m(X^m))
    xy_obs,u_obs = build_u_obs_all(
        num_obs,
        xy_ints,
        xy_bdys,
        vmapped_u_true_functions,
        pkey(5)
    )

    # Build operator features
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

    # Build interpolants for u's
    k_u = get_gaussianRBF(0.5)
    u_operators = (eval_k,)
    u_models = tuple([CholInducedRKHS(
        xy_all[i],
        u_operators,
        k_u
        ) for i in range(m)])
    
    # Get necessary tuples
    observation_points = tuple(xy_obs)
    observation_values = tuple(u_obs)
    collocation_points = xy_ints

    rhs_values = tuple(rhs_func(xy_int) for xy_int,rhs_func in zip(xy_ints,rhs_functions))

    all_u_params_init = tuple([
        model.get_fitted_params(obs_loc,obs_val)
        for obs_loc,obs_val,model in zip(observation_points,observation_values,u_models)])

    grid_features_u_init = jnp.vstack([(
        model.evaluate_operators(feature_operators,xy_int,model_params)).reshape(
                len(xy_int),
                len(feature_operators),
                order = 'F'
            ) for xy_int,model,model_params in zip(xy_ints,u_models,all_u_params_init) ])

    grid_features_u_init = jnp.hstack([jnp.vstack(xy_ints),grid_features_u_init])

    # P kernel
    k_P_u_part = get_centered_scaled_poly_kernel(1,grid_features_u_init[:,2:],c=1)
    k_P_x_part = get_gaussianRBF(0.4)
    
    def k_P(x,y):
        return k_P_x_part(x[:2],y[:2]) * k_P_u_part(x[2:],y[2:])

    # P object        
    P_model = InducedOperatorModel(grid_features_u_init,k_P)
    num_P_params = len(grid_features_u_init)

    # P, u, f object
    collocation_points = xy_ints
    EqnModel  = OperatorPDEModel(
        P_model,
        u_models,
        observation_points,
        observation_values,
        collocation_points,
        feature_operators,
        rhs_values,
        datafit_weight = 5.,
        num_P_operator_params = num_P_params
    )

    ### Optimize LM

    # Initialize
    # rhs_values = tuple(rhs_func(int_points) for rhs_func,int_points in zip(rhs_functions,collocation_points))
    
    # P_init = P_model.get_fitted_params(grid_features_u_init,jnp.hstack(rhs_values))
    # params_init = jnp.hstack(list(all_u_params_init)+[P_init])
    params_init = jnp.hstack(list(all_u_params_init)+[jnp.zeros(m*len(xy_ints[0]))])

    # Optimizer hyperparameters
    optparams = LMParams(max_iter = 301,
                         line_search_increase_ratio = 1.4,
                         print_every = 100,
                         tol = 1e-10)
    
    # Run CholeskyLM
    params,convergence_data = CholeskyLM(
        params_init.copy(),
        EqnModel,
        beta = 1e-8,
        optParams = optparams
    )

    # p_adjusted,refine_convergence_data = SVD_LM(
    #     params,
    #     EqnModel,
    #     beta = 1e-13,
    #     optParams = optparams
    # )

    # Optimized parameters
    u_sols = EqnModel.get_u_params(params)
    P_sol = EqnModel.get_P_params(params)

    ## Errors

    # Testing grid
    num_fine_grid = 50
    x_fine,y_fine = np.meshgrid(np.linspace(0,1,num_fine_grid+4)[2:-2],np.linspace(0,1,num_fine_grid+4)[2:-2])
    xy_fine_int = np.vstack([x_fine.flatten(),y_fine.flatten()]).T

    # Estimated P from 1.5 step method
    model_grid_features_all =jnp.vstack([EqnModel.single_eqn_features(u_model,u_params,eval_points) 
                                          for u_model,u_params,eval_points in zip(
                                            EqnModel.u_models,
                                            u_sols,
                                            EqnModel.collocation_points)])    
    S_train = model_grid_features_all
    P_func = lambda x: P_model.predict(x,P_sol)

    # Estimated P from 2 step method
    init_P_features = jnp.vstack([EqnModel.single_eqn_features(u_model,u_params,eval_points) 
                                          for u_model,u_params,eval_points in zip(
                                            EqnModel.u_models,
                                            all_u_params_init,
                                            EqnModel.collocation_points)])
    rhs_stacked = EqnModel.stacked_collocation_rhs
    P_params_naive = P_model.get_fitted_params(init_P_features,rhs_stacked)
    P_func2 = lambda x: P_model.predict(x,P_params_naive)

    # P[\phi(w)](fine_grid)
    def evaluate_hatP(P_func, w, fine_grid, feature_operators):

        # Build S_test
        w_features = jnp.array([jax.vmap(operator(w,0))(xy_fine_int) for operator in feature_operators]).T
        model_fine_features = jnp.hstack([fine_grid, w_features])
        S_test = model_fine_features


        #P_preds_model_features = P_model.kernel_function(S_test,S_train)@P_sol 
        P_preds = P_func(S_test)
        return P_preds
    
    # Out of distribution
    M = 3

    kernel_GP = get_gaussianRBF(0.2) # Rougher than the functions we train Phat
    # Sample M test functions from GP(0,K)
    w_test_functions = GP_sampler(num_samples = M,
                    X = xy_pairs, 
                    kernel = kernel_GP,
                    reg = 1e-12,
                    seed = run
                    )
    vmapped_w_test_functions = tuple([jax.vmap(w) for w in w_test_functions]) # vmap'ed
    w_rhs_functions = tuple([jax.vmap(get_rhs_darcy(w)) for w in w_test_functions]) #vmap'ed

    # mean 
    true = [f_w(xy_fine_int) for f_w in w_rhs_functions]
    #pred = [evaluate_hatP(w, xy_fine_int, u_sols, P_sol,feature_operators) for w in w_test_functions]
    # pred1_5 = [
    #     evaluate_hatP(
    #     lambda x:P_model.kernel_function(x,S_train)@P_sol,
    #     w, xy_fine_int,feature_operators) for w in w_test_functions
    # ]
    pred1_5 = [
        evaluate_hatP(
        P_func,
        w, xy_fine_int,feature_operators) for w in w_test_functions
    ]

    pred2 = [
        evaluate_hatP(
        P_func2,
        u, xy_fine_int,feature_operators) for u in w_test_functions
    ]

    o_dis_1_5 = jnp.mean(jnp.array([get_nrmse(t,p) for t,p in zip(true,pred1_5)]))
    o_dis_2 = jnp.mean(jnp.array([get_nrmse(t,p) for t,p in zip(true,pred2)]))

    return o_dis_1_5, o_dis_2

# Dictionary to store results
err = {
    '1_5_mthd': {
        '2_obs': {'o_dis': []},
        '4_obs': {'o_dis': []},
        '6_obs': {'o_dis': []},
        '8_obs': {'o_dis': []},
        '10_obs': {'o_dis': []}
                  },
    '2_mthd':   {
        '2_obs': {'o_dis': []},
        '4_obs': {'o_dis': []},
        '6_obs': {'o_dis': []},
        '8_obs': {'o_dis': []},
        '10_obs': {'o_dis': []}
                }
}

# Run main loop
NUM_FUN_LIST = [2,4,6,8]
NUM_RUNS = 10
OBS_PTS_LIST = [2,4,6,8,10]
for obs_pt in OBS_PTS_LIST:
    for m in NUM_FUN_LIST:
        o_dis_1_5 = []
        o_dis_2 = []
        for run in range(NUM_RUNS):
            # Run
            res = run_exp_o_dis_err(m, obs_pt, run)
            # Append
            o_dis_1_5.append(res[0])
            o_dis_2.append(res[1])
        # Append each list    
        err['1_5_mthd'][f'{obs_pt}_obs']['o_dis'].append(o_dis_1_5)
        err['2_mthd'][f'{obs_pt}_obs']['o_dis'].append(o_dis_2)   
    # Save after
    jnp.save('errors', err)
print('sucess!')
