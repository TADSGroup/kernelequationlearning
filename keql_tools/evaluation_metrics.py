import jax.numpy as jnp
import pandas as pd

def get_nrmse(true,pred):
    '''
        Computes normalized root squared mse
    '''
    return jnp.sqrt(jnp.mean((true-pred)**2)/jnp.mean(true**2))

def table_u_errors(xy_fine, u_models, u_sols, vmapped_u_true_functions, all_u_params_init):
    '''
        Draw pandas df containing nrmse between pred u's and true u's
    '''
    # Create empty dict to store errors
    results_dict = dict()
    for i in range(len(vmapped_u_true_functions)):
        # Compute u pred from 1.5 step method at xy_fine
        u_eval_fine = u_models[i].point_evaluate(xy_fine,u_sols[i])
        # Compute u pred from 2 step(init) method at xy_fine
        u_eval_fine_init = u_models[i].point_evaluate(xy_fine,all_u_params_init[i])
        # Compute u true at xy_fine
        u_true_fine = vmapped_u_true_functions[i](xy_fine)
        # Compute rnmse's and append to dict
        results_dict[i]=[float(get_nrmse(u_true_fine,u_eval_fine)),float(get_nrmse(u_true_fine,u_eval_fine_init))]
    # Create pandas df from dict
    result_df_u = pd.DataFrame.from_dict(results_dict,orient = 'index',columns = ['1 step','2 step (at initialization)'])
    # Display the table
    display(100*result_df_u.round(4))
    return None
