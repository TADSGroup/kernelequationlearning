import jax.numpy as jnp
from jax.scipy.linalg import cholesky,cho_solve,solve
from tqdm.auto import tqdm
import time
from dataclasses import dataclass, field
from Optimizers.solvers_base import LMParams,ConvergenceHistory
from EquationModel import SharedOperatorPDEModel
import jax
from functools import partial

def setup_arrow_functions(model,beta_reg_u,beta_reg_P,datafit_weight):
    def single_function_residuals(
        u_param,
        P_params,
        single_collocation_points,
        single_rhs,
        single_observation_points,
        single_observation_values,
    ):
        datafit_res = model.datafit_residual_single(
            u_param,
            single_observation_points,
            single_observation_values,
        )
        eqn_res = model.equation_residual_single(
            u_param,
            P_params,
            single_collocation_points,
            single_rhs
        )
        return jnp.hstack([datafit_res*jnp.sqrt(datafit_weight/len(datafit_res)),eqn_res/jnp.sqrt(len(eqn_res))])

    stacked_colloc = jnp.stack(model.collocation_points)
    stacked_rhs = jnp.stack(model.rhs_forcing_values)
    stacked_obs_points = jnp.stack(model.observation_points)
    stacked_obs_values = jnp.stack(model.observation_values)
    data_args = [stacked_colloc,stacked_rhs,stacked_obs_points,stacked_obs_values]

    u_vmap_axes = (0,None,0,0,0,0)

    @jax.jit
    def full_loss(u_params,P_params):
        residuals = (
            jax.vmap(
                single_function_residuals,in_axes = u_vmap_axes
                )(
                    u_params,P_params,
                    *data_args
                    )
        )
        return 0.5 * (
            jnp.sum(residuals**2) + 
            beta_reg_P * jnp.sum(P_params**2) + 
            beta_reg_u * jnp.sum(u_params**2)
            )
        
    
    _jacU_func = jax.vmap(jax.jacrev(single_function_residuals,argnums = 0),in_axes=u_vmap_axes)

    @jax.jit
    def jacU(u_params,P_params):
        return _jacU_func(u_params,P_params,*data_args)
    
    _jacP_func = jax.vmap(jax.jacrev(single_function_residuals,argnums = 1),in_axes=u_vmap_axes)

    @jax.jit
    def jacP(u_params,P_params):
        return _jacP_func(u_params,P_params,*data_args)
    
    return full_loss,single_function_residuals,jacU,jacP,data_args

def BlockArrowLM(
        u_init,
        P_init,
        model:SharedOperatorPDEModel,
        beta_reg_u:float = 1e-12,
        beta_reg_P:float = 1e-12, 
        optParams: LMParams = LMParams() 
        ):
    """Adaptively regularized Levenberg Marquardt optimizer
    Parameters
    ----------
    init_params : jax array
        initial guess
    model :
        Object that contains model.F, and model.jac, and model.damping_matrix
    beta : float
        (global) regularization strength
    optParams: LMParams
        optimizer hyperparameters

    Returns
    -------
    solution
        approximate minimizer
    convergence_dict
        dictionary of data tracking convergencef """
    u_params = u_init
    P_params = P_init
    full_loss,single_function_residuals,jacU,jacP,data_args = (
        setup_arrow_functions(model,beta_reg_u,beta_reg_P,model.datafit_weight)
    )
    #We're being pretty sloppy with flops here
    #calling the loss and grad separately, and then 
    #building the rhs again (essentially also grad)
    grad_loss = jax.jit(jax.grad(full_loss))


    conv_history = ConvergenceHistory(track_iterates=False)
    start_time = time.time()
    alpha = optParams.init_alpha
    loss = full_loss(u_params,P_params)

    conv_history.update(
        loss = loss,
        gradnorm = jnp.linalg.norm(grad_loss(u_params,P_params)),
        iterate = None,
        armijo_ratio = 1.,
        alpha = alpha,
        cumulative_time = time.time() - start_time,
        linear_system_rel_residual=0.
    )

    @jax.jit
    def evaluate_objective(u_params,P_params):
        JU = jacU(u_params,P_params)
        JP = jacP(u_params,P_params)
        resF = jax.vmap(
            single_function_residuals,in_axes = (0,None,0,0,0,0)
            )(u_params,P_params,*data_args)

        #build RHS vectors
        Fp = jnp.sum(
            jax.vmap(lambda J,f:J.T@f)(JP,resF),axis=0
        )+beta_reg_P*P_params

        Fu = jax.vmap(lambda J,f,up:J.T@f + beta_reg_u*up)(JU,resF,u_params)
        return JU, JP, resF, Fp, Fu
    
    @jax.jit
    def compute_step(u_params,P_params,JU,JP,resF,Fp,Fu,alpha,previous_loss):
        #build RHS vectors
        Fp = jnp.sum(
            jax.vmap(lambda J,f:J.T@f)(JP,resF),axis=0
        )+beta_reg_P*P_params
        Fu = jax.vmap(lambda J,f,up:J.T@f + beta_reg_u*up)(JU,resF,u_params)

        #Build normal equation matrices
        Ap = jnp.sum(jax.vmap(lambda J:J.T@J)(JP),axis=0)
        Ap = Ap + (alpha + beta_reg_P)*jnp.eye(len(Ap))
        Du = jax.vmap(lambda a:a.T@a + (alpha + beta_reg_P)*jnp.eye(len(a[0])))(JU)

        #Cholesky factor Du since we have two solves with it
        cholDu = jax.vmap(lambda A:cholesky(A))(Du)
        #Compute Bup = Jp.T@ Ju
        Bup = jax.vmap(lambda Ju,Jp:Ju.T@Jp)(JU,JP)
        #Compute Schur complement S
        S = Ap - jnp.sum(
            jax.vmap(lambda b,d:b.T@(cho_solve((d,False),b)))(Bup,cholDu),axis=0
        )
        #Compute rhs piece
        BDFu = jnp.sum(
            jax.vmap(lambda b,d,f:b.T@cho_solve((d,False),f))(Bup,cholDu,Fu),axis=0
        )
        #Compute steps in P and u from block matrix inversion
        P_step = solve(S,Fp - BDFu,assume_a = 'pos')
        u_step = jax.vmap(
            lambda d,f:cho_solve((d,False),f)
            )(cholDu,Fu - (Bup@P_step))
        
        updated_u = u_params - u_step
        updated_P = P_params - P_step

        #Track the linear system residual
        linear_error = jnp.linalg.norm(Ap@P_step + jnp.sum(jax.vmap(lambda M,u: M.T@u)(Bup,u_step),axis = 0) - Fp)**2
        linear_error = linear_error + jnp.linalg.norm(
            (Bup@P_step) + jax.vmap(lambda d,u:d@u)(Du,u_step) - Fu
        )**2
        
        gradnorm = jnp.sqrt(jnp.linalg.norm(Fp)**2 + jnp.linalg.norm(Fu)**2)
        linear_system_rel_residual = (
            jnp.sqrt(linear_error)/gradnorm
        )

        #Compute step and if we decreased loss
        new_loss = full_loss(updated_u,updated_P)

        Jstep = jax.vmap(lambda j,u:j@u)(JU,u_step) + jax.vmap(lambda j,p:j@p,in_axes = (0,None))(JP,P_step)
        new_reg_norm = (1/2) * (beta_reg_P * jnp.sum(updated_P**2) + beta_reg_u * jnp.sum(updated_u**2))

        predicted_loss = (1/2)*(jnp.sum((Jstep - resF)**2)) + new_reg_norm

        improvement_ratio = (previous_loss - new_loss)/(previous_loss - predicted_loss)

        return updated_u,updated_P,new_loss,improvement_ratio,linear_system_rel_residual,gradnorm

    def LevenbergMarquadtUpdate(u_params,P_params,alpha,previous_loss):
        JU, JP, resF, Fp, Fu = evaluate_objective(u_params,P_params)
        alpha =jnp.clip(alpha,optParams.min_alpha,optParams.max_alpha)
        
        for i in range(optParams.max_line_search_iterations):
            updated_u,updated_P,new_loss,improvement_ratio,linear_system_rel_residual,gradnorm = (
                compute_step(u_params,P_params,JU,JP,resF,Fp,Fu,alpha,previous_loss)
            )
            if improvement_ratio >= optParams.cmin:
                #Check if we get at least some proportion of predicted improvement from local model
                succeeded = True
                return updated_u,updated_P, new_loss, gradnorm, improvement_ratio,alpha,linear_system_rel_residual,succeeded
            else:
                alpha = optParams.line_search_increase_ratio * alpha
            succeeded = False
        return updated_u,updated_P, new_loss, gradnorm, improvement_ratio,alpha,linear_system_rel_residual,succeeded

    for i in tqdm(range(optParams.max_iter)):
        u_params,P_params, loss, gradnorm, improvement_ratio,alpha,linear_system_rel_residual,succeeded = (
            LevenbergMarquadtUpdate(u_params,P_params,alpha,loss)
        )
        # Get new value for alpha
        multiplier = optParams.step_adapt_multiplier
        if improvement_ratio <= 0.2:
            alpha = multiplier * alpha
        if improvement_ratio >= 0.8:
            alpha = alpha/multiplier

        if succeeded==False:
            print("Line Search Failed!")
            print("Final Iteration Results")
            print(
                f"Iteration {i}, loss = {loss:.4},"
                f" gradnorm = {conv_history.gradnorm[-1]:.4}, alpha = {alpha:.4},"
                f" improvement_ratio = {improvement_ratio:.4}"
                )
            conv_history.finish()
            return u_params,P_params,conv_history

        conv_history.update(
            loss = loss,
            gradnorm = gradnorm,
            iterate = None,
            armijo_ratio = improvement_ratio,
            alpha = alpha,
            cumulative_time = time.time() - start_time,
            linear_system_rel_residual = linear_system_rel_residual
        )

        if conv_history.gradnorm[-1]<=optParams.tol:
            break
        if i%optParams.print_every ==0 or i<=5 or i == optParams.max_iter:
            print(
                f"Iteration {i}, loss = {loss:.4},"
                f" gradnorm = {conv_history.gradnorm[-1]:.4}, alpha = {alpha:.4},"
                f" improvement_ratio = {improvement_ratio:.4}"
                )
            if optParams.callback:
                optParams.callback(u_params,P_params)
    conv_history.finish()
    return u_params,P_params,conv_history
