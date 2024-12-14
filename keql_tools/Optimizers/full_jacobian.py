import jax.numpy as jnp
from jax.scipy.linalg import cho_factor,cho_solve
from tqdm.auto import tqdm
import time
from dataclasses import dataclass, field
from .solvers_base import LMParams,ConvergenceHistory
import jax

def print_progress(
    i,
    loss,
    gradnorm,
    alpha,
    improvement_ratio,
):
    print(  f"Iteration {i}, loss = {loss:.4},"
            f" gradnorm = {gradnorm:.4}, alpha = {alpha:.4},"
            f" improvement_ratio = {improvement_ratio:.4}"
            )

def CholeskyLM(
        init_params,
        model,
        beta,
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
        dictionary of data tracking convergence
    """
    conv_history = ConvergenceHistory(optParams.track_iterates)
    start_time = time.time()
    params = init_params.copy()
    J = model.jac(params)
    residuals = model.F(params)
    damping_matrix = model.damping_matrix(params)
    alpha = optParams.init_alpha
    if optParams.show_progress is True:
        loop_wrapper = tqdm
    else:
        loop_wrapper = lambda x:x

    conv_history.update(
        loss = (1/2)*jnp.sum(residuals**2) + (1/2)*beta * params.T@damping_matrix@params,
        gradnorm = jnp.linalg.norm(J.T@residuals + beta * damping_matrix@params),
        iterate = params,
        armijo_ratio = 1.,
        alpha = alpha,
        cumulative_time = time.time() - start_time,
        linear_system_rel_residual=0.
    )

    def evaluate_objective(params):
        J = model.jac(params)
        residuals = model.F(params)
        damping_matrix = model.damping_matrix(params)
        loss = (1/2)*jnp.sum(residuals**2) + (1/2)*beta * params.T@damping_matrix@params
        JtJ = J.T@J
        rhs = J.T@residuals + beta * damping_matrix@params
        return J,residuals,damping_matrix,loss,JtJ,rhs
    
    if optParams.use_jit is True:
        evaluate_objective = jax.jit(evaluate_objective)
    
    @jax.jit
    def compute_step(params,alpha,J,JtJ,residuals,rhs,previous_loss,damping_matrix):
        #Form and solve linear system for step
        M = JtJ + (alpha + beta) * damping_matrix
        Mchol = cho_factor(M)
        step = cho_solve(Mchol,rhs)
        Jstep = J@step

        #Track the linear system residual
        linear_residual = (
            J.T@(Jstep - residuals) + 
            (alpha+beta) * damping_matrix@step - 
            beta * damping_matrix@params
        )
        linear_system_rel_residual = (
            jnp.linalg.norm(linear_residual)/jnp.linalg.norm(rhs)
        )

        #Compute step and if we decreased loss
        new_params = params - step
        new_reg_norm = beta * new_params.T@damping_matrix@new_params
        new_loss = (1/2)*(jnp.sum(model.F(new_params)**2) + new_reg_norm)
        predicted_loss = (1/2)*(jnp.sum((Jstep-residuals)**2) + new_reg_norm)
        improvement_ratio = (previous_loss - new_loss)/(previous_loss - predicted_loss)

        return step,new_params,new_loss,improvement_ratio,linear_system_rel_residual

    def LevenbergMarquadtUpdate(params,alpha):
        J,residuals,damping_matrix,loss,JtJ,rhs = evaluate_objective(params)
        alpha =jnp.clip(alpha,optParams.min_alpha,optParams.max_alpha)
        for i in range(optParams.max_line_search_iterations):
            step,new_params,new_loss,improvement_ratio,linear_system_rel_residual = (
                compute_step(params,alpha,J,JtJ,residuals,rhs,loss,damping_matrix)
            )
            if improvement_ratio >= optParams.cmin:
                #Check if we get at least some proportion of predicted improvement from local model
                succeeded = True
                return new_params, new_loss, rhs, improvement_ratio,alpha,linear_system_rel_residual,succeeded
            else:
                alpha = optParams.line_search_increase_ratio * alpha
            succeeded = False
        return new_params, new_loss, rhs, improvement_ratio,alpha,linear_system_rel_residual,succeeded

    for i in loop_wrapper(range(optParams.max_iter)):
        params,loss,rhs,improvement_ratio,alpha,linear_system_rel_residual,succeeded = (
            LevenbergMarquadtUpdate(params,alpha)
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
            if optParams.show_progress is True:
                print_progress(i,loss,conv_history.gradnorm[-1],alpha,improvement_ratio)
            conv_history.finish()
            return params,conv_history

        conv_history.update(
            loss = loss,
            gradnorm = jnp.linalg.norm(rhs),
            iterate = params,
            armijo_ratio = improvement_ratio,
            alpha = alpha,
            cumulative_time = time.time() - start_time,
            linear_system_rel_residual = linear_system_rel_residual
        )

        if conv_history.gradnorm[-1]<=optParams.tol:
            break
        if i%optParams.print_every ==0 or i<=5 or i == optParams.max_iter:
            if optParams.show_progress is True:
                print_progress(i,loss,conv_history.gradnorm[-1],alpha,improvement_ratio)
            if optParams.callback:
                optParams.callback(params)
    conv_history.finish()
    return params,conv_history


class SVD_LMParams(LMParams):
    def __init__(self, max_iter=201, tol=1e-10, cmin=0.05,
                 line_search_increase_ratio=2.0, min_alpha=0.0,
                 step_adapt_multiplier=1.7, **kwargs):
        # Update defaults by merging any overrides from kwargs
        super().__init__(max_iter=max_iter,
                         tol=tol,
                         cmin=cmin,
                         line_search_increase_ratio=line_search_increase_ratio,
                         min_alpha=min_alpha,
                         step_adapt_multiplier=step_adapt_multiplier,
                         **kwargs)

def SVD_LM(
        init_params,
        model,
        beta,
        optParams: SVD_LMParams = SVD_LMParams()
        ):
    """Adaptively regularized Levenberg Marquardt optimizer
    Uses svd solver, assumes damping is given by identity
    Parameters
    ----------
    init_params : jax array
        initial guess
    model :
        Object that contains model.F, and model.jac
    beta : float
        (global) regularization strength
    optParams: LMParams
        optimizer hyperparameters

    Returns
    -------
    solution
        approximate minimizer
    convergence_dict
        dictionary of data tracking convergence
    """
    conv_history = ConvergenceHistory(optParams.track_iterates)
    start_time = time.time()
    params = init_params.copy()
    J = model.jac(params)
    residuals = model.F(params)
    damping_matrix = jnp.eye(len(params))
    alpha = optParams.init_alpha

    conv_history.update(
        loss = (1/2)*jnp.sum(residuals**2) + (1/2)*beta * params.T@params,
        gradnorm = jnp.linalg.norm(J.T@residuals + beta * params),
        iterate = params,
        armijo_ratio = 1.,
        alpha = alpha,
        cumulative_time = time.time() - start_time,
        linear_system_rel_residual=0.
    )

    # TODO: This pair of functions can be handled more elegantly
    # Make something like an objective_data object
    @jax.jit
    def evaluate_objective(params):
        J = model.jac(params)
        residuals = model.F(params)
        loss = (1/2)*jnp.sum(residuals**2) + (1/2)*beta * jnp.sum(params**2)
        svdResult = jnp.linalg.svd(J,full_matrices = False)
        return J,residuals,loss,svdResult
    
    @jax.jit
    def compute_step(params,alpha,J,residuals,rhs,previous_loss,svdResult):
        #Form and solve linear system for step
        U,sigma,Vt = svdResult
        
        step = Vt.T@(
            (1/(sigma**2 + alpha + beta))*(rhs)
            )
        Jstep = J@step

        #Track the linear system residual
        linear_residual = (
            J.T@(Jstep - residuals) + 
            (alpha+beta) * step - 
            beta * params
        )
        linear_system_rel_residual = (
            jnp.linalg.norm(linear_residual)/jnp.linalg.norm((rhs))
        )

        #Compute step and if we decreased loss
        new_params = params - step
        new_reg_norm = beta * jnp.sum(new_params**2)
        new_loss = (1/2)*(jnp.sum(model.F(new_params)**2) + new_reg_norm)
        predicted_loss = (1/2)*(jnp.sum((Jstep-residuals)**2) + new_reg_norm)
        improvement_ratio = (previous_loss - new_loss)/(previous_loss - predicted_loss)

        return step,new_params,new_loss,improvement_ratio,linear_system_rel_residual

    def LevenbergMarquadtUpdate(params,alpha):

        #Query objective
        J,residuals,loss,svdResult = evaluate_objective(params)

        #Precompute some stuff
        rhs = svdResult.S*(svdResult.U.T@residuals) + beta*svdResult.Vh@params

        alpha =jnp.clip(alpha,optParams.min_alpha,optParams.max_alpha)
        for i in range(optParams.max_line_search_iterations):

            #Compute steps along line search
            step,new_params,new_loss,improvement_ratio,linear_system_rel_residual = (
                compute_step(params,alpha,J,residuals,rhs,loss,svdResult)
            )
            if improvement_ratio >= optParams.cmin:
                #Check if we get at least some proportion of predicted improvement from local model
                succeeded = True
                return new_params, new_loss, rhs, improvement_ratio,alpha,linear_system_rel_residual,succeeded
            else:
                alpha = optParams.line_search_increase_ratio * alpha
            succeeded = False
        return new_params, new_loss, rhs, improvement_ratio,alpha,linear_system_rel_residual,succeeded

    for i in tqdm(range(optParams.max_iter)):
        params,loss,rhs,improvement_ratio,alpha,linear_system_rel_residual,succeeded = (
            LevenbergMarquadtUpdate(params,alpha)
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
            return params,conv_history

        conv_history.update(
            loss = loss,
            gradnorm = jnp.linalg.norm(rhs),
            iterate = params,
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
                optParams.callback(params)
    conv_history.finish()
    return params,conv_history

def oldSVDRefinement(
        params,
        equation_model,
        initial_reg = 1e-4,
        num_iter = 100,
        mult = 0.7,
        overall_regularization = 0.,
        tol = 1e-15,
        print_every = 1000000,
        min_iter = 25
        ):
    """
    Refines solution with Levenberg-Marquadt algorithm ignoring model regularization
    This ignores the function space structure, but taking advantage of extra 
    accuracy from SVD somehow pays off
    """
    lam = overall_regularization
    refinement_losses = [equation_model.loss(params) + lam * jnp.linalg.norm(params)**2]
    refined_params = params.copy()
    reg_vals = [initial_reg]
    gradnorms = [jnp.linalg.norm(jax.grad(equation_model.loss)(params) + lam * params)]
    reg = initial_reg
    for i in tqdm(range(num_iter)):
        candidate_regs = [mult * reg,reg,reg/mult]

        J = equation_model.jac(refined_params)
        F = equation_model.F(refined_params)
        
        U,sigma,Vt = jnp.linalg.svd(J, full_matrices=False)
        rhs = sigma*(U.T@F) + lam*Vt@refined_params
        gradnorm = jnp.linalg.norm(J.T@F + lam * refined_params)
        gradnorms.append(gradnorm)
        if gradnorm<=tol:
            print("Converged by gradnorm")
            break
        if i >min_iter:
            recent_decrease = jnp.min(jnp.array(refinement_losses[-20:-10])) - jnp.min(jnp.array(refinement_losses[-10:]))
            if recent_decrease <=  tol:
                print("Converged by no improvement")
                break
        if i%print_every==0:
            print(f"Iteration {i}, loss = {refinement_losses[-1]}")
        
        candidate_steps = [Vt.T@(
            (1/(sigma**2+ S + lam))*(rhs)
            )
        for S in candidate_regs]
        
        loss_vals = jnp.array([
            equation_model.loss(
                refined_params - step) + lam * jnp.linalg.norm(refined_params - step)**2
             for step in candidate_steps])
        choice = jnp.argmin(loss_vals)
        reg = candidate_regs[choice]
        step = candidate_steps[choice]
        if jnp.min(loss_vals)<refinement_losses[-1]:
            #Step accepted
            refined_params = refined_params - step
            refinement_losses.append(loss_vals[choice])
            reg_vals.append(reg)
        else:
            print(f"Iteration {i} Step Failed")
            #Step failed
            refinement_losses.append(refinement_losses[-1])
            reg_vals.append(reg_vals[-1])
            reg = reg * 2
    convergence_data = {
        "loss_vals":jnp.array(refinement_losses),
        "reg_vals":jnp.array(reg_vals),
        "gradnorms":jnp.array(gradnorms)
    }
    return refined_params,convergence_data


