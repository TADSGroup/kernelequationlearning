import jax.numpy as jnp
from jax.scipy.linalg import cho_factor,cho_solve
from tqdm.auto import tqdm
import time
from dataclasses import dataclass, field
from .solvers_base import LMParams,ConvergenceHistory
from EquationModel import SharedOperatorPDEModel
import jax

def BlockArrowLM(
        init_params,
        model:SharedOperatorPDEModel,
        beta:float = 1e-12,
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

    conv_history.update(
        loss = (1/2)*jnp.sum(residuals**2) + (1/2)*beta * params.T@damping_matrix@params,
        gradnorm = jnp.linalg.norm(J.T@residuals + beta * damping_matrix@params),
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
        damping_matrix = model.damping_matrix(params)
        loss = (1/2)*jnp.sum(residuals**2) + (1/2)*beta * params.T@damping_matrix@params
        JtJ = J.T@J
        rhs = J.T@residuals + beta * damping_matrix@params
        return J,residuals,damping_matrix,loss,JtJ,rhs
    
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
