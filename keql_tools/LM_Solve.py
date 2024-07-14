import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import jit
from tqdm.auto import tqdm

def LevenbergMarquadtMinimize(
        init_params,
        model,
        beta,
        max_iter = 200, tol = 1e-8,
        cmin = 0.1,line_search_increase_ratio = 1.5,max_line_search_iterations = 20,
        min_alpha = 1e-6,
        max_alpha = 20.,
        init_alpha = 3.,
        step_adapt_multiplier = 1.2,
        callback = None,
        print_every = 50
        ):
    """Adaptively regularized Levenberg Marquadt optimizer
    TODO: Wrap up convergence data into a convergence_data dictionary or object
    Parameters
    ----------
    init_params : jax array
        initial guess
    model :
        Object that contains model.F, and model.jac, and model.damping_matrix
    beta : float
        (global) regularization strength
    max_iter : int, optional
        by default 200
    tol : float, optional
        Gradient norm stopping tolerance
    cmin : float, optional
        Minimum armijo ratio to accept step, by default 0.1
    line_search_increase_ratio : float, optional
        constant to increase reg strength by in backtracking line search, by default 1.5
    max_line_search_iterations : int, optional
        by default 20
    min_alpha : _type_, optional
        min damping strength, by default 1e-6
    max_alpha : _type_, optional
        max damping strength, by default 20.
    init_alpha : _type_, optional
        initial damping strength, by default 3.
    step_adapt_multipler : float, optional
        value to use for adapting alpha, by default 1.2
    callback : callable, optional
        function called to print another loss each iteration, by default None
    print_every : int, optional
        How often to print convergence data, by default 50

    Returns
    -------
    _type_
        _description_
    """
    params = init_params.copy()
    J = model.jac(params)
    residuals = model.F(params)
    damping_matrix = model.damping_matrix(params)
    loss_vals = [
        (1/2)*jnp.sum(residuals**2) + (1/2)*beta * params.T@damping_matrix@params
    ]
    JtRes = [jnp.linalg.norm(J.T@residuals + beta * damping_matrix@params)]
    iterate_history = [params]
    improvement_ratios = []
    alpha_vals = []
    alpha = init_alpha

    def LevenbergMarquadtUpdate(params,alpha):
        J = model.jac(params)
        residuals = model.F(params)
        damping_matrix = model.damping_matrix(params)
        loss = (1/2)*jnp.sum(residuals**2) + (1/2)*beta * params.T@damping_matrix@params

        JtJ = J.T@J
        rhs = J.T@residuals + beta * damping_matrix@params
        alpha =jnp.clip(alpha,min_alpha,max_alpha)
        for i in range(max_line_search_iterations):
            M = JtJ + (alpha + beta) * damping_matrix
            step = solve(M,rhs,assume_a = 'pos')
            new_params = params - step
            new_reg_norm = beta * new_params.T@damping_matrix@new_params
            new_loss = (1/2)*(jnp.sum(model.F(new_params)**2) + new_reg_norm)
            predicted_loss = (1/2)*(jnp.sum((J@step-residuals)**2) + new_reg_norm)
            improvement_ratio = (loss - new_loss)/(loss - predicted_loss)

            if improvement_ratio >= cmin:
                #Check if we get at least some proportion of predicted improvement from local model
                succeeded = True
                return new_params, new_loss, rhs, improvement_ratio,alpha,succeeded
            else:
                alpha = line_search_increase_ratio * alpha
            succeeded = False
        return new_params, new_loss, rhs, improvement_ratio,alpha,succeeded

    for i in tqdm(range(max_iter)):
        params,loss,rhs,improvement_ratio,alpha,succeeded = LevenbergMarquadtUpdate(params,alpha)
        # Get new value for alpha
        multiplier = step_adapt_multiplier
        if improvement_ratio <= 0.2:
            alpha = multiplier * alpha
        if improvement_ratio >= 0.8:
            alpha = alpha/multiplier

        if succeeded==False:
            print("Line Search Failed!")
            print("Final Iteration Results")
            print(f"Iteration {i}, loss = {loss:.4}, Jres = {JtRes[-1]:.4}, alpha = {alpha:.4}")
            return params,loss_vals,JtRes,improvement_ratios,alpha_vals,iterate_history
        loss_vals += [loss]
        JtRes += [jnp.linalg.norm(rhs)]
        iterate_history += [params]
        improvement_ratios +=[improvement_ratio]
        alpha_vals +=[alpha]
        if JtRes[-1]<=tol:
            break

        if i%print_every ==0 or i<=5:
            print(f"Iteration {i}, loss = {loss:.4}, Jres = {JtRes[-1]:.4}, alpha = {alpha:.4}, improvement_ratio = {improvement_ratio:.4}")
            if callback:
                callback(params)

    return params,loss_vals,JtRes,improvement_ratios,alpha_vals,iterate_history


@jit
def l2reg_lstsq(A, y, reg=1e-10):
    U,sigma,Vt = jnp.linalg.svd(A, full_matrices=False)
    return Vt.T@((sigma/(sigma**2+reg))*(U.T@y))

def refine_solution(params,equation_model,reg_sequence = 10**(jnp.arange(-4.,-18,-0.5))):
    """Refines solution with almost pure gauss newton through SVD"""
    refinement_losses = []
    refined_params = params.copy()
    for reg in tqdm(reg_sequence):
        J = equation_model.jac(refined_params)
        F = equation_model.F(refined_params)
        refined_params = refined_params - l2reg_lstsq(J,F,reg)
        refinement_losses += [equation_model.loss(refined_params)]
    return refined_params,jnp.array(refinement_losses)

def adaptive_refine_solution(params,equation_model,initial_reg = 1e-4,num_iter = 100,mult = 0.7):
    refinement_losses = [equation_model.loss(params)]
    refined_params = params.copy()
    reg_vals = [initial_reg]
    reg = initial_reg
    for i in tqdm(range(num_iter)):
        J = equation_model.jac(refined_params)
        F = equation_model.F(refined_params)
        U,sigma,Vt = jnp.linalg.svd(J, full_matrices=False)

        candidate_regs = [mult * reg,reg,reg/mult]
        candidate_steps = [Vt.T@((sigma/(sigma**2+S))*(U.T@F)) for S in candidate_regs]
        
        loss_vals = jnp.array([
            equation_model.loss(
                refined_params - step)
             for step in candidate_steps])
        choice = jnp.argmin(loss_vals)
        reg = candidate_regs[choice]
        step = candidate_steps[choice]
        refined_params = refined_params - step
        refinement_losses.append(loss_vals[choice])
        reg_vals.append(reg)
    return refined_params,jnp.array(refinement_losses),jnp.array(reg_vals)

import jax
def run_jaxopt(solver,x0):
    state = solver.init_state(x0)
    sol = x0
    values,errors,stepsizes = [state.value],[state.error],[state.stepsize]
    update = lambda sol,state:solver.update(sol,state)
    jitted_update = jax.jit(update)
    for iter_num in tqdm(range(solver.maxiter)):
        sol,state = jitted_update(sol,state)
        values.append(state.value)
        errors.append(state.error)
        stepsizes.append(state.stepsize)
        if solver.verbose > 0:
            print("Gradient Norm: ",state.error)
            print("Loss Value: ",state.value)
        if state.error<=solver.tol:
            break
        if stepsizes[-1]==0:
            print("Restart")
            state = solver.init_state(sol)
    convergence_data = {
        "values":jnp.array(values),
        "gradnorms":jnp.array(errors),
        "stepsizes":jnp.array(stepsizes)
    }
    return sol,convergence_data,state