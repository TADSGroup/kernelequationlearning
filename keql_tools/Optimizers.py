import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax.scipy.sparse.linalg import cg
from jax import jit
from tqdm.auto import tqdm
import time


def CholeskyLM(
        init_params,
        model,
        beta,
        max_iter = 200, tol = 1e-8,
        cmin = 0.05,line_search_increase_ratio = 1.5,max_line_search_iterations = 20,
        min_alpha = 1e-6,
        max_alpha = 20.,
        init_alpha = 3.,
        step_adapt_multiplier = 1.2,
        callback = None,
        print_every = 50,
        track_iterates = False
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
    start_time = time.time()
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
    cumulative_time = []
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
            convergence_results = {
                "loss_vals":loss_vals,
                "norm_JtRes":JtRes,
                "armijo_ratios":improvement_ratios,
                "alpha_vals":alpha_vals,

            }
            return params,convergence_results
        cumulative_time.append(time.time()-start_time)
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
    convergence_results = {
        "loss_vals":loss_vals,
        "norm_JtRes":JtRes,
        "armijo_ratios":improvement_ratios,
        "alpha_vals":alpha_vals,
        'time_spent':cumulative_time
    }
    if track_iterates is True:
        convergence_results['iterate_history']=iterate_history
    return params,convergence_results


@jit
def l2reg_lstsq(A, y, reg=1e-10):
    U,sigma,Vt = jnp.linalg.svd(A, full_matrices=False)
    return Vt.T@((sigma/(sigma**2+reg))*(U.T@y))

def refine_solution(params,equation_model,reg_sequence = 10**(jnp.arange(-4.,-18,-0.5))):
    """
    Deprecated! Use adaptive refine solution
    Refines solution with almost pure gauss newton through SVD
    """
    refinement_losses = []
    refined_params = params.copy()
    for reg in tqdm(reg_sequence):
        J = equation_model.jac(refined_params)
        F = equation_model.F(refined_params)
        refined_params = refined_params - l2reg_lstsq(J,F,reg)
        refinement_losses += [equation_model.loss(refined_params)]
    return refined_params,jnp.array(refinement_losses)

def SVD_LM(
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

def SketchedLM(
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
        print_every = 50,
        track_iterates = False,
        sketch_size = 200,
        random_key = jax.random.PRNGKey(310)
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
    start_time = time.time()
    params = init_params.copy()

    residuals,F_vjp = jax.vjp(model.F,params)

    residuals = model.F(params)
    damping_matrix = model.damping_matrix(params)

    loss_vals = [
        (1/2)*jnp.sum(residuals**2) + (1/2)*beta * params.T@damping_matrix@params
    ]
    JtRes = [jnp.linalg.norm(F_vjp(residuals)[0] + beta * damping_matrix@params)]
    iterate_history = [params]
    improvement_ratios = []
    alpha_vals = []
    cumulative_time = []
    alpha = init_alpha

    key = random_key

    @jax.jit
    def sketch_objective(params,sketch):
        residuals,F_vjp = jax.vjp(model.F,params)
        SJ = jax.vmap(F_vjp)(sketch)[0]
        g = F_vjp(residuals)[0]
        return residuals,g,SJ


    def LevenbergMarquadtUpdate(params,alpha,sketch):
        residuals,g,SJ = sketch_objective(params,sketch)
        

        damping_matrix = model.damping_matrix(params)
        loss = (1/2)*jnp.sum(residuals**2) + (1/2)*beta * params.T@damping_matrix@params

        JtJ = SJ.T@SJ

        #I differentiate these two because one is sketched
        # rhs = SJ.T@residuals + beta * damping_matrix@params 
        full_gradient = g + beta * damping_matrix@params

        alpha =jnp.clip(alpha,min_alpha,max_alpha)
        for i in range(max_line_search_iterations):
            M = JtJ + (alpha + beta) * damping_matrix
            step = solve(M,full_gradient,assume_a = 'pos')
            new_params = params - step
            new_reg_norm = beta * new_params.T@damping_matrix@new_params
            new_loss = (1/2)*(jnp.sum(model.F(new_params)**2) + new_reg_norm)
            Jstep = jax.jvp(model.F,(params,),(step,))[1]

            predicted_loss = (1/2)*(jnp.sum((Jstep-residuals)**2) + new_reg_norm)

            improvement_ratio = (loss - new_loss)/(loss - predicted_loss)

            if improvement_ratio >= cmin and (new_loss < loss):
                #Check if we get at least some proportion of predicted improvement from local model
                succeeded = True
                return new_params, new_loss, full_gradient, improvement_ratio,alpha,succeeded
            else:
                alpha = line_search_increase_ratio * alpha
            succeeded = False
        return new_params, new_loss, full_gradient, improvement_ratio,alpha,succeeded

    for i in tqdm(range(max_iter)):
        sampling_key,key = jax.random.split(key)
        sketch = jax.random.normal(sampling_key,shape = (sketch_size,len(residuals)))/jnp.sqrt(sketch_size)
        params,loss,full_gradient,improvement_ratio,alpha,succeeded = LevenbergMarquadtUpdate(params,alpha,sketch)
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
            convergence_results = {
                "loss_vals":loss_vals,
                "norm_JtRes":JtRes,
                "armijo_ratios":improvement_ratios,
                "alpha_vals":alpha_vals,
                'time_spent':cumulative_time

            }
            return params,convergence_results
        cumulative_time.append(time.time()-start_time)
        loss_vals += [loss]
        JtRes += [jnp.linalg.norm(full_gradient)]
        iterate_history += [params]
        improvement_ratios +=[improvement_ratio]
        alpha_vals +=[alpha]
        if JtRes[-1]<=tol:
            break

        if i%print_every ==0 or i<=5:
            print(f"Iteration {i}, loss = {loss:.4}, Jres = {JtRes[-1]:.4}, alpha = {alpha:.4}, improvement_ratio = {improvement_ratio:.4}")
            if callback:
                callback(params)
    convergence_results = {
        "loss_vals":loss_vals,
        "norm_JtRes":JtRes,
        "armijo_ratios":improvement_ratios,
        "alpha_vals":alpha_vals,
        'time_spent':cumulative_time
    }
    if track_iterates is True:
        convergence_results['iterate_history']=iterate_history
    return params,convergence_results

import time
from jax.scipy.sparse.linalg import cg
from jax.scipy.linalg import solve,cho_factor,cho_solve

def SketchedCG_LM(
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
        print_every = 50,
        track_iterates = False,
        sketch_size = 200,
        random_key = jax.random.PRNGKey(310)
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
    fitted_params
        optimized parameters
    convergence_data
        dictionary of convergence data
    """
    start_time = time.time()
    params = init_params.copy()

    residuals,F_vjp = jax.vjp(model.F,params)

    residuals = model.F(params)
    damping_matrix = model.damping_matrix(params)

    loss_vals = [
        (1/2)*jnp.sum(residuals**2) + (1/2)*beta * params.T@damping_matrix@params
    ]
    JtRes = [jnp.linalg.norm(F_vjp(residuals)[0] + beta * damping_matrix@params)]
    iterate_history = [params]
    improvement_ratios = []
    alpha_vals = []
    cumulative_time = []
    alpha = init_alpha

    key = random_key

    @jax.jit
    def sketch_objective(params,sketch):
        residuals,F_vjp = jax.vjp(model.F,params)
        SJ = jax.vmap(F_vjp)(sketch)[0]
        g = F_vjp(residuals)[0]
        return residuals,g,SJ
    
    @jax.jit
    def get_step(params,sketched_JtJ,alpha,damping_matrix,g,z):
        full_gradient = g + beta * z
        M = sketched_JtJ + (alpha + beta) * damping_matrix
        chol = cho_factor(M)
        init_step = cho_solve(chol,full_gradient)
        # fvals,Jstep = jax.jvp(model.F,(params,),(init_step,))
        # linear_residual = jax.vjp(model.F,params)[1](Jstep - fvals)[0] + beta * z - alpha*damping_matrix@step
        # step = step - cho_solve(chol,linear_residual)
        linop = (
            lambda x:
            jax.vjp(model.F,params)[1](jax.jvp(model.F,(params,),(x,))[1])[0] + alpha * damping_matrix@x
        )
        step,info = cg(linop,full_gradient,init_step,M = lambda x:cho_solve(chol,x),tol = 1e-1,maxiter = 50)

        return step


    def LevenbergMarquadtUpdate(params,alpha,sketch):
        residuals,g,SJ = sketch_objective(params,sketch)
        sketched_JtJ = SJ.T@SJ
        damping_matrix = model.damping_matrix(params)
        z = damping_matrix@params
        loss = (1/2)*jnp.sum(residuals**2) + (1/2)*beta * params.T@z
        full_gradient = g + beta * z

        for i in range(max_line_search_iterations):
            alpha =jnp.clip(alpha,min_alpha,max_alpha)

            #Step Computation Here
            step = get_step(params,sketched_JtJ,alpha,damping_matrix,g,z)
            #Step computation end
            

            new_params = params - step
            new_reg_norm = beta * new_params.T@damping_matrix@new_params
            new_loss = (1/2)*(jnp.sum(model.F(new_params)**2) + new_reg_norm)
            Jstep = jax.jvp(model.F,(params,),(step,))[1]

            predicted_loss = (1/2)*(jnp.sum((Jstep-residuals)**2) + new_reg_norm)

            improvement_ratio = (loss - new_loss)/(loss - predicted_loss)

            if improvement_ratio >= cmin and (new_loss < loss):
                #Check if we get at least some proportion of predicted improvement from local model
                succeeded = True
                return new_params, new_loss, full_gradient, improvement_ratio,alpha,succeeded
            else:
                alpha = line_search_increase_ratio * alpha
            succeeded = False
        return new_params, new_loss, full_gradient, improvement_ratio,alpha,succeeded

    for i in tqdm(range(max_iter)):
        sampling_key,key = jax.random.split(key)
        sketch = jax.random.normal(sampling_key,shape = (sketch_size,len(residuals)))/jnp.sqrt(sketch_size)
        params,loss,full_gradient,improvement_ratio,alpha,succeeded = LevenbergMarquadtUpdate(params,alpha,sketch)
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
            convergence_results = {
                "loss_vals":loss_vals,
                "norm_JtRes":JtRes,
                "armijo_ratios":improvement_ratios,
                "alpha_vals":alpha_vals,
                'time_spent':cumulative_time
            }
            return params,convergence_results
        cumulative_time.append(time.time()-start_time)
        loss_vals += [loss]
        JtRes += [jnp.linalg.norm(full_gradient)]
        iterate_history += [params]
        improvement_ratios +=[improvement_ratio]
        alpha_vals +=[alpha]
        if JtRes[-1]<=tol:
            break

        if i%print_every ==0 or i<=5:
            print(f"Iteration {i}, loss = {loss:.4}, Jres = {JtRes[-1]:.4}, alpha = {alpha:.4}, improvement_ratio = {improvement_ratio:.4}")
            if callback:
                callback(params)
    convergence_results = {
        "loss_vals":loss_vals,
        "norm_JtRes":JtRes,
        "armijo_ratios":improvement_ratios,
        "alpha_vals":alpha_vals,
        'time_spent':cumulative_time
    }
    if track_iterates is True:
        convergence_results['iterate_history']=iterate_history
    return params,convergence_results




def pad_1025(A):
    padding_shape = jnp.array(A.shape)
    pad_index = jnp.argmax(padding_shape)
    padding_shape = padding_shape.at[jnp.argmax(padding_shape)].set(jnp.maximum(0,1025 - padding_shape[jnp.argmax(padding_shape)]))
    return jnp.concat([A,jnp.zeros(padding_shape)],axis = pad_index)


