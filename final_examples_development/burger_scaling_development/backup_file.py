import time
from jax.scipy.sparse.linalg import cg
from jax.scipy.linalg import solve

def Sketched_LM(
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

        out,F_jvp = jax.linearize(model.F,params)

        #@jax.jit
        def GN_mvp(v):
            return jax.vjp(model.F,params)[1](F_jvp(v))[0]
        

        damping_matrix = model.damping_matrix(params)
        loss = (1/2)*jnp.sum(residuals**2) + (1/2)*beta * params.T@damping_matrix@params

        JtJ = SJ.T@SJ

        #I differentiate these two because one is sketched
        # rhs = SJ.T@residuals + beta * damping_matrix@params 
        full_gradient = g + beta * damping_matrix@params

        alpha =jnp.clip(alpha,min_alpha,max_alpha)
        for i in range(max_line_search_iterations):
            M = JtJ + (alpha + beta) * damping_matrix
            step_init = solve(M,full_gradient,assume_a = 'pos')
            step,info = cg(A = lambda x:GN_mvp(x)+(alpha + beta)*damping_matrix@x,b = full_gradient,x0 = step_init,M = M,maxiter = 10)

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
