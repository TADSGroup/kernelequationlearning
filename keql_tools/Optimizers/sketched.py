import jax.numpy as jnp
from jax.scipy.linalg import cho_factor,cho_solve
from tqdm.auto import tqdm
import time
from dataclasses import dataclass, field
from .solvers_base import LMParams,ConvergenceHistory
import jax
from jax.scipy.sparse.linalg import cg

@dataclass
class SketchedLMParams(LMParams):
    random_key: jax.Array = field(default_factory = lambda:jax.random.PRNGKey(130))
    sketch_size: int = 200

def SketchedLM(
        init_params,
        model,
        beta,
        optParams: SketchedLMParams = SketchedLMParams(),
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
    key = optParams.random_key

    conv_history.update(
        loss = (1/2)*jnp.sum(residuals**2) + (1/2)*beta * params.T@damping_matrix@params,
        gradnorm = jnp.linalg.norm(J.T@residuals + beta * damping_matrix@params),
        iterate = params,
        armijo_ratio = 1.,
        alpha = alpha,
        cumulative_time = time.time() - start_time,
        linear_system_rel_residual=0.
    )

    @jax.jit
    def get_F_jvp(params,v):
        return jax.jvp(model.F,(params,),(v,))[1]
    
    @jax.jit
    def get_F_vjp(params,v):
        return jax.vjp(model.F,params)[1](v)[0]

    @jax.jit
    def sketch_objective(params,sketch):
        residuals,F_vjp = jax.vjp(model.F,params)
        SJ = jax.vmap(F_vjp)(sketch)[0]
        g = F_vjp(residuals)[0]
        JtJ = SJ.T@SJ
        return SJ,JtJ,residuals,g,damping_matrix
    
    @jax.jit
    def compute_step(
        params,alpha,SJ,JtJ,residuals,rhs,previous_loss,damping_matrix
        ):
        M = JtJ + (alpha + beta) * damping_matrix
        Mchol = jax.scipy.linalg.cho_factor(M)
        step = jax.scipy.linalg.cho_solve(Mchol,rhs)

        new_params = params - step
        new_reg_norm = beta * new_params.T@damping_matrix@new_params
        new_loss = (1/2)*(jnp.sum(model.F(new_params)**2) + new_reg_norm)
        Jstep = jax.jvp(model.F,(params,),(step,))[1]

        predicted_loss = (1/2)*(jnp.sum((Jstep-residuals)**2) + new_reg_norm)
        
        improvement_ratio = (previous_loss - new_loss)/(previous_loss - predicted_loss)
        Jstep = get_F_jvp(params,step)

        #Track the linear system residual
        linear_residual = (
            get_F_vjp(params,Jstep - residuals) + 
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

    def LevenbergMarquadtUpdate(params,alpha,sketch):
        SJ,JtJ,residuals,g,damping_matrix = sketch_objective(params,sketch)
        loss = (1/2)*jnp.sum(residuals**2) + (1/2)*beta * params.T@damping_matrix@params
        rhs = g + beta * damping_matrix@params

        alpha =jnp.clip(alpha,optParams.min_alpha,optParams.max_alpha)
        for i in range(optParams.max_line_search_iterations):
            step,new_params,new_loss,improvement_ratio,linear_system_rel_residual =(
                compute_step(params,alpha,SJ,JtJ,residuals,rhs,loss,damping_matrix)
            )

            if improvement_ratio >= optParams.cmin and (new_loss < loss):
                #Check if we get at least some proportion of predicted improvement from local model
                succeeded = True
                return new_params, new_loss, rhs, improvement_ratio,alpha,linear_system_rel_residual,succeeded
            else:
                alpha = optParams.line_search_increase_ratio * alpha
            succeeded = False
        return new_params, new_loss, rhs, improvement_ratio,alpha,linear_system_rel_residual,succeeded

    for i in tqdm(range(optParams.max_iter)):
        sampling_key,key = jax.random.split(key)
        sketch = jax.random.normal(
            sampling_key,
            shape = (optParams.sketch_size,len(residuals)))/jnp.sqrt(optParams.sketch_size)
        params,loss,rhs,improvement_ratio,alpha,linear_system_rel_residual,succeeded = (
            LevenbergMarquadtUpdate(params,alpha,sketch)
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


@dataclass
class SketchedCGLMParams(SketchedLMParams):
    max_cg_iter:int = 5
    target_rel_tol:float = 0.1



def SketchedCGLM(
        init_params,
        model,
        beta,
        optParams: SketchedCGLMParams = SketchedCGLMParams(),
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
    key = optParams.random_key

    conv_history.update(
        loss = (1/2)*jnp.sum(residuals**2) + (1/2)*beta * params.T@damping_matrix@params,
        gradnorm = jnp.linalg.norm(J.T@residuals + beta * damping_matrix@params),
        iterate = params,
        armijo_ratio = 1.,
        alpha = alpha,
        cumulative_time = time.time() - start_time,
        linear_system_rel_residual=0.
    )

    @jax.jit
    def apply_F_jvp(params,v):
        return jax.jvp(model.F,(params,),(v,))[1]
    
    @jax.jit
    def apply_F_vjp(params,v):
        return jax.vjp(model.F,params)[1](v)[0]

    @jax.jit
    def sketch_objective(params,sketch):
        residuals,F_vjp = jax.vjp(model.F,params)
        SJ = jax.vmap(F_vjp)(sketch)[0]
        g = F_vjp(residuals)[0]
        JtJ = SJ.T@SJ
        return SJ,JtJ,residuals,g,damping_matrix
    
    @jax.jit
    def compute_step(
        params,alpha,SJ,JtJ,residuals,rhs,previous_loss,damping_matrix
        ):
        M = JtJ + (alpha + beta) * damping_matrix
        Mchol = cho_factor(M)
        init_step = cho_solve(Mchol,rhs)


        linop = (
            lambda x:
            apply_F_vjp(params,apply_F_jvp(params,x)) + alpha * damping_matrix@x
        )
        step,info = cg(
            linop,rhs,init_step,M = lambda x:cho_solve(Mchol,x),
            tol = optParams.target_rel_tol,
            maxiter = optParams.max_cg_iter
            )

        new_params = params - step
        new_reg_norm = beta * new_params.T@damping_matrix@new_params
        new_loss = (1/2)*(jnp.sum(model.F(new_params)**2) + new_reg_norm)
        Jstep = apply_F_jvp(params,step)

        predicted_loss = (1/2)*(jnp.sum((Jstep-residuals)**2) + new_reg_norm)
        improvement_ratio = (previous_loss - new_loss)/(previous_loss - predicted_loss)

        #Track the linear system residual
        linear_residual = (
            apply_F_vjp(params,Jstep - residuals) + 
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

    def LevenbergMarquadtUpdate(params,alpha,sketch):
        SJ,JtJ,residuals,g,damping_matrix = sketch_objective(params,sketch)
        loss = (1/2)*jnp.sum(residuals**2) + (1/2)*beta * params.T@damping_matrix@params
        rhs = g + beta * damping_matrix@params

        alpha =jnp.clip(alpha,optParams.min_alpha,optParams.max_alpha)
        for i in range(optParams.max_line_search_iterations):
            step,new_params,new_loss,improvement_ratio,linear_system_rel_residual =(
                compute_step(params,alpha,SJ,JtJ,residuals,rhs,loss,damping_matrix)
            )

            if improvement_ratio >= optParams.cmin and (new_loss < loss):
                #Check if we get at least some proportion of predicted improvement from local model
                succeeded = True
                return new_params, new_loss, rhs, improvement_ratio,alpha,linear_system_rel_residual,succeeded
            else:
                alpha = optParams.line_search_increase_ratio * alpha
            succeeded = False
        return new_params, new_loss, rhs, improvement_ratio,alpha,linear_system_rel_residual,succeeded

    for i in tqdm(range(optParams.max_iter)):
        sampling_key,key = jax.random.split(key)
        sketch = jax.random.normal(
            sampling_key,
            shape = (optParams.sketch_size,len(residuals)))/jnp.sqrt(optParams.sketch_size)
        params,loss,rhs,improvement_ratio,alpha,linear_system_rel_residual,succeeded = (
            LevenbergMarquadtUpdate(params,alpha,sketch)
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
