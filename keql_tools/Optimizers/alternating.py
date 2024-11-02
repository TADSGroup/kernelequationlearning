import jax.numpy as jnp
from jax.scipy.linalg import cho_factor,cho_solve
from tqdm.auto import tqdm
import time
from dataclasses import dataclass, field
from Optimizers.solvers_base import LMParams,ConvergenceHistory
import jax
from functools import partial
from jaxopt import AndersonAcceleration

def build_updates_alternating(model,beta_reg_P,beta_reg_u,datafit_weight):
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

    def P_residual_single(
        u_param,
        P_params,
        single_collocation_points,
        single_rhs
    ):
        eqn_res = model.equation_residual_single(
            u_param,
            P_params,
            single_collocation_points,
            single_rhs
        )
        return eqn_res/jnp.sqrt(len(eqn_res))
    stacked_colloc = jnp.stack(model.collocation_points)
    stacked_rhs = jnp.stack(model.rhs_forcing_values)
    stacked_obs_points = jnp.stack(model.observation_points)
    stacked_obs_values = jnp.stack(model.observation_values)
    data_args = [stacked_colloc,stacked_rhs,stacked_obs_points,stacked_obs_values]
    u_vmap_axes = (0,None,0,0,0,0)

    @jax.jit
    def update_u(u_params,P_params,alpha):
        jacobians = jax.vmap(
            jax.jacrev(
                single_function_residuals,argnums=0
                )
                ,in_axes = u_vmap_axes)(u_params,P_params,*data_args)
        
        F = jax.vmap(
            single_function_residuals
            ,in_axes = u_vmap_axes
            )(u_params,P_params,*data_args)
        
        g = jax.vmap(lambda J,f:J.T@f)(jacobians,F)
        rhs = g + beta_reg_u * u_params
        def LM_step(u_param,J,rhs_single):
            JtJ = J.T@J + (alpha + beta_reg_u) * jnp.eye(len(u_param))
            return u_param - jnp.linalg.solve(JtJ,rhs_single)
        return jax.vmap(LM_step)(u_params,jacobians,rhs)

    
    @jax.jit
    def update_P(u_params,P_params,alpha):
        p_vmap_axes = (0,None,0,0)
        jacobians = jax.vmap(
            jax.jacrev(
                P_residual_single,argnums=1
                )
                ,in_axes = p_vmap_axes)(u_params,P_params,stacked_colloc,stacked_rhs)

        F = jax.vmap(
            P_residual_single
            ,in_axes = p_vmap_axes
            )(u_params,P_params,stacked_colloc,stacked_rhs)
        
        g = jnp.sum(jax.vmap(lambda J,f:J.T@f)(jacobians,F),axis=0)

        rhs = g + beta_reg_P * P_params
        JtJ = jnp.sum(jax.vmap(lambda J:J.T@J)(jacobians),axis=0) + (alpha + beta_reg_P) * jnp.eye(len(P_params))
        return P_params - jnp.linalg.solve(JtJ,rhs)

    @jax.jit
    @partial(jax.value_and_grad,argnums = (0,1))
    def full_loss_valgrad(u_params,P_params):
        residuals = (
            jax.vmap(
                single_function_residuals,in_axes = u_vmap_axes
                )(
                    u_params,P_params,
                    *data_args
                    )
        )
        return (
            jnp.sum(residuals**2) + 
            beta_reg_P * jnp.sum(P_params**2) + 
            beta_reg_u * jnp.sum(jnp.mean(u_params**2,axis=1)
                                )
        )
    return update_u,update_P,full_loss_valgrad,single_function_residuals,P_residual_single


@dataclass
class AltLMParams:
    """
    max_iter : int, optional
        by default 201
    tol : float, optional
        Gradient norm stopping tolerance
    cmin : float, optional
        Minimum armijo ratio to accept step, by default 0.05
    line_search_increase_ratio : float, optional
        constant to increase reg strength by in backtracking line search, by default 1.5
    max_line_search_iterations : int, optional
        by default 20
    min_alpha : float, optional
        min damping strength, by default 1e-6
    max_alpha : float, optional
        max damping strength, by default 50.
    init_alpha : float, optional
        initial damping strength, by default 3.
    step_adapt_multipler : float, optional
        value to use for adapting alpha, by default 1.2
    callback : callable, optional
        function called to print another loss each iteration, by default None
    print_every : int, optional
        How often to print convergence data, by default 50
    """
    max_iter: int = 201
    tol: float = 1e-8
    line_search_increase_ratio: float = 1.5
    max_line_search_iterations: int = 20
    min_alpha: float = 1e-8
    max_alpha: float = 50.0
    init_alpha: float = 2.
    alpha_decrease: float = 0.75
    callback: callable = None
    print_every: int = 50
    track_iterates: bool = False

@dataclass
class AlternatingConvergenceHistory:
    track_iterates: bool = False
    loss_vals: list = field(default_factory=list)
    u_gradnorm: list = field(default_factory=list)
    P_gradnorm: list = field(default_factory=list)
    u_iterate_history: list = field(default_factory=list)
    P_iterate_history:list = field(default_factory=list)
    alpha_vals: list = field(default_factory=list)
    P_alpha_vals: list = field(default_factory=list)
    cumulative_time: list = field(default_factory=list)

    def update(
        self,
        loss,
        u_gradnorm,
        P_gradnorm,
        u_iterate,
        P_iterate,
        alpha,
        cumulative_time,
        ):
        # Append the new values to the corresponding lists
        self.loss_vals.append(loss)
        self.u_gradnorm.append(u_gradnorm)
        self.P_gradnorm.append(P_gradnorm)
        self.alpha_vals.append(alpha)
        self.cumulative_time.append(cumulative_time)
        
        # Conditionally track iterates if enabled
        if self.track_iterates:
            self.u_iterate_history.append(u_iterate)
            self.P_iterate_history.append(P_iterate)

    def finish(self):
        # Convert lists to JAX arrays
        self.loss_vals = jnp.array(self.loss_vals)
        self.u_gradnorm = jnp.array(self.u_gradnorm)
        self.P_gradnorm = jnp.array(self.P_gradnorm)
        self.alpha_vals = jnp.array(self.alpha_vals)
        self.cumulative_time = jnp.array(self.cumulative_time)
        if self.track_iterates:
            self.u_iterate_history = jnp.array(self.u_iterate_history)
            self.P_iterate_history = jnp.array(self.P_iterate_history)

def AlternatingLM(
    u_init,
    P_init,
    model,
    beta_reg_P,
    beta_reg_u,
    optParams: AltLMParams = AltLMParams()
):
    start_time = time.time()
    u_params = u_init
    P_params = P_init
    conv_history = AlternatingConvergenceHistory(optParams.track_iterates)
    datafit_weight = model.datafit_weight
    update_u,update_P,full_loss_valgrad,single_function_residuals,P_residual_single = (
        build_updates_alternating(model,beta_reg_P,beta_reg_u,datafit_weight)
    )
    alpha = optParams.init_alpha

    previous_loss,(u_grad,P_grad) = full_loss_valgrad(u_params,P_params)
    
    conv_history.update(
        loss = previous_loss,
        u_gradnorm = jnp.linalg.norm(u_grad),
        P_gradnorm = jnp.linalg.norm(P_grad),
        u_iterate = u_params,
        P_iterate = P_params,
        alpha = alpha,
        cumulative_time = time.time() - start_time,
    )
    def UpdateStep(u_params,P_params,alpha,previous_loss):
        alpha = jnp.clip(alpha,optParams.min_alpha,optParams.max_alpha)
        succeeded = False
        for i in range(optParams.max_line_search_iterations):
            u_params = update_u(u_params,P_params,alpha)
            P_params = update_P(u_params,P_params,alpha)
            new_loss,(u_grad,P_grad) = full_loss_valgrad(u_params,P_params)
            if new_loss<previous_loss:
                succeeded = True
                return u_params,P_params,u_grad,P_grad,new_loss,alpha,succeeded
            alpha = optParams.line_search_increase_ratio * alpha
        return u_params,P_params,u_grad,P_grad,new_loss,alpha,succeeded
    
    for i in tqdm(range(optParams.max_iter)):
        u_params,P_params,u_grad,P_grad,loss,alpha,succeeded = UpdateStep(u_params,P_params,alpha,previous_loss)
        if succeeded is False:
            print("Line Search Failed!")
            print("Final Iteration Results")
            print(
                f"Iteration {i}, loss = {loss:.4},"
                f" u_gradnorm = {conv_history.u_gradnorm[-1]:.4}, "
                f" P_gradnorm = {conv_history.P_gradnorm[-1]:.4}, alpha = {alpha:.4},"
                )
            conv_history.finish()
            return u_params,P_params,conv_history
        alpha = optParams.alpha_decrease * alpha
        conv_history.update(
            loss = loss,
            u_gradnorm = jnp.linalg.norm(u_grad),
            P_gradnorm = jnp.linalg.norm(P_grad),
            u_iterate = u_params,
            P_iterate = P_params,
            alpha = alpha,
            cumulative_time = time.time() - start_time,
        )
        if conv_history.u_gradnorm[-1]<=optParams.tol and conv_history.P_gradnorm[-1]<=optParams.tol:
            break
        if i%optParams.print_every ==0 or i<=5 or i == optParams.max_iter:
            print(
                f"Iteration {i}, loss = {loss:.4},"
                f" u_gradnorm = {conv_history.u_gradnorm[-1]:.4}, "
                f" P_gradnorm = {conv_history.P_gradnorm[-1]:.4}, alpha = {alpha:.4},"
                )
            if optParams.callback:
                optParams.callback(u_params,P_params)
        previous_loss = loss
    conv_history.finish()
    return u_params,P_params,conv_history

@dataclass
class AndersonAltLMParams:
    """
    max_iter : int, optional
        by default 201
    tol : float, optional
        Gradient norm stopping tolerance
    alpha : float, optional
        Proximal Regularization, by default 1e-6
    callback : callable, optional
        function called to print another loss each iteration, by default None
    print_every : int, optional
        How often to print convergence data, by default 50
    track_iterates : bool, optional
        Whether to track iterates, by default False
    """
    tol: float = 1e-8
    alpha: float = 1e-6
    callback: callable = None
    print_every: int = 50
    track_iterates: bool = False
    AA_params: dict = field(
        default_factory=lambda:{
            'history_size':5,
            'maxiter':201
            })

@dataclass
class AndersonConvergenceHistory:
    track_iterates: bool = False
    loss_vals: list = field(default_factory=list)
    u_gradnorm: list = field(default_factory=list)
    P_gradnorm: list = field(default_factory=list)
    u_iterate_history: list = field(default_factory=list)
    P_iterate_history:list = field(default_factory=list)
    cumulative_time: list = field(default_factory=list)

    def update(
        self,
        loss,
        u_gradnorm,
        P_gradnorm,
        u_iterate,
        P_iterate,
        cumulative_time,
        ):
        # Append the new values to the corresponding lists
        self.loss_vals.append(loss)
        self.u_gradnorm.append(u_gradnorm)
        self.P_gradnorm.append(P_gradnorm)
        self.cumulative_time.append(cumulative_time)
        
        # Conditionally track iterates if enabled
        if self.track_iterates:
            self.u_iterate_history.append(u_iterate)
            self.P_iterate_history.append(P_iterate)

    def finish(self):
        # Convert lists to JAX arrays
        self.loss_vals = jnp.array(self.loss_vals)
        self.u_gradnorm = jnp.array(self.u_gradnorm)
        self.P_gradnorm = jnp.array(self.P_gradnorm)
        self.cumulative_time = jnp.array(self.cumulative_time)
        if self.track_iterates:
            self.u_iterate_history = jnp.array(self.u_iterate_history)
            self.P_iterate_history = jnp.array(self.P_iterate_history)

def AndersonAlternatingLM(
    u_init,
    P_init,
    model,
    beta_reg_P,
    beta_reg_u,
    optParams: AndersonAltLMParams = AndersonAltLMParams()
):
    start_time = time.time()
    u_params = u_init
    P_params = P_init
    conv_history = AndersonConvergenceHistory(optParams.track_iterates)
    datafit_weight = model.datafit_weight
    update_u,update_P,full_loss_valgrad,single_function_residuals,P_residual_single = (
        build_updates_alternating(model,beta_reg_P,beta_reg_u,datafit_weight)
    )
    alpha = optParams.alpha

    loss,(u_grad,P_grad) = full_loss_valgrad(u_params,P_params)
    
    conv_history.update(
        loss = loss,
        u_gradnorm = jnp.linalg.norm(u_grad),
        P_gradnorm = jnp.linalg.norm(P_grad),
        u_iterate = u_params,
        P_iterate = P_params,
        cumulative_time = time.time() - start_time,
    )
    def FixedPointUpdate(u_P_param_tuple):
        u_params,P_params = u_P_param_tuple
        u_params = update_u(u_params,P_params,alpha)
        P_params = update_P(u_params,P_params,alpha)
        return u_params,P_params
    solver = AndersonAcceleration(
        FixedPointUpdate,
        **optParams.AA_params
        )
    
    state = solver.init_state((u_params,P_params))
    update = lambda sol,state:solver.update(sol,state)
    jitted_update = jax.jit(update)
    sol = (u_params,P_params)
    for iter_num in tqdm(range(solver.maxiter)):
        sol,state = jitted_update(sol,state)
        u_params,P_params = sol
        loss,(u_grad,P_grad) = full_loss_valgrad(u_params,P_params)
        conv_history.update(
            loss = loss,
            u_gradnorm = jnp.linalg.norm(u_grad),
            P_gradnorm = jnp.linalg.norm(P_grad),
            u_iterate = u_params,
            P_iterate = P_params,
            cumulative_time = time.time() - start_time,
            )
        if state.error<=solver.tol:
            break
        if (
            jnp.linalg.norm(jnp.linalg.norm(u_grad))<optParams.tol and 
            jnp.linalg.norm(jnp.linalg.norm(P_grad))<optParams.tol
        ):
            break
    conv_history.finish()
    return u_params,P_params,conv_history,state
