from dataclasses import dataclass, field
import jax.numpy as jnp
import jax
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import time

@dataclass
class LMParams:
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
    cmin: float = 0.05
    line_search_increase_ratio: float = 1.5
    max_line_search_iterations: int = 20
    min_alpha: float = 1e-6
    max_alpha: float = 50.0
    init_alpha: float = 3.0
    step_adapt_multiplier: float = 1.2
    callback: callable = None
    print_every: int = 50
    track_iterates: bool = False

@dataclass
class ConvergenceHistory:
    track_iterates: bool = False
    loss_vals: list = field(default_factory=list)
    gradnorm: list = field(default_factory=list)
    iterate_history: list = field(default_factory=list)
    improvement_ratios: list = field(default_factory=list)
    alpha_vals: list = field(default_factory=list)
    cumulative_time: list = field(default_factory=list)
    linear_system_rel_residual: list = field(default_factory=list)

    def update(
        self,
        loss,
        gradnorm,
        iterate,
        armijo_ratio,
        alpha,
        cumulative_time,
        linear_system_rel_residual
        ):
        # Append the new values to the corresponding lists
        self.loss_vals.append(loss)
        self.gradnorm.append(gradnorm)
        self.improvement_ratios.append(armijo_ratio)
        self.alpha_vals.append(alpha)
        self.cumulative_time.append(cumulative_time)
        self.linear_system_rel_residual.append(linear_system_rel_residual)
        
        # Conditionally track iterates if enabled
        if self.track_iterates:
            self.iterate_history.append(iterate)

    def finish(self):
        # Convert lists to JAX arrays
        self.loss_vals = jnp.array(self.loss_vals)
        self.gradnorm = jnp.array(self.gradnorm)
        self.improvement_ratios = jnp.array(self.improvement_ratios)
        self.alpha_vals = jnp.array(self.alpha_vals)
        self.cumulative_time = jnp.array(self.cumulative_time)
        self.linear_system_rel_residual = jnp.array(self.linear_system_rel_residual)
        if self.track_iterates:
            self.iterate_history = jnp.array(self.iterate_history)

@jax.jit
def l2reg_lstsq(A, y, reg=1e-10):
    U,sigma,Vt = jnp.linalg.svd(A, full_matrices=False)
    return Vt.T@((sigma/(sigma**2+reg))*(U.T@y))


@dataclass
class JaxoptHistory():
    track_iterates: bool = False
    loss_vals: list = field(default_factory=list)
    gradnorm: list = field(default_factory=list)
    iterate_history: list = field(default_factory=list)
    cumulative_time: list = field(default_factory=list)
    stepsizes: list = field(default_factory=list)

    def update(
        self,
        loss,
        gradnorm,
        iterate,
        cumulative_time,
        stepsize,
        ):
        # Append the new values to the corresponding lists
        self.loss_vals.append(loss)
        self.gradnorm.append(gradnorm)
        self.cumulative_time.append(cumulative_time)
        self.stepsizes.append(stepsize)
        
        # Conditionally track iterates if enabled
        if self.track_iterates:
            self.iterate_history.append(iterate)

    def finish(self):
        # Convert lists to JAX arrays
        self.loss_vals = jnp.array(self.loss_vals)
        self.gradnorm = jnp.array(self.gradnorm)
        self.cumulative_time = jnp.array(self.cumulative_time)
        self.stepsizes = jnp.array(self.stepsizes)
        if self.track_iterates:
            self.iterate_history = jnp.array(self.iterate_history)

def run_jaxopt(solver,x0,track_iterates = False):
    start_time = time.time()
    state = solver.init_state(x0)
    sol = x0
    history = JaxoptHistory(track_iterates=track_iterates)
    history.update(state.value,state.error,sol,0.,state.stepsize)
    update = lambda sol,state:solver.update(sol,state)
    jitted_update = jax.jit(update)
    for iter_num in tqdm(range(solver.maxiter)):
        sol,state = jitted_update(sol,state)
        history.update(state.value,state.error,sol,time.time() - start_time,state.stepsize)

        if solver.verbose > 0:
            print("Gradient Norm: ",state.error)
            print("Loss Value: ",state.value)
        if state.error<=solver.tol:
            break
        if history.stepsizes[-1]==0:
            print("Restart")
            state = solver.init_state(sol)
    history.finish()
    return sol,history,state

def pad_1025(A):
    padding_shape = jnp.array(A.shape)
    pad_index = jnp.argmax(padding_shape)
    padding_shape = padding_shape.at[jnp.argmax(padding_shape)].set(jnp.maximum(0,1025 - padding_shape[jnp.argmax(padding_shape)]))
    return jnp.concat([A,jnp.zeros(padding_shape)],axis = pad_index)

def plot_optimization_results(convergence:ConvergenceHistory):
    plt.figure(figsize=(12,5))
    plt.subplot(2,2,1)
    plt.title("Loss by iteration")
    plt.plot(convergence.loss_vals)
    plt.yscale('log')
    plt.ylabel("Loss")

    plt.subplot(2,2,2)
    plt.title("$\\|J^T (Js - F) + (\\alpha + \\beta) * Ds - \\beta * Dx\\|/\\|J^T F + \\beta Dx\\|$")
    plt.plot(convergence.linear_system_rel_residual)
    plt.yscale('log')
    plt.ylabel("Linear system residual")

    plt.subplot(2,2,3)
    plt.title("Gradient Norm")
    plt.plot(convergence.gradnorm)
    plt.yscale('log')
    plt.ylabel("Gradient Norm")
    plt.xlabel("Iteration number")

    plt.subplot(2,2,4)
    plt.title("Proximal Parameter alpha")
    plt.plot(convergence.alpha_vals)
    plt.xlabel("Iteration Number")
    plt.yscale("log")

    plt.tight_layout()
    plt.show()