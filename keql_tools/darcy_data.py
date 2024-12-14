import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from data_utils import build_xy_grid
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular
from Kernels import get_gaussianRBF
from KernelTools import diagpart, eval_k, make_block, vectorize_kfunc

def build_darcy_op(a):
    def darcy_op(k,index):
        gradk = jax.grad(k,argnums = index)
        def agradk(*args):
            return a(args[index])*gradk(*args)
        def darcy_result(*args):
            return jnp.trace(jax.jacfwd(agradk,argnums = index)(*args))
        return darcy_result
    return darcy_op

def get_darcy_solver(
    a:callable,
    num_grid = 50,
    k_u = get_gaussianRBF(0.2),
    nugget = 2e-10,
    num_refinements = 4,
):
    darcy_op = build_darcy_op(a)
    xy_int,xy_bdy = build_xy_grid([0,1],[0,1],num_grid,num_grid)

    operators = (eval_k,darcy_op)
    point_blocks = (xy_bdy,xy_int)
    Kphiphi_blocks = [
        [
            make_block(k_u,L,R)(left_points,right_points) 
            for R,right_points in zip(operators,point_blocks)] for L,left_points in zip(operators,point_blocks)
    ]
    Kphiphi = jnp.block(Kphiphi_blocks)
    chol = cho_factor(Kphiphi + nugget * diagpart(Kphiphi))

    @jax.jit
    def get_alpha(fvals):
        rhs = jnp.hstack([jnp.zeros(len(xy_bdy)),fvals])
        # residual_norms = []
        alpha = cho_solve(chol,rhs)
        residual = rhs - Kphiphi@alpha
        for i in range(num_refinements):
            alpha = alpha + cho_solve(chol,residual)
            residual = rhs - Kphiphi@alpha
            # residual_norms.append(jnp.linalg.norm(residual))
        return alpha


    def solve_darcy(f):
        fvals = jax.vmap(f)(xy_int)
        alpha = get_alpha(fvals)

        def u(x):
            kxphi = jnp.hstack([
                jax.vmap(R(k_u,1),in_axes = (None,0))(x,right_points)
                for R,right_points in zip(operators,point_blocks)
            ])
            return jnp.dot(kxphi,alpha)
        return u
    return solve_darcy

def sample_gp_function(key,kernel,num_grid = 100,bounds = jnp.array([0,1])):
    grid = jnp.linspace(bounds[0],bounds[1],num_grid)
    x,y = jnp.meshgrid(grid,grid)
    xy_grid = jnp.vstack([x.flatten(),y.flatten()]).T
    vec_kf = vectorize_kfunc(kernel)
    Kf = vec_kf(xy_grid,xy_grid)
    fChol = cho_factor(Kf+1e-8*diagpart(Kf))
    f_alpha = solve_triangular(fChol[0],jax.random.normal(key,(len(Kf),)))
    def f(x):
        return jnp.dot(f_alpha,jax.vmap(kernel,in_axes = (None,0))(x,xy_grid))
    return f