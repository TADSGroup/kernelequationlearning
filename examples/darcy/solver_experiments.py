def learned_operator(k,index=0):
    def op_k(*args):
        features = jnp.array([op(k,index)(*args) for op in feature_operators])
        features = jnp.hstack([args[index],features]).reshape(1,-1)
        return P_model.predict(features,P_sol)[0]
    return op_k


from Kernels import get_gaussianRBF
from data_utils import build_xy_grid
from jax.scipy.linalg import solve_triangular,cho_factor,cho_solve
from KernelTools import make_block,diagpart

def get_solver(
    linear_operator,
    num_grid = 20,
    k_u = get_gaussianRBF(0.2),
    nugget = 2e-10,
    num_refinements = 5,
):
    xy_int,xy_bdy = build_xy_grid([0,1],[0,1],num_grid,num_grid)

    operators = (eval_k,linear_operator)
    point_blocks = (xy_bdy,xy_int)
    cpu = jax.devices("cpu")[0]
    Kphiphi_blocks = [
        [
            jax.jit(make_block(k_u,L,R),device = cpu)(left_points,right_points) 
            for R,right_points in zip(operators,point_blocks)] for L,left_points in zip(operators,point_blocks)
    ]
    Kphiphi = jnp.block(Kphiphi_blocks)
    num_refinements = 4
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

    def solve_pde(f):
        fvals = jax.vmap(f)(xy_int)
        alpha = get_alpha(fvals)

        def u(x):
            kxphi = jnp.hstack([
                jax.vmap(R(k_u,1),in_axes = (None,0))(x,right_points)
                for R,right_points in zip(operators,point_blocks)
            ])
            return jnp.dot(kxphi,alpha)
        return u
    return solve_pde

evaluation_key = pkey(10)
num_evaluation_functions = 10
kernel_f = get_gaussianRBF(0.15)
keys = jax.random.split(evaluation_key,num_evaluation_functions)

rhs_functions_eval = tuple(
    sample_gp_function(subkey,kernel_f) for subkey in keys
)
eval_solutions = [darcy_solve(f) for f in rhs_functions_eval]


num_grid = 50
xy_int,xy_bdy = build_xy_grid([0,1],[0,1],num_grid,num_grid)

pde_kernel = get_gaussianRBF(0.1)

operators = (eval_k,learned_operator)
point_blocks = (xy_bdy,xy_int)
cpu = jax.devices("cpu")[0]
Kphiphi_blocks = [
    [
        jax.jit(make_block(pde_kernel,L,R),device = cpu)(left_points,right_points) 
        for R,right_points in zip(operators,point_blocks)] for L,left_points in zip(operators,point_blocks)
]
Kphiphi = jnp.block(Kphiphi_blocks)

nugget = 1e-4
chol = cho_factor(Kphiphi + nugget * diagpart(Kphiphi))
num_refinements = 4
ps_inv = jnp.linalg.pinv(Kphiphi,1e-7,hermitian = True)
@jax.jit
def get_alpha(fvals):
    rhs = jnp.hstack([jnp.zeros(len(xy_bdy)),fvals])
    alpha = jnp.linalg.solve(Kphiphi,rhs)
    # # residual_norms = []
    # alpha = cho_solve(chol,rhs)
    # residual = rhs - Kphiphi@alpha
    # for i in range(num_refinements):
    #     alpha = alpha + cho_solve(chol,residual)
    #     residual = rhs - Kphiphi@alpha
    #     # residual_norms.append(jnp.linalg.norm(residual))
    return alpha#,ps_inv@rhs

def solve_pde(f):
    fvals = jax.vmap(f)(xy_int)
    alpha = get_alpha(fvals)

    def u(x):
        kxphi = jnp.hstack([
            jax.vmap(R(k_u,1),in_axes = (None,0))(x,right_points)
            for R,right_points in zip(operators,point_blocks)
        ])
        return jnp.dot(kxphi,alpha)
    return u


learned_solutions = [solve_pde(f) for f in rhs_functions_eval]

grid = jnp.linspace(0,1,100)
x,y = jnp.meshgrid(grid,grid)
fine_grid = jnp.vstack([x.flatten(),y.flatten()]).T

u_learned = jax.lax.map(learned_solutions[0],fine_grid,batch_size = 50)

jnp.linalg.norm(u_learned - jax.vmap(eval_solutions[0])(fine_grid))/jnp.linalg.norm(jax.vmap(eval_solutions[0])(fine_grid))


plt.tricontourf(fine_grid[:,0],fine_grid[:,1],u_learned,100)
plt.colorbar()
plt.show()
plt.tricontourf(fine_grid[:,0],fine_grid[:,1],jax.vmap(eval_solutions[0])(fine_grid),100)
plt.colorbar()
plt.show()