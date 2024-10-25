import jax
import jax.numpy as jnp

def conjugate_gradient(A, b, x0=None, tol=1e-6, maxiter=1000, M_inv=None):
    """
    Solve the linear system Ax = b using the Conjugate Gradient method with optional preconditioning.

    Parameters:
        A (callable or array): Function that computes A @ x or the matrix A itself.
        b (array): Right-hand side vector.
        x0 (array, optional): Initial guess for the solution. Defaults to zeros_like(b).
        tol (float, optional): Tolerance for convergence. Defaults to 1e-6.
        maxiter (int, optional): Maximum number of iterations. Defaults to 1000.
        M_inv (callable, optional): Function that computes the preconditioner M⁻¹ @ r.

    Returns:
        x (array): Approximate solution to Ax = b.
        residual_norms (array): Residual norms at each iteration.
    """
    if x0 is None:
        x0 = jnp.zeros_like(b)

    # Define how to apply A and M_inv
    if callable(A):
        matvec_A = A
    else:
        matvec_A = lambda x: A @ x

    if M_inv is None:
        def preconditioner(r):
            return r
    else:
        preconditioner = M_inv

    def cond_fun(state):
        i, _, _, _, _, r_norm, _ = state
        return jnp.logical_and(r_norm > tol, i < maxiter)

    def body_fun(state):
        i, x, r, p, r_dot_z, r_norm, res_norms = state

        Ap = matvec_A(p)
        denom = jnp.dot(p, Ap)
        # Prevent divide by zero
        # denom = jnp.where(denom == 0, jnp.finfo(denom.dtype).eps, denom)
        alpha = r_dot_z / denom
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        z_new = preconditioner(r_new)
        r_dot_z_new = jnp.dot(r_new, z_new)
        beta = r_dot_z_new / r_dot_z
        p_new = z_new + beta * p
        r_norm_new = jnp.linalg.norm(r_new)
        res_norms = res_norms.at[i + 1].set(r_norm_new)
        i_new = i + 1
        return (i_new, x_new, r_new, p_new, r_dot_z_new, r_norm_new, res_norms)

    # Initial residuals and directions
    r0 = b - matvec_A(x0)
    z0 = preconditioner(r0)
    p0 = z0
    r_dot_z0 = jnp.dot(r0, z0)
    r_norm0 = jnp.linalg.norm(r0)
    res_norms_init = jnp.zeros(maxiter + 1).at[0].set(r_norm0)
    state0 = (0, x0, r0, p0, r_dot_z0, r_norm0, res_norms_init)

    # Run the CG iterations with lax.while_loop
    state_final = jax.lax.while_loop(cond_fun, body_fun, state0)

    # Extract the final solution and residual norms
    _, x_final, _, _, _, _, res_norms = state_final

    # Trim the residual norms array to the actual number of iterations
    num_iters = state_final[0]
    residual_norms = res_norms[:num_iters + 1]

    return x_final, (residual_norms)

import jax
import jax.numpy as jnp

# Define a symmetric positive-definite matrix A and vector b
A_matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
b_vector = jnp.array([1.0, 2.0])

# Optional: Define a preconditioner function M_inv
def M_inv(r):
    # For demonstration, use the inverse of the diagonal of A as the preconditioner
    M_diag = jnp.diag(A_matrix)
    return r / M_diag

# Solve Ax = b using the CG method
x_approx, residuals = conjugate_gradient(A_matrix, b_vector, M_inv=M_inv)

print("Approximate solution:", x_approx)
print("Residual norms:", residuals)
