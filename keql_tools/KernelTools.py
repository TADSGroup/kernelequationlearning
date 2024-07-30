import jax.numpy as jnp
from jax import jit,grad
import jax


def diagpart(M):
    return jnp.diag(jnp.diag(M))

def vectorize_kfunc(k):
    return jax.vmap(jax.vmap(k, in_axes=(None,0)), in_axes=(0,None))

def op_k_apply(k,L_op,R_op):
    return R_op(L_op(k,0),1)

def make_block(k,L_op,R_op):
    return vectorize_kfunc(op_k_apply(k,L_op,R_op))

def get_kernel_block_ops(k,ops_left,ops_right,output_dim=1):
    def k_super(x,y):
        I = jnp.eye(output_dim)
        blocks = (
            [
                [jnp.kron(make_block(k,L_op,R_op)(x,y),I) for R_op in ops_right]
                for L_op in ops_left
            ]
        )
        return jnp.block(blocks)
    return k_super

def eval_k(k,index):
    return k

def diff_k(k,index):
    return grad(k,index)

def diff2_k(k,index):
    return grad(grad(k,index),index)

def get_selected_grad(k,index,selected_index):
    gradf = grad(k,index)
    def selgrad(*args):
        return gradf(*args)[selected_index]
    return selgrad

def dx_k(k,index):
    return get_selected_grad(k,index,1)

def dxx_k(k,index):
    return get_selected_grad(get_selected_grad(k,index,1),index,1)

def dt_k(k,index):
    return get_selected_grad(k,index,0)

def laplacian_k(k,index):
    def lapk(*args):
        return jnp.trace(jax.hessian(k,index)(*args))
    return lapk