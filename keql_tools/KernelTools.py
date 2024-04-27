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
    def k_super(t_left,t_right):
        I = jnp.eye(output_dim)
        blocks = (
            [
                [jnp.kron(make_block(k,L_op,R_op)(t_left,t_right),I) for R_op in ops_right]
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
    def selgrad(x1,x2):
        return gradf(x1,x2)[selected_index]
    return selgrad


def dx_k(k,index):
    return get_selected_grad(k,index,1)

def dxx_k(k,index):
    return get_selected_grad(get_selected_grad(k,index,1),index,1)


def dt_k(k,index):
    return get_selected_grad(k,index,0)


from functools import partial

class InducedRKHS():
    """
    Still have to go back and allow for multiple operator sets
        For example, points on boundary only need evaluation, not the rest of the operators if we know boundary conditions
    This only does 1 dimensional output for now. 
    """
    def __init__(
            self,
            basis_points,
            operators,
            kernel_function,
            ) -> None:
        self.basis_points = basis_points
        self.operators = operators
        self.k = kernel_function
        self.get_all_op_kernel_matrix = jit(get_kernel_block_ops(self.k,self.operators,self.operators))
        self.get_eval_op_kernel_matrix = jit(get_kernel_block_ops(self.k,[eval_k],self.operators))
        self.kmat = self.get_all_op_kernel_matrix(self.basis_points,self.basis_points)
        self.num_params = len(basis_points) * len(operators)
    
    @partial(jax.jit, static_argnames=['self'])
    def evaluate_all_ops(self,eval_points,params):
        return self.get_all_op_kernel_matrix(eval_points,self.basis_points)@params
    
    @partial(jax.jit, static_argnames=['self'])
    def point_evaluate(self,eval_points,params):
        return self.get_eval_op_kernel_matrix(eval_points,self.basis_points)@params
    
    def evaluate_operators(self,operators,eval_points,params):
        return get_kernel_block_ops(self.k,operators,self.operators)(eval_points,self.basis_points)@params
    
    def get_fitted_params(self,X,y,lam = 1e-4):
        K = self.get_eval_op_kernel_matrix(X,self.basis_points)
        coeffs = jax.scipy.linalg.solve(K.T@K + lam * (self.kmat+1e-4 * diagpart(self.kmat)),K.T@y)
        return coeffs
    

