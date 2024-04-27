import jax.numpy as jnp
from jax import jit,grad
import jax
from functools import partial
from KernelTools import get_kernel_block_ops,eval_k,diagpart

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
    
    def get_fitted_params(self,X,y,lam = 1e-6,eps = 1e-4):
        K = self.get_eval_op_kernel_matrix(X,self.basis_points)
        coeffs = jax.scipy.linalg.solve(K.T@K + lam * (self.kmat+eps * diagpart(self.kmat)),K.T@y,assume_a = 'pos')
        return coeffs
