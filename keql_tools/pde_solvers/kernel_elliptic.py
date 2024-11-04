from functools import partial
from Optimizers import LMParams,CholeskyLM
from parabolic_data_utils import build_alpha_chebyshev
from EquationModel import CholInducedRKHS
from data_utils import make_grids
import jax
import jax.numpy as jnp

class EllipticPDEModel():
    def __init__(
        self,
        kernel,
        vec_P_function,
        feature_operators,
        num_grid = 50,
        bdy_weight = 10,
        solverParams = LMParams(max_iter = 501,min_alpha = 1e-8,use_jit = False,show_progress = True)
        ) -> None:
        self.solverParams = solverParams
        self.kernel = kernel
        self.vec_P_function = vec_P_function
        self.feature_operators = feature_operators
        self.num_grid = num_grid
        self.bdy_weight = bdy_weight

        point_grid = build_alpha_chebyshev([0,1],self.num_grid,1.)
        self.xy_int,self.xy_bdy = make_grids(point_grid,point_grid)
        self.u_model = CholInducedRKHS(jnp.vstack([self.xy_int,self.xy_bdy]),feature_operators,kernel) 

    def input_features(
        self,
        u_params,
        evaluation_points,
        ):
        """
        Computes features as input to P_operator_model
        """
        num_points = len(evaluation_points)
        num_ops = len(self.feature_operators)
        op_evaluation = self.u_model.evaluate_operators(self.feature_operators,evaluation_points,u_params)
        u_op_features = op_evaluation.reshape(num_points,num_ops,order = 'F')
        full_features = jnp.hstack([evaluation_points,u_op_features])
        return full_features
    
    @partial(jax.jit,static_argnames = ['self'])
    def F(self,u_params,rhs_values,bdy_values):
        P_features = self.input_features(u_params,self.xy_int)
        pde_residual = rhs_values - self.vec_P_function(P_features)
        u_bdy_values = self.u_model.point_evaluate(self.xy_bdy,u_params)
        bdy_residual = jnp.sqrt(self.bdy_weight)*(bdy_values - u_bdy_values)
        return jnp.hstack([pde_residual/jnp.sqrt(len(pde_residual)),bdy_residual/jnp.sqrt(len(bdy_residual))])
    
    @partial(jax.jit, static_argnames=['self'])
    def jac(self,u_params,rhs_values,bdy_values):
        """This is to allow for custom jacobian operators, and a choice
        between forward and reverse mode autodiff.
        In particular, we may want batched map based JVP
        """
        return jax.jacrev(self.F,argnums = 0)(u_params,rhs_values,bdy_values)
    
    def damping_matrix(self,u_params):
        return jnp.identity(len(u_params))
    
    def solve(self,rhs_function,bdy_function = lambda x:0):
        rhs_values = jax.vmap(rhs_function)(self.xy_int)
        bdy_values = jax.vmap(bdy_function)(self.xy_bdy)
        class pde():
            def __init__(
                self,
                rhs_values,
                bdy_values,
                main_model = self
                ) -> None:
                self.main_model = main_model
                self.rhs_values = rhs_values
                self.bdy_values = bdy_values

            def F(self,u_params):
                return self.main_model.F(u_params,self.rhs_values,self.bdy_values)
            
            def jac(self,u_params):
                return self.main_model.jac(u_params,self.rhs_values,self.bdy_values)
            
            def damping_matrix(self,u_params):
                return self.main_model.damping_matrix(u_params)

        u_init = jnp.zeros(self.u_model.num_params)
        u_sol,conv_data = CholeskyLM(u_init,pde(rhs_values,bdy_values),1e-14,self.solverParams)
        return self.u_model,u_sol,conv_data