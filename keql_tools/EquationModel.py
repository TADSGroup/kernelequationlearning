import jax.numpy as jnp
from jax import jit,jacrev
import jax
from functools import partial
from KernelTools import get_kernel_block_ops,eval_k,diagpart
from jax.scipy.linalg import block_diag,cholesky,solve_triangular
from typing import Optional

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
    
    @partial(jit, static_argnames=['self'])
    def evaluate_all_ops(self,eval_points,params):
        return self.get_all_op_kernel_matrix(eval_points,self.basis_points)@params
    
    @partial(jit, static_argnames=['self'])
    def point_evaluate(self,eval_points,params):
        return self.get_eval_op_kernel_matrix(eval_points,self.basis_points)@params
    
    @partial(jit, static_argnames=['self','operators'])
    def evaluate_operators(self,operators,eval_points,params):
        return get_kernel_block_ops(self.k,operators,self.operators)(eval_points,self.basis_points)@params
    
    def get_fitted_params(self,X,y,lam = 1e-6,eps = 1e-4):
        K = self.get_eval_op_kernel_matrix(X,self.basis_points)
        coeffs = jax.scipy.linalg.solve(K.T@K + lam * (self.kmat+eps * diagpart(self.kmat)),K.T@y,assume_a = 'pos')
        return coeffs
    
    def get_eval_function(self,params):
        def u(x):
            return (self.get_eval_op_kernel_matrix(x.reshape(1,-1),self.basis_points)@params)[0]
        return u
    
    def get_damping(self):
        return self.kmat

class CholInducedRKHS():
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
            nugget_size = 1e-6
            ) -> None:
        self.basis_points = basis_points
        self.operators = operators
        self.k = kernel_function
        self.get_all_op_kernel_matrix = jit(get_kernel_block_ops(self.k,self.operators,self.operators))
        self.get_eval_op_kernel_matrix = jit(get_kernel_block_ops(self.k,[eval_k],self.operators))
        self.kmat = self.get_all_op_kernel_matrix(self.basis_points,self.basis_points)
        self.cholT = cholesky(self.kmat + nugget_size * diagpart(self.kmat),lower = False)
        self.num_params = len(basis_points) * len(operators)
    
    @partial(jit, static_argnames=['self'])
    def evaluate_all_ops(self,eval_points,params):
        return self.get_all_op_kernel_matrix(eval_points,self.basis_points)@solve_triangular(self.cholT,params)
    
    @partial(jit, static_argnames=['self'])
    def point_evaluate(self,eval_points,params):
        return self.get_eval_op_kernel_matrix(eval_points,self.basis_points)@solve_triangular(self.cholT,params)
    
    @partial(jit, static_argnames=['self','operators'])
    def evaluate_operators(self,operators,eval_points,params):
        return get_kernel_block_ops(self.k,operators,self.operators)(eval_points,self.basis_points)@solve_triangular(self.cholT,params)
    
    def get_fitted_params(self,X,y,lam = 1e-6,eps = 1e-4):
        K = self.get_eval_op_kernel_matrix(X,self.basis_points)
        alpha_coeffs = jax.scipy.linalg.solve(K.T@K + lam * (self.kmat+eps * diagpart(self.kmat)),K.T@y,assume_a = 'pos')
        params = self.cholT@alpha_coeffs
        return params
    
    def get_eval_function(self,params):
        def u(x):
            return (self.get_eval_op_kernel_matrix(x.reshape(1,-1),self.basis_points)@solve_triangular(self.cholT,params))[0]
        return u
    
    def get_damping(self):
        return jnp.identity(self.num_params)
    
    
def check_OperatorPDEModel(
        u_models:tuple,
        observation_points:tuple,
        observation_values:tuple,
        collocation_points:tuple,
        rhs_forcing_values:tuple,
    ):
    try:
        consistent_dimensions = (len(u_models)==
            len(observation_points)==
            len(observation_values)==
            len(collocation_points)==
            len(rhs_forcing_values))
        assert consistent_dimensions, "Data dimensions don't match up"
    except AssertionError as message: 
        print(message)
        print("u_models given: ",len(u_models))
        print("sets of observation_points given: ",len(observation_points))
        print("sets of observation_values given: ",len(observation_values))
        print("sets of collocation_points given: ",len(collocation_points))
        print("sets of rhs_forcing_values given: ",len(rhs_forcing_values))
        print("These should all match")
    for obs_points,obs_vals in zip(observation_points,observation_values):
        assert len(obs_points)==len(obs_vals), "Number of observation locations don't match number of observed values"
    for col_points,rhs_vals in zip(collocation_points,rhs_forcing_values):
        assert len(col_points)==len(rhs_vals), "Number of collocation points don't match number of rhs values"

class OperatorModel():
    def __init__(
        self,
        kernel,
        nugget_size = 1e-7
    ):
        self.kernel_function = kernel
        self.nugget_size = nugget_size

    def predict(self,input_data,params):
        K = self.kernel_function(input_data,input_data)
        return K@params
        
    def predict_new(self,X,anchors,params):
        return self.kernel_function(X,anchors)@params
    
    def fit_params(self,X,y,nugget = 1e-8):
        K = self.kernel_function(X,X)
        return jnp.linalg.solve(K + nugget * diagpart(K),y)
    
    def rkhs_mat(self,X):
        return self.kernel_function(X,X)

class CholOperatorModel():
    def __init__(
        self,
        kernel,
        nugget_size = 1e-8,
    ):
        self.kernel_function = kernel
        self.nugget_size = nugget_size

    def predict(self,X,params):
        K = self.kernel_function(X,X)
        cholT = cholesky(K + self.nugget_size * diagpart(K),lower = False)
        return K@solve_triangular(cholT,params)
    
    def predict_new(self,X,anchors,params):
        K = self.kernel_function(anchors,anchors)
        cholT = cholesky(K + self.nugget_size * diagpart(K),lower = False)
        return self.kernel_function(X,anchors)@solve_triangular(cholT,params)
    
    def fit_params(self,X,y,nugget = 1e-8):
        K = self.kernel_function(X,X)
        cholT = cholesky(K + self.nugget_size * diagpart(K),lower = False)
        return cholT@jnp.linalg.solve(K + nugget * diagpart(K),y)

    def rkhs_mat(self,X):
        return jnp.eye(len(X))


class OperatorPDEModel():
    def __init__(
        self,
        operator_model,
        u_models:tuple,
        observation_points:tuple,
        observation_values:tuple,
        collocation_points:tuple,
        feature_operators:tuple,
        rhs_forcing_values:Optional[tuple]=None,
        rhs_operator=None,
        datafit_weight = 10,
    ):
        if rhs_forcing_values is None:
            rhs_forcing_values = tuple(jnp.zeros(len(col_points)) for col_points in collocation_points)
        check_OperatorPDEModel(u_models,observation_points,observation_values,collocation_points,rhs_forcing_values)
        self.u_models = u_models
        self.operator_model = operator_model

        self.num_operator_params = sum(map(len,collocation_points))
        self.observation_points = observation_points
        self.observation_values = observation_values
        self.collocation_points = collocation_points
        self.rhs_forcing_values = rhs_forcing_values
        self.feature_operators = feature_operators
        self.datafit_weight = datafit_weight

        self.rhs_operator = rhs_operator

        self.stacked_observation_values = jnp.hstack(observation_values)
        self.stacked_collocation_rhs = jnp.hstack(rhs_forcing_values)
        
        #Precompute parameter indices to pull out different parameter blocks
        self.total_parameters = sum([model.num_params for model in u_models]) + self.num_operator_params
        
        #Assume we put the parameters for the operators at the start of the flattened parameter set
        self.operator_model_indices = jnp.arange(self.total_parameters - self.num_operator_params,self.total_parameters)

        #Compute the start and end indices of the parameter sets for u_models
        u_param_inds = jnp.cumsum(jnp.array([0]+[model.num_params for model in u_models]))
        self.u_indexing = [
            jnp.arange(p,q) for p,q in zip(u_param_inds[:-1],u_param_inds[1:])
        ]

    def get_P_params(self,all_params):
        return all_params[self.operator_model_indices]
    
    def get_u_params(self,all_params):
        return [all_params[ind_set] for ind_set in self.u_indexing]

    @partial(jit, static_argnames=['self','u_model'])
    def get_single_eqn_features(
        self,
        u_model, 
        u_params,
        evaluation_points,
        ):
        num_points = len(evaluation_points)
        num_ops = len(self.feature_operators)
        op_evaluation = u_model.evaluate_operators(self.feature_operators,evaluation_points,u_params)
        u_op_features = op_evaluation.reshape(num_points,num_ops,order = 'F')
        full_features = jnp.hstack([evaluation_points,u_op_features])
        return full_features
    
    @partial(jit, static_argnames=['self'])
    def get_stacked_eqn_features(
        self,
        all_u_params
    ):
        return jnp.vstack([
            self.get_single_eqn_features(u_model,u_params,eval_points) 
            for u_model,u_params,eval_points in zip(
                self.u_models,
                all_u_params,
                self.collocation_points
            )
        ])
    
    @partial(jit, static_argnames=['self','u_model'])
    def get_rhs_op_single(self,u_model,u_params,evaluation_points):
        if self.rhs_operator is None:
            return jnp.zeros(len(evaluation_points))
        else:
            op_evaluation = u_model.evaluate_operators((self.rhs_operator,),evaluation_points,u_params)
            return op_evaluation.flatten()
        
    @partial(jit, static_argnames=['self'])
    def get_stacked_rhs_op(
        self,
        all_u_params
    ):
        return jnp.hstack([
            self.get_rhs_op_single(u_model,u_params,eval_points) 
            for u_model,u_params,eval_points in zip(
                self.u_models,
                all_u_params,
                self.collocation_points
            )
        ])
    
    def get_overall_rhs(self,all_u_params):
        return self.get_stacked_rhs_op(all_u_params) + self.stacked_collocation_rhs
    
    @partial(jit, static_argnames=['self'])
    def equation_residual(self,full_params):
        """
        In the future, we may want to break this up, 
        calculating residuals before stacking so we can look at errors on each function individually
        """
        all_u_params = self.get_u_params(full_params)
        P_params = self.get_P_params(full_params)
        stacked_features = self.get_stacked_eqn_features(all_u_params=all_u_params)
        P_preds = self.operator_model.predict(stacked_features,P_params)
        overall_rhs = self.get_overall_rhs(all_u_params)
    
        return (overall_rhs - P_preds)


    @partial(jit, static_argnames=['self'])
    def datafit_residual(self,full_params):
        all_u_params = self.get_u_params(full_params)
        obs_preds = jnp.hstack(
            [
                model.point_evaluate(obs_points,u_params) 
                for model,obs_points,u_params in zip(
                    self.u_models,
                    self.observation_points,
                    all_u_params)
                ])
        return self.stacked_observation_values - obs_preds
    
    @partial(jit, static_argnames=['self'])
    def F(self,full_params):
        eqn_res = self.equation_residual(full_params)
        data_res = self.datafit_residual(full_params)
        return jnp.hstack([
            jnp.sqrt(self.datafit_weight) * data_res/jnp.sqrt(len(data_res)),
            eqn_res/jnp.sqrt(len(eqn_res))
            ])
    
    #usually jacrev is faster than jacfwd on examples I've tested
    jac = jit(jacrev(F,argnums = 1),static_argnames='self')

    @partial(jit, static_argnames=['self'])
    def loss(self,full_params):
        """
        TODO: Include the regularization term here instead of in EqnModel"""
        return (1/2) * jnp.linalg.norm(self.F(full_params))**2
    
    @partial(jit, static_argnames=['self'])
    def damping_matrix(self,full_params,nugget = 1e-3):
        """
        Presumably, I shouldn't build the kernel matrix for P again, but that would make the code
        more unwieldy
        """
        u_params = self.get_u_params(full_params)
        grid_feats = self.get_stacked_eqn_features(u_params)
        dmat = block_diag(
            *([model.get_damping() for model in self.u_models]+[self.operator_model.rkhs_mat(grid_feats)])
            )
        dmat= dmat + nugget*diagpart(dmat)
        return dmat
    
class SplitOperatorPDEModel():
    def __init__(
        self,
        operator_model,
        u_models:tuple,
        observation_points:tuple,
        observation_values:tuple,
        collocation_points:tuple,
        feature_operators:tuple,
        rhs_forcing_values:Optional[tuple]=None,
        rhs_operator=None,
        datafit_weight = 10,
        num_P_operator_params = None
    ):
        if rhs_forcing_values is None:
            rhs_forcing_values = tuple(jnp.zeros(len(col_points)) for col_points in collocation_points)
        check_OperatorPDEModel(u_models,observation_points,observation_values,collocation_points,rhs_forcing_values)
        self.u_models = u_models
        self.operator_model = operator_model

        if num_P_operator_params is None:
            self.num_operator_params = sum(map(len,collocation_points))
        else:
            self.num_operator_params = num_P_operator_params
        self.observation_points = observation_points
        self.observation_values = observation_values
        self.collocation_points = collocation_points
        self.rhs_forcing_values = rhs_forcing_values
        self.feature_operators = feature_operators
        self.datafit_weight = datafit_weight

        self.rhs_operator = rhs_operator

        self.stacked_observation_values = jnp.hstack(observation_values)
        self.stacked_collocation_rhs = jnp.hstack(rhs_forcing_values)
        
        #Precompute parameter indices to pull out different parameter blocks
        self.total_parameters = sum([model.num_params for model in u_models]) + self.num_operator_params
        
        #Assume we put the parameters for the operators at the start of the flattened parameter set
        self.operator_model_indices = jnp.arange(self.total_parameters - self.num_operator_params,self.total_parameters)

        #Compute the start and end indices of the parameter sets for u_models
        u_param_inds = jnp.cumsum(jnp.array([0]+[model.num_params for model in u_models]))
        self.u_indexing = [
            jnp.arange(p,q) for p,q in zip(u_param_inds[:-1],u_param_inds[1:])
        ]

    def get_P_params(self,all_params):
        """
        Extract parameters associated to P from the stacked all_params
        """
        return all_params[self.operator_model_indices]
    
    def get_u_params(self,all_params):
        """
        Extract the parameters associated to 
        interpolating each function from stacked all_params
        """
        return tuple(all_params[ind_set] for ind_set in self.u_indexing)
    
    def apply_rhs_op_single(self,u_model,u_params,evaluation_points):
        """
        Applies the previously specified fixed RHS operator to all of 
        u_model parameterized by u_params at evalutaion_points
        """
        if self.rhs_operator is None:
            return jnp.zeros(len(evaluation_points))
        else:
            op_evaluation = u_model.evaluate_operators((self.rhs_operator,),evaluation_points,u_params)
            return op_evaluation.flatten()
    
    @partial(jit, static_argnames=['self','u_model'])
    def single_eqn_features(
        self,
        u_model, 
        u_params,
        evaluation_points,
        ):
        """
        Computes features as input to P_operator_model
        """
        num_points = len(evaluation_points)
        num_ops = len(self.feature_operators)
        op_evaluation = u_model.evaluate_operators(self.feature_operators,evaluation_points,u_params)
        u_op_features = op_evaluation.reshape(num_points,num_ops,order = 'F')
        full_features = jnp.hstack([evaluation_points,u_op_features])
        return full_features
    
    def equation_residual_single(
        self,
        u_model,
        u_params,
        P_params,
        evaluation_points,
        rhs_forcing
    ):
        """
        Computes the equation residual associated to 
        u_model parametrized by u_params,
        P_operator_model parametrized by P_params

        based on rhs_operator(u) + rhs_forcing at evaluation_points
        """
        features = self.single_eqn_features(u_model,u_params,evaluation_points)
        P_predictions = self.operator_model.predict(features,P_params)
        rhs_values = self.apply_rhs_op_single(u_model,u_params,evaluation_points) + rhs_forcing
        return rhs_values - P_predictions

    def stacked_equation_residual(
        self,
        all_u_params:tuple,
        P_params:jax.Array
    ):
        return jnp.hstack(
            [
                self.equation_residual_single(
                    u_model,u_params,P_params,eval_points,rhs_forcing
                    ) for u_model,u_params,eval_points,rhs_forcing in zip(
                        self.u_models,all_u_params,
                        self.collocation_points,self.rhs_forcing_values
                        )
                ]
                )

    def datafit_residual_single(
        self,
        u_model,
        u_params,
        obs_points,
        obs_vals,
        ):
        return obs_vals - u_model.point_evaluate(obs_points,u_params)
    
    def stacked_datafit_residual(
        self,
        all_u_params:tuple
    ):
        return jnp.hstack(
            [
                self.datafit_residual_single(u_model,u_params,obs_points,obs_vals)
                for u_model,u_params,obs_points,obs_vals in zip(self.u_models,all_u_params,self.observation_points,self.observation_values)
                ]
        )
    
    def F(self,full_params):
        all_u_params = self.get_u_params(full_params)
        P_params = self.get_P_params(full_params)
        eqn_res = self.stacked_equation_residual(all_u_params,P_params)
        data_res = self.stacked_datafit_residual(all_u_params)
        return jnp.hstack([
            jnp.sqrt(self.datafit_weight) * data_res/jnp.sqrt(len(data_res)),
            eqn_res/jnp.sqrt(len(eqn_res))
            ])
        
    #usually jacrev is faster than jacfwd on examples I've tested
    jac = jit(jacrev(F,argnums = 1),static_argnames='self')

    def loss(self,full_params):
        """
        TODO: Include the regularization term here instead of in EqnModel"""
        return (1/2) * jnp.linalg.norm(self.F(full_params))**2
    
    def damping_matrix(self,full_params,nugget = 1e-3):
        u_params = self.get_u_params(full_params)
        grid_feats = jnp.vstack([self.single_eqn_features(u_model,u_params,eval_points) for 
                                 u_model,u_params,eval_points in zip(
                                    self.u_models,u_params,self.collocation_points)])
        dmat = block_diag(
            *([model.get_damping() for model in self.u_models]+[self.operator_model.rkhs_mat(grid_feats)])
            )
        dmat= dmat + nugget*diagpart(dmat)
        return dmat