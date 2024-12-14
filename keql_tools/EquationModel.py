import jax.numpy as jnp
from jax import jit,jacrev
import jax
from functools import partial
from KernelTools import get_kernel_block_ops,eval_k,diagpart,vectorize_kfunc
from jax.scipy.linalg import block_diag,cholesky,solve_triangular
from Optimizers import l2reg_lstsq
from typing import Optional

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
    
    def get_fitted_params(self,X,y,lam = 1e-6):
        K = self.get_eval_op_kernel_matrix(X,self.basis_points)
        M = solve_triangular(self.cholT.T,K.T,lower = True).T
        return l2reg_lstsq(M,y,reg = lam)
    
    def get_eval_function(self,params):
        def u(x):
            return (self.get_eval_op_kernel_matrix(x.reshape(1,-1),self.basis_points)@solve_triangular(self.cholT,params))[0]
        return u
    
    def get_damping(self):
        return jnp.identity(self.num_params)

from functools import partial
from jax import tree_util
@tree_util.register_pytree_node_class
class AltCholInducedRKHS():
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
        self.operators = operators
        self.k = kernel_function

        self.basis_points = basis_points
        self.kmat = get_kernel_block_ops(self.k,self.operators,self.operators)(self.basis_points,self.basis_points)
        self.cholT = cholesky(self.kmat + nugget_size * diagpart(self.kmat),lower = False)
        self.num_params = len(basis_points) * len(operators)
    
    # Define tree_flatten
    def tree_flatten(self):
        # Dynamic attributes (leaves)
        children = (
            self.basis_points,
            self.kmat,
            self.cholT,
            self.num_params,
        )
        # Static attributes
        aux_data = {
            'operators': self.operators,
            'k': self.k,
        }
        return children, aux_data
    
    # Define tree_unflatten
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        (
            obj.basis_points,
            obj.kmat,
            obj.cholT,
            obj.num_params,
        ) = children
        obj.operators = aux_data['operators']
        obj.k = aux_data['k']
        return obj

    @jax.jit
    def evaluate_basis(self,x):
        return jnp.hstack([
            jax.vmap(op(self.k,1),in_axes = (None,0))(x,self.basis_points)
            for op in self.operators])
    
    @jax.jit
    def point_eval_single(self,x,params):
        return jnp.dot(self.evaluate_basis(x),solve_triangular(self.cholT,params))
    
    @partial(jax.jit,static_argnames = 'operator')
    def evaluate_operator(self,operator,x,params):
        return operator(self.point_eval_single,0)(x,params)
    
    def point_evaluate(self,eval_points,params):
        return jax.vmap(self.point_eval_single,in_axes = (0,None))(eval_points,params)
    
    @partial(jax.jit,static_argnames = 'operators')
    def evaluate_operators(self,operators,eval_points,params):
        return jnp.hstack([
            jax.vmap(self.evaluate_operator,in_axes = (None,0,None))(op,eval_points,params)
            for op in operators])
    
    def evaluate_all_ops(self,eval_points,params):
        return self.evaluate_operators(self.operators,eval_points,params)    
    
    def get_fitted_params(self,X,y,lam = 1e-6):
        K = get_kernel_block_ops(self.k,(eval_k,),self.operators)(X,self.basis_points)
        M = solve_triangular(self.cholT.T,K.T,lower = True).T
        return l2reg_lstsq(M,y,reg = lam)
    
    def get_eval_function(self,params):
        return lambda x:self.point_eval_single(x,params)
    
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
    
class InducedOperatorModel():
    def __init__(
            self,
            basis_points,
            kernel_function,
            nugget_size = 1e-6
            ) -> None:
        self.basis_points = basis_points
        self.k = kernel_function
        self.kvec = vectorize_kfunc(self.k)
        self.kmat = self.kvec(self.basis_points,self.basis_points)
        self.cholT = cholesky(self.kmat + nugget_size * diagpart(self.kmat),lower = False)
        self.num_params = len(basis_points)
    
    
    @partial(jit, static_argnames=['self'])
    def predict(self,eval_points,params):
        return self.kvec(eval_points,self.basis_points)@solve_triangular(self.cholT,params)
    
    def get_fitted_params(self,X,y,lam = 1e-6):
        K = self.kvec(X,self.basis_points)
        M = solve_triangular(self.cholT.T,K.T,lower = True).T
        return l2reg_lstsq(M,y,reg = lam)

        
    def get_damping(self):
        return jnp.identity(self.num_params)

    def rkhs_mat(self,X):
        return jnp.identity(self.num_params)


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
        jacobian_operator = jax.jacrev,
        num_P_operator_params = None
    ):
        if rhs_forcing_values is None:
            rhs_forcing_values = tuple(jnp.zeros(len(col_points)) for col_points in collocation_points)
        check_OperatorPDEModel(u_models,observation_points,observation_values,collocation_points,rhs_forcing_values)
        self.u_models = u_models
        self.operator_model = operator_model
        self.residual_dimension = sum([len(a) for a in observation_points]) + sum([len(a) for a in collocation_points])

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
        self.jacobian_operator = jacobian_operator

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
    @partial(jit,static_argnames = ['self'])
    def F(self,full_params):
        print("F")
        all_u_params = self.get_u_params(full_params)
        P_params = self.get_P_params(full_params)
        eqn_res = self.stacked_equation_residual(all_u_params,P_params)
        data_res = self.stacked_datafit_residual(all_u_params)
        return jnp.hstack([
            jnp.sqrt(self.datafit_weight) * data_res/jnp.sqrt(len(data_res)),
            eqn_res/jnp.sqrt(len(eqn_res))
            ])
    
    @partial(jit, static_argnames=['self'])
    def jac(self,full_params):
        """This is to allow for custom jacobian operators, and a choice
        between forward and reverse mode autodiff.
        In particular, we may want batched map based JVP
        """
        return self.jacobian_operator(self.F)(full_params)
        
    #usually jacrev is faster than jacfwd on examples I've tested
    # jac = jit(jacrev(F,argnums = 1),static_argnames='self')

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
    
class SharedOperatorPDEModel():
    def __init__(
        self,
        operator_model,
        u_model:CholInducedRKHS,
        observation_points:tuple,
        observation_values:tuple,
        collocation_points:tuple,
        feature_operators:tuple,
        rhs_forcing_values:Optional[tuple]=None,
        rhs_operator=None,
        datafit_weight = 10,
        jacobian_operator = jax.jacrev,
        num_P_operator_params = None
    ):
        self.num_functions = len(observation_points)
        if rhs_forcing_values is None:
            rhs_forcing_values = tuple(jnp.zeros(len(col_points)) for col_points in collocation_points)
        self.u_model = u_model
        self.operator_model = operator_model
        self.residual_dimension = sum([len(a) for a in observation_points]) + sum([len(a) for a in collocation_points])

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
        self.jacobian_operator = jacobian_operator

        self.rhs_operator = rhs_operator

        self.stacked_observation_values = jnp.hstack(observation_values)
        self.stacked_collocation_rhs = jnp.hstack(rhs_forcing_values)
        
        #Precompute parameter indices to pull out different parameter blocks
        self.total_parameters = len(observation_points)*u_model.num_params + self.num_operator_params
        
        #Assume we put the parameters for the operators at the start of the flattened parameter set
        self.operator_model_indices = jnp.arange(self.total_parameters - self.num_operator_params,self.total_parameters)

        #Compute the start and end indices of the parameter sets for u_models
        u_param_inds = jnp.cumsum(jnp.array([0]+[self.u_model.num_params]*self.num_functions))
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
    
    def apply_rhs_op_single(self,u_params,evaluation_points):
        """
        Applies the previously specified fixed RHS operator to all of 
        u_model parameterized by u_params at evalutaion_points
        """
        if self.rhs_operator is None:
            return jnp.zeros(len(evaluation_points))
        else:
            op_evaluation = self.u_model.evaluate_operators((self.rhs_operator,),evaluation_points,u_params)
            return op_evaluation.flatten()
    
    @partial(jit, static_argnames=['self'])
    def single_eqn_features(
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
    
    def equation_residual_single(
        self,
        u_params,
        P_params,
        evaluation_points,
        rhs_forcing
    ):
        """
        Computes the equation residual associated to 
        u_model parametrized by u_params,
        P_operator_model parametrized by P_params

        based on (rhs_operator(u) + rhs_forcing) at evaluation_points
        """
        features = self.single_eqn_features(u_params,evaluation_points)
        P_predictions = self.operator_model.predict(features,P_params)
        rhs_values = self.apply_rhs_op_single(u_params,evaluation_points) + rhs_forcing
        return rhs_values - P_predictions

    def stacked_equation_residual(
        self,
        all_u_params:tuple,
        P_params:jax.Array
    ):
        return jnp.hstack(
            [
                self.equation_residual_single(
                    u_params,P_params,eval_points,rhs_forcing
                    ) for 
                    u_params,eval_points,rhs_forcing in zip(
                        all_u_params,
                        self.collocation_points,
                        self.rhs_forcing_values
                        )
                ]
                )

    def datafit_residual_single(
        self,
        u_params,
        obs_points,
        obs_vals,
        ):
        return obs_vals - self.u_model.point_evaluate(obs_points,u_params)
    
    def stacked_datafit_residual(
        self,
        all_u_params:tuple
    ):
        return jnp.hstack(
            [
                self.datafit_residual_single(u_params,obs_points,obs_vals)
                for u_params,obs_points,obs_vals in zip(all_u_params,self.observation_points,self.observation_values)
                ]
        )
    
    def F(self,full_params):
        all_u_params = self.get_u_params(full_params)
        P_params = self.get_P_params(full_params)
        eqn_res = self.stacked_equation_residual(all_u_params,P_params)
        data_res = self.stacked_datafit_residual(all_u_params)
        return jnp.hstack([
            jnp.sqrt(self.datafit_weight/len(data_res)) * data_res,
            jnp.sqrt(1/len(eqn_res)) * eqn_res
            ])
    
    @partial(jit, static_argnames=['self'])
    def jac(self,full_params):
        """This is to allow for custom jacobian operators, and a choice
        between forward and reverse mode autodiff.
        In particular, we may want batched map based JVP
        """
        return self.jacobian_operator(self.F)(full_params)
        
    #usually jacrev is faster than jacfwd on examples I've tested
    # jac = jit(jacrev(F,argnums = 1),static_argnames='self')

    def loss(self,full_params):
        """
        TODO: Include the regularization term here instead of in EqnModel"""
        return (1/2) * jnp.linalg.norm(self.F(full_params))**2
    
    def damping_matrix(self,full_params,nugget = 1e-3):
        u_params = self.get_u_params(full_params)
        grid_feats = jnp.vstack([self.single_eqn_features(u_params,eval_points) for 
                                 u_params,eval_points in zip(
                                    u_params,self.collocation_points)])
        dmat = block_diag(
            *([self.u_model.get_damping()]*self.num_functions+[self.operator_model.rkhs_mat(grid_feats)])
            )
        dmat= dmat + nugget*diagpart(dmat)
        return dmat


def build_batched_jac_func(batch_size= 500):
    def jac(F):
        def eval_jac(x):
            fval,Jfunc = jax.linearize(F,x)
            J = jax.lax.map(Jfunc,jnp.identity(len(x)), batch_size=batch_size).T
            return J
        return eval_jac
    return jac
