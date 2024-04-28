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

def check_OperatorPDEModel(
        u_models:list,
        observation_points:list,
        observation_values:list,
        collocation_points:list,
        rhs_values:list,
    ):
    try:
        assert len(u_models)==len(observation_points)==len(observation_values)==len(collocation_points)==len(rhs_values), "Data dimensions don't match up"
    except AssertionError as message: 
        print(message)
        print("u_models given: ",len(u_models))
        print("sets of observation_points given: ",len(observation_points))
        print("sets of observation_values given: ",len(observation_values))
        print("sets of collocation_points given: ",len(collocation_points))
        print("sets of rhs_values given: ",len(rhs_values))
        print("These should all match")
    for obs_points,obs_vals in zip(observation_points,observation_values):
        assert len(obs_points)==len(obs_vals), "Number of observation locations don't match number of observed values"
    for col_points,rhs_vals in zip(collocation_points,rhs_values):
        assert len(col_points)==len(rhs_vals), "Number of collocation points don't match number of rhs values"



class OperatorPDEModel():
    def __init__(
        self,
        operator_model,
        u_models:list,
        observation_points:list,
        observation_values:list,
        collocation_points:list,
        rhs_values:list,
        feature_operators:list,
        datafit_weight = 10,
    ):
        check_OperatorPDEModel(observation_points,observation_values,collocation_points,rhs_values)
        self.u_models = u_models
        self.operator_model = operator_model
        self.observation_points = observation_points
        self.observation_values = observation_values
        self.collocation_points = collocation_points
        self.rhs_values = rhs_values
        self.feature_operators = feature_operators
    
        self.stacked_observation_values = jnp.hstack(observation_values)
        self.stacked_collocation_rhs = jnp.hstack(rhs_values)
        
        #Assume we put the parameters for the operators at the start of the flattened parameter set
        self.operator_model_indices = jnp.arange(0,self.operator_model.num_params)
        #Compute the start and end indices of the parameter sets for u_models
        u_param_inds = self.operator_model.num_params+jnp.cumsum(jnp.array([0]+[model.num_params for model in u_models]))
        self.u_indexing = [
            jnp.arange(p,q) for p,q in zip(u_param_inds[:-1],u_param_inds[1:])
        ]

    def get_P_params(self,all_params):
        return all_params[self.operator_model_indices]
    
    def get_u_params(self,all_params):
        return [all_params[ind_set] for ind_set in self.u_indexing]

    def get_grid_features(
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

    @jit
    def datafit_residual(self,all_params):
        all_u_parameters = self.get_u_params(all_params)
        obs_preds = jnp.hstack(
            [
                model.point_evaluate(obs_points,u_params) 
                for model,obs_points,u_params in zip(
                    self.u_models,
                    self.observation_points,
                    all_u_parameters)
                ])
        return self.stacked_observation_values - obs_preds 





# class EqnModel():
#     datafit_weight = 40

#     @jit
#     def get_grid_features(u_params):
#         u_params1, u_params2, u_params3 = split_u_params(u_params)
#         evaluation_u1 = u1_model.evaluate_operators(feature_operators,xy_int,u_params1)
#         evaluation_u2 = u2_model.evaluate_operators(feature_operators,xy_int,u_params2)
#         evaluation_u3 = u3_model.evaluate_operators(feature_operators,xy_int,u_params3)
#         grid_features_u1 = evaluation_u1.reshape(len(xy_int),len(feature_operators),order = 'F')
#         grid_features_u2 = evaluation_u2.reshape(len(xy_int),len(feature_operators),order = 'F')
#         grid_features_u3 = evaluation_u3.reshape(len(xy_int),len(feature_operators),order = 'F')
#         grid_features_u = jnp.vstack([grid_features_u1, grid_features_u2, grid_features_u3])
#         full_features = jnp.hstack([jnp.tile(xy_int.T,3).T,grid_features_u])
#         return full_features
    
#     @jit
#     def get_grid_target(u_params):
#         #return jnp.ones(len(xy_int))
#         return jnp.concatenate([eval_rhs1(xy_int), eval_rhs2(xy_int), eval_rhs3(xy_int)])
    
#     @jit
#     def eval_obs_points(u_params):
#         u_params1, u_params2, u_params3 = split_u_params(u_params)
#         return jnp.concatenate([u1_model.point_evaluate(xy_obs1,u_params1),
#                                u2_model.point_evaluate(xy_obs2,u_params2),
#                                u3_model.point_evaluate(xy_obs3,u_params3)])
    
#     @jit
#     def datafit_residual(u_params):
#         obs_preds = EqnModel.eval_obs_points(u_params)
#         return u_obs - obs_preds
    
#     @jit
#     def equation_residual(full_params):
#         u_model_num_params = u1_model.num_params + u2_model.num_params + u3_model.num_params
#         u_params = full_params[:u_model_num_params]
#         P_params = full_params[u_model_num_params:]
#         P_features = EqnModel.get_grid_features(u_params)
#         P_model_preds = P_model.predict(P_features,P_params)
#         ugrid_target = EqnModel.get_grid_target(u_params)
#         return (ugrid_target - P_model_preds)
    
#     @jit
#     def F(full_params):
#         u_model_num_params = u1_model.num_params + u2_model.num_params + u3_model.num_params
#         u_params = full_params[:u_model_num_params]
#         eqn_res = EqnModel.equation_residual(full_params)
#         data_res = EqnModel.datafit_residual(u_params)
#         return jnp.hstack([
#             EqnModel.datafit_weight * data_res/jnp.sqrt(len(data_res)),
#             eqn_res/jnp.sqrt(len(eqn_res))
#             ])
    
#     jac = jit(jacrev(F))

#     def loss(full_params):
#         return jnp.linalg.norm(EqnModel.F(full_params))**2
    
#     @jit
#     def damping_matrix(full_params):
#         u_model_num_params = u1_model.num_params + u2_model.num_params + u3_model.num_params
#         u_params = full_params[:u_model_num_params]
#         grid_feats = EqnModel.get_grid_features(u_params)
#         kmat_P = P_model.kernel_function(grid_feats,grid_feats)
#         dmat = block_diag(
#             u1_model.kmat+1e-3 * diagpart(u1_model.kmat),
#             u2_model.kmat+1e-3 * diagpart(u2_model.kmat),
#             u3_model.kmat+1e-3 * diagpart(u3_model.kmat),
#             1e-3 * (kmat_P+1e-3 * jnp.identity(len(kmat_P)))
#         )
#         return dmat
