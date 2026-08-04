import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices()[0])
import jax.numpy as jnp
import numpy as np

from parabolic_data_utils import (
    build_burgers_data,
    build_tx_grid_chebyshev,
    setup_problem_data,
    GP_Sampler_1D_Pinned
)
from Kernels import get_gaussianRBF

# Same IC every point -- fixedICs
IC_SEED = 2
N_COLL_T, N_COLL_X = 26, 31
N_OBS = [10, 30, 50, 100, 200, 300, 400, 500, 600]


def get_data_for_pinns(n_obs, obs_seed, ic_seed):
    kernel_GP = get_gaussianRBF(0.2)
    xy_pts = jnp.linspace(0, 1, 50)
    u0_true_function = GP_Sampler_1D_Pinned(
        num_samples=1, X=xy_pts, smooth=2, kernel=kernel_GP, reg=1e-8, seed=ic_seed
    )
    vec_u0_true_function = np.vectorize(u0_true_function[0])

    kappa, alpha = 0.01, 1.
    u_true_function, ut_true_function, interp, _, _ = build_burgers_data(
        func_u0=vec_u0_true_function, kappa=kappa, alpha=alpha,
        k_timestep=0.0001, n_finite_diff=1999
    )

    tx_int, tx_bdy = build_tx_grid_chebyshev([0, 1], [0, 1], N_COLL_T, N_COLL_X, alpha=0.5)
    tx_all, tx_obs = setup_problem_data(
        tx_int, tx_bdy, n_obs, jax.random.PRNGKey(obs_seed), times_to_observe=(0,)
    )

    t_fine, x_fine = jnp.meshgrid(jnp.linspace(0, 1, 300), jnp.linspace(0, 1, 300))
    tx_fine = jnp.vstack([t_fine.flatten(), x_fine.flatten()]).T

    tx_train = tx_obs
    u_train = u_true_function(tx_train)
    tx_val = jax.random.choice(
        key=jax.random.PRNGKey(0), a=tx_fine,
        shape=(int(jnp.ceil(tx_train.shape[0] * (1 - 0.8) / 0.8)),), replace=False
    )
    u_val = u_true_function(tx_val)
    lb = tx_fine.min(0)
    ub = tx_fine.max(0)

    # PINN-SR expects (x, t) column order, KEQL builds (t, x) -- swap
    tx_train = tx_train.at[:, [1, 0]].set(tx_train[:, [0, 1]])
    tx_val = tx_val.at[:, [1, 0]].set(tx_val[:, [0, 1]])
    tx_all = tx_all.at[:, [1, 0]].set(tx_all[:, [0, 1]])
    tx_all = jnp.vstack([tx_all, tx_train])

    return [tx_train, u_train, tx_all, tx_val, u_val, lb, ub]


if __name__ == '__main__':
    data_in_PINNSR = []
    for i, n_obs in enumerate(N_OBS):
        data_list = get_data_for_pinns(n_obs, obs_seed=0, ic_seed=IC_SEED)
        data_list = list(map(np.array, data_list))
        data_in_PINNSR.append([n_obs, 0, data_list])
    np.save('data_in_PINNSR.npy', np.array(data_in_PINNSR, dtype=object), allow_pickle=True)
    print(f'wrote data_in_PINNSR.npy with {len(data_in_PINNSR)} elements (ic_seed={IC_SEED} for all)')
