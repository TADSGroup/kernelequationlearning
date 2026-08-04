import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices()[0])
import jax.numpy as jnp
import numpy as np

from parabolic_data_utils import build_burgers_data, GP_Sampler_1D_Pinned
from Kernels import get_gaussianRBF
from evaluation_metrics import get_nrmse

N_OBS = [10, 30, 50, 100, 200, 300, 400, 500, 600]
ARCHS = ['2x64', '3x128', '4x256']


def get_ground_truth(ic_seed):
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
    return u_true_function, ut_true_function, interp


def evaluate_Phat(coef, interp, fine_grid):
    def u_true_function(x):
        return interp(x[:, 0], x[:, 1], grid=False)

    def ux_true_function(x):
        return interp.partial_derivative(0, 1)(x[:, 0], x[:, 1], grid=False)

    def uxx_true_function(x):
        return interp.partial_derivative(0, 2)(x[:, 0], x[:, 1], grid=False)

    u = u_true_function(fine_grid)
    ux = ux_true_function(fine_grid)
    uxx = uxx_true_function(fine_grid)
    Phi = jnp.stack([
        jnp.ones_like(u), u, ux, uxx,
        u * u, u * ux, u * uxx,
        ux * ux, ux * uxx, uxx * uxx,
    ], axis=1)
    return Phi @ coef


def get_arch_error(arch, n_obs, ic_seed):
    data_out = np.load(f'data_out_PINNSR_{arch}.npy', allow_pickle=True)
    for element in data_out:
        if element[0] == n_obs and element[1] == 0 and not isinstance(element[2], str):
            coef = jnp.array(element[2].flatten().astype(np.float64))
            u_pred = jnp.array(element[3].flatten().astype(np.float64))
            break
    else:
        return np.nan, np.nan

    u_true_function, ut_true_function, interp = get_ground_truth(ic_seed)

    num_fine_grid = 100
    t_fine, x_fine = np.meshgrid(
        np.linspace(0, 1, num_fine_grid + 4)[2:-2],
        np.linspace(0, 1, num_fine_grid + 4)[2:-2]
    )
    tx_fine_int = np.vstack([t_fine.flatten(), x_fine.flatten()]).T

    u_true = u_true_function(tx_fine_int)
    error_u = get_nrmse(u_true, u_pred)

    Phat_u = evaluate_Phat(coef, interp, tx_fine_int)
    u_t = ut_true_function(tx_fine_int)
    error_Phat = get_nrmse(u_t, Phat_u)

    return float(error_u), float(error_Phat)


if __name__ == '__main__':
    errors = {arch: {'u': [], 'P': []} for arch in ARCHS}
    for i, n_obs in enumerate(N_OBS):
        # ic_seed = i matches generate_data.py's varyICs convention (IC varies across the sweep)
        for arch in ARCHS:
            eu, ep = get_arch_error(arch, n_obs, ic_seed=i)
            errors[arch]['u'].append(eu)
            errors[arch]['P'].append(ep)
            print(f'n_obs={n_obs} (ic_seed={i}) arch={arch}: u_error={eu:.4e} P_error={ep:.4e}')
    np.save('errors_archsweep.npy', errors, allow_pickle=True)
    print('wrote errors_archsweep.npy')
