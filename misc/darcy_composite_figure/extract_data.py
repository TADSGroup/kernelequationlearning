"""
One-off data extraction for the Darcy composite figure.

Reproduces the u/f fields from examples/Darcy/figs/plot.ipynb (same seeds,
GP sampler and permeability) and flattens the error dictionaries of
operator_learning/ and in_distribution/ into plain numpy arrays, so that
make_figure.py only needs numpy + matplotlib.

Run in the keql_legacy env:
    conda activate keql_legacy
    CUDA_VISIBLE_DEVICES="" python extract_data.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DARCY = os.path.join(REPO, 'examples', 'Darcy')
sys.path.insert(0, os.path.join(REPO, 'keql_tools'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.random import PRNGKey as pkey
import numpy as np

from Kernels import get_gaussianRBF
from data_utils import (
    get_xy_grid_pairs,
    GP_sampler,
    build_xy_grid,
    build_u_obs_all,
)

OBS_PTS_LIST = [2, 4, 8]
NUM_FUN_LIST = [2, 4, 8, 16, 32, 64, 128]
out = {'M': np.array(NUM_FUN_LIST)}


# ---------------------------------------------------------------- errors
def summarize(err, mthd, obs_pt, key):
    runs = [np.asarray(e) for e in np.asarray(err[mthd][f'{obs_pt}_obs'][key])]
    mean = np.array([np.mean(e) for e in runs])
    upper = np.array([np.nanmax(e) for e in runs])
    lower = np.array([np.nanmin(e) for e in runs])
    return mean, lower, upper


err_ol = np.load(os.path.join(DARCY, 'operator_learning', 'errors.npy'), allow_pickle=True).item()
err_id = np.load(os.path.join(DARCY, 'in_distribution', 'errors.npy'), allow_pickle=True).item()

for tag, err, key in [('ol', err_ol, 'i_opt'), ('id', err_id, 'i_dis')]:
    for mthd, short in [('2_mthd', '2step'), ('1_5_mthd', '1step')]:
        for obs_pt in OBS_PTS_LIST:
            mean, lower, upper = summarize(err, mthd, obs_pt, key)
            out[f'{tag}_{short}_{obs_pt}_mean'] = mean
            out[f'{tag}_{short}_{obs_pt}_lower'] = lower
            out[f'{tag}_{short}_{obs_pt}_upper'] = upper


# ---------------------------------------------------------------- fields
def A(xy):
    x = xy[0]
    y = xy[1]
    return jnp.exp(jnp.sin(jnp.cos(x) + jnp.cos(y)))


def get_rhs_darcy(u):
    def Agradu(xy):
        return A(xy) * jax.grad(u)(xy)

    def Pu(xy):
        return jnp.trace(jax.jacfwd(Agradu)(xy))
    return Pu


m = 5
run = 0
kernel_GP = get_gaussianRBF(0.5)
xy_pairs = get_xy_grid_pairs(50, 0, 1, 0, 1)
xy_fine = jnp.vstack(build_xy_grid([0, 1], [0, 1], 100, 100))
xy_int_single, xy_bdy_single = build_xy_grid([0, 1], [0, 1], 15, 15)
xy_ints = (xy_int_single,) * m
xy_bdys = (xy_bdy_single,) * m
xy_all = jnp.vstack([xy_int_single, xy_bdy_single])

out['xy_fine'] = np.asarray(xy_fine)
out['xy_coll'] = np.asarray(xy_all)

for obs_pts in OBS_PTS_LIST:
    seed = int(m * obs_pts * (run + 1))
    u_true_functions = tuple(GP_sampler(num_samples=m, X=xy_pairs, kernel=kernel_GP,
                                        reg=1e-12, seed=seed))
    u_s = tuple(jax.vmap(u) for u in u_true_functions)
    f_s = tuple(jax.vmap(get_rhs_darcy(u)) for u in u_true_functions)
    xy_obs, _ = build_u_obs_all([obs_pts] * m, xy_ints, xy_bdys, u_s, pkey(seed))
    out[f'u_{obs_pts}obs'] = np.stack([np.asarray(u(xy_fine)) for u in u_s])
    out[f'f_{obs_pts}obs'] = np.stack([np.asarray(f(xy_fine)) for f in f_s])
    out[f'xyobs_{obs_pts}obs'] = np.stack([np.asarray(x) for x in xy_obs])

np.savez(os.path.join(HERE, 'data.npz'), **out)
print('saved', sorted(out))
