import matplotlib.pyplot as plt
import jax.numpy as jnp

def analyze_hessian(H,g):
    H_vals,H_vecs = jnp.linalg.eigh(H)
    g_H = H_vecs.T@g
    gh_energy = g_H**2
    plt.figure(figsize = (12,7))
    plt.subplot(2,2,1)
    plt.title("gradient components in positive eig directions")
    plt.scatter(H_vals[H_vals>0],gh_energy[H_vals>0],c='black',s = 6)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("lam")
    plt.ylabel("gradient component squared")

    plt.subplot(2,2,2)

    plt.title("gradient components in negative eig directions")
    plt.scatter(-1 * H_vals[H_vals<0],gh_energy[H_vals<0],c='black',s = 6)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("-1 * lam")
    plt.ylabel("gradient component squared")
    plt.subplot(2,2,3)

    plt.title("Hessian weighted gradient \n components in positive eig directions")
    keep_inds = H_vals>0
    plt.scatter(H_vals[keep_inds],H_vals[keep_inds] * gh_energy[keep_inds],c='black',s = 6)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("lam")
    plt.ylabel("gradient component squared")

    plt.subplot(2,2,4)

    plt.title("Hessian weighted gradient \n components in negative eig directions")
    keep_inds = H_vals<0
    plt.scatter(-1 * H_vals[keep_inds],-1 * H_vals[keep_inds] * gh_energy[keep_inds],c='black',s = 6)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("-1 * lam")
    plt.ylabel("gradient component squared")
    plt.tight_layout()
    plt.show()
    print("Most negative eigenvalue ",jnp.min(H_vals))