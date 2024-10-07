
import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_obs(xy_fine, xy_all, xy_obs, vmapped_funcs, title = None):
    """
        Plots the up to three functions the observed values. 

        Args:
            xy_fine (Array): Pairs of points in fine grid.
            xy_all (list): Pairs of ghosts points.
            xy_obs (list): List of pairs of observed points per function.
            vmapped_funcs (list): List of vectorized functions with vmap.
            title (str): Title name.

        Returns:
            None: Plots functions.   

    """

    n = len(vmapped_funcs)
    if n > 3:
        n = 3
    fig, axs = plt.subplots(figsize = (20,4), nrows=1 , ncols = n, sharex = True, sharey = True)
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    fig.suptitle(title)
    for i in range(n):
        # Contour plot in blue
        # axsi = axs[i].tricontourf(xy_fine[:,0],xy_fine[:,1],vmapped_funcs[i](xy_fine),cmap = 'Blues')
        axsi = axs[i].tricontourf(xy_fine[:,0],xy_fine[:,1],vmapped_funcs[i](xy_fine))
        plt.colorbar(axsi, ax = axs[i])
        axs[i].scatter(xy_obs[i][:,0],xy_obs[i][:,1],c='red', s = 50)
        axs[i].scatter(xy_all[i][:,0],xy_all[i][:,1],c='black',s = 10)
        axs[i].set_ylabel(' ')
        axs[i].set_xlabel(' ')
        axs[i].set_xlim(-0.1,1.1)
        axs[i].set_ylim(-0.1,1.1)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        # axs[i].get_xaxis().set_ticks([])
        # axs[i].get_yaxis().set_ticks([])
    plt.show()
    return None

def plot_contours(xy_fine, vmapped_funcs, title = None):
    """
        Plots the up to three functions the observed values. 

        Args:
            xy_fine (Array): Pairs of points in fine grid.
            vmapped_funcs (list): List of vectorized functions with vmap.
            title (str): Title name.

        Returns:
            None: Plots functions.   

    """

    n = len(vmapped_funcs)
    if n > 3:
        n = 3
    fig, axs = plt.subplots(figsize = (20,4), nrows=1 , ncols = n, sharex = True, sharey = True)
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    fig.suptitle(title)
    for i in range(n):
        # Contour plot in blue
        # axsi = axs[i].tricontourf(xy_fine[:,0],xy_fine[:,1],vmapped_funcs[i](xy_fine),cmap = 'Blues')
        axsi = axs[i].tricontourf(xy_fine[:,0],xy_fine[:,1],vmapped_funcs[i](xy_fine))
        plt.colorbar(axsi, ax = axs[i])
        axs[i].set_ylabel(' ')
        axs[i].set_xlabel(' ')
        axs[i].set_xlim(-0.1,1.1)
        axs[i].set_ylim(-0.1,1.1)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        # axs[i].get_xaxis().set_ticks([])
        # axs[i].get_yaxis().set_ticks([])
    plt.show()
    return None

def plot_obs_parabolic(xy_fine, xy_all, xy_obs, vmapped_funcs, title = None):
    """
        Plots the up to three functions the observed values. 

        Args:
            xy_fine (Array): Pairs of points in fine grid.
            xy_all (list): Pairs of ghosts points.
            xy_obs (list): List of pairs of observed points per function.
            vmapped_funcs (list): List of vectorized functions with vmap.
            title (str): Title name.

        Returns:
            None: Plots functions.   

    """

    n = len(vmapped_funcs)
    if n > 3:
        n = 3
    fig, axs = plt.subplots(figsize = (20,4), nrows=1 , ncols = n, sharex = True, sharey = True)
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    fig.suptitle(title)
    for i in range(n):
        # Contour plot in blue
        axsi = axs[i].tricontourf(xy_fine[:,0],xy_fine[:,1],vmapped_funcs[i](xy_fine),50)
        plt.colorbar(axsi, ax = axs[i])
        axs[i].scatter(xy_obs[i][:,0],xy_obs[i][:,1],c='red', s = 50, alpha = 0.2)
        axs[i].scatter(xy_all[i][:,0],xy_all[i][:,1],c='black',s = 10, alpha = 0.1)
        axs[i].set_xlabel('t')
        axs[i].set_ylabel('x')
        axs[i].set_xlim(-0.1,1.1)
        axs[i].set_ylim(-0.1,1.1)
    plt.show()
    return None


def plot_init_final_parabolic(grid, vmapped_funcs, title = None):
    """
        Plots the up to three functions the observed values. 

        Args:
            grid (1D Array): Fine grid to plot every function.
            vmapped_funcs (list): List of vectorized functions with vmap.
            title (str): Title name.

        Returns:
            None: Plots functions.   

    """
    num_grid = len(grid)
    n = len(vmapped_funcs)
    if n > 3:
        n = 3
    fig, axs = plt.subplots(figsize = (20,4), nrows=1 , ncols = n, sharex = True, sharey = True)
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    fig.suptitle(title)
    for i in range(n):
        axs[i].plot(grid,vmapped_funcs[i](jnp.vstack([0.0*jnp.ones(num_grid),grid]).T),label = 't=0')
        axs[i].plot(grid,vmapped_funcs[i](jnp.vstack([1.*jnp.ones(num_grid),grid]).T),label = 't=1')
        axs[i].set_xlabel('t')
        axs[i].set_ylabel('x')
        axs[i].legend()
    plt.show()
    return None

def plot_compare_error(
        xy_fine, xy_all, xy_obsi, 
        vmapped_func_pred, 
        vmapped_func_true, title = None):
    """
        Plots predicted, true function and their abs error. 

        Args:
            xy_fine (Array): Pairs of points in fine grid.
            xy_all (Array): Pairs of ghosts points.
            xy_obsi (Array): Pairs of observed points by true function.

        Returns:
            Plots predicted and true function and their abs error.   

    """ 
    pred_vals = vmapped_func_pred(xy_fine)
    true_vals = vmapped_func_true(xy_fine)
    plot_compare_error_values(
        xy_fine,xy_all,xy_obsi,pred_vals,true_vals,title
    )
    return None

def plot_compare_error_values(
        xy_fine, xy_all, xy_obsi, 
        pred_vals, 
        true_vals, title = None):
    """
        Plots predicted, true function and their abs error. 

        Args:
            xy_fine (Array): Pairs of points in fine grid.
            xy_all (Array): Pairs of ghosts points.
            xy_obsi (Array): Pairs of observed points by true function.

        Returns:
            Plots predicted and true function and their abs error.   

    """ 
    x_bounds = jnp.min(xy_fine[:,0]),jnp.max(xy_fine[:,0])
    y_bounds = jnp.min(xy_fine[:,1]),jnp.max(xy_fine[:,1])

    plt.figure(figsize=(13,4))
    plt.suptitle(title)
    plt.subplot(1,3,1)
    plt.title("Learned u")
    plt.tricontourf(xy_fine[:,0],xy_fine[:,1], pred_vals, 200)
    plt.colorbar()
    plt.xlim(x_bounds[0] - 0.05,x_bounds[1] + 0.05)
    plt.ylim(y_bounds[0] - 0.05,y_bounds[1] + 0.05)

    plt.subplot(1,3,2)
    plt.title("True u")
    plt.tricontourf(xy_fine[:,0],xy_fine[:,1], true_vals, 200)
    plt.colorbar()
    plt.xlim(x_bounds[0] - 0.05,x_bounds[1] + 0.05)
    plt.ylim(y_bounds[0] - 0.05,y_bounds[1] + 0.05)

    plt.subplot(1,3,3)
    plt.title("Error")
    plt.tricontourf(xy_fine[:,0],xy_fine[:,1], true_vals-pred_vals,250)
    plt.colorbar()
    if xy_obsi is not None:
        plt.scatter(xy_obsi[:,0],xy_obsi[:,1],c='red',s = 50)
    if xy_all is not None:
        plt.scatter(xy_all[:,0],xy_all[:,1],c='black',s = 4)
    plt.xlim(x_bounds[0] - 0.05,x_bounds[1] + 0.05)
    plt.ylim(y_bounds[0] - 0.05,y_bounds[1] + 0.05)
    plt.show()
    return None


def plot_input_data(
    obs_points,
    all_points,
    func_to_plot,
    fine_grid,
    xlabel = 't',
    ylabel = 'x',
    include_collocation = True
):
    plt.figure(figsize=(8,5))
    x_bounds = jnp.min(fine_grid[:,0]),jnp.max(fine_grid[:,0])
    y_bounds = jnp.min(fine_grid[:,1]),jnp.max(fine_grid[:,1])
    plt.tricontourf(fine_grid[:,0],fine_grid[:,1],func_to_plot(fine_grid),50)
    plt.colorbar()
    plt.scatter(obs_points[:,0],obs_points[:,1],c='red', s = 50,label = "Function Value Observed")
    if include_collocation is True:
        plt.scatter(all_points[:,0],all_points[:,1],c='black',s = 5,label = "Collocation Point")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(x_bounds[0] - 0.05,x_bounds[1] + 0.05)
    plt.ylim(y_bounds[0] - 0.05,y_bounds[1] + 0.05)
    plt.figlegend(loc = 'upper center')



def compare_values(x,y):
    plt.plot([jnp.min(x),jnp.max(x)],
            [jnp.min(x),jnp.max(x)],c = 'blue',lw = 0.8)
    plt.scatter(x,y,c = 'black',s = 4)
