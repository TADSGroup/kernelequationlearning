
import matplotlib.pyplot as plt



def plot_obs(xy_fine, xy_all, xy_obs, vmapped_funcs, title = None):
    """
        Plots the up to three functions the observed values. 

        Args:
            xy_fine (Array): Pairs of points in fine grid.
            xy_all (Array): Pairs of ghosts points.
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
        axsi = axs[i].tricontourf(xy_fine[:,0],xy_fine[:,1],vmapped_funcs[i](xy_fine),cmap = 'Blues')
        plt.colorbar(axsi, ax = axs[i])
        axs[i].scatter(xy_obs[i][:,0],xy_obs[i][:,1],c='red', s = 50)
        axs[i].scatter(xy_all[:,0],xy_all[:,1],c='black',s = 10)
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('y')
        axs[i].set_xlim(-0.1,1.1)
        axs[i].set_ylim(-0.1,1.1)
    plt.show()
    return None

def plot_compare_error(xy_fine, xy_all, xy_obsi, vmapped_func_pred, vmapped_func_true, title = None):
    """
        Plots predicted, true function and their abs error. 

        Args:
            xy_fine (Array): Pairs of points in fine grid.
            xy_all (Array): Pairs of ghosts points.
            xy_obsi (Array): Pairs of observed points by true function.
            vmapped_funcs (list): List of vectorized functions with vmap.
            title (str): Title name.

        Returns:
            Plots predicted and true functions and their abs error.   

    """ 
    
    plt.figure(figsize=(13,4))
    plt.suptitle(title)
    plt.subplot(1,3,1)
    plt.title("Learned u")
    pred_vals = vmapped_func_pred(xy_fine)
    plt.tricontourf(xy_fine[:,0],xy_fine[:,1], pred_vals, 200)

    plt.subplot(1,3,2)
    plt.title("True u")
    true_vals = vmapped_func_true(xy_fine)
    plt.tricontourf(xy_fine[:,0],xy_fine[:,1], true_vals, 200)

    plt.subplot(1,3,3)
    plt.title("Error")
    plt.tricontourf(xy_fine[:,0],xy_fine[:,1], true_vals-pred_vals,250)
    plt.colorbar()
    plt.scatter(xy_obsi[:,0],xy_obsi[:,1],c='red',s = 50)
    plt.scatter(xy_all[:,0],xy_all[:,1],c='black',s = 4)
    plt.show()
    return None