import jax.numpy as jnp
import numpy as np



class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit_transform(self, X):
        X_min = jnp.min(X, axis=0)
        X_max = jnp.max(X, axis=0)
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (X_max - X_min)
        self.min_ = self.feature_range[0] - X_min * self.scale_
        return self.transform(X)

    def transform(self, X):
        return X * self.scale_ + self.min_

    def inverse_transform(self, X_scaled):
        return (X_scaled - self.min_) / self.scale_
def rel_mse(true, pred, root = True):

    '''
    true: Array of ground truth. 
    pred: Array of predictions. 
    root: If True, it computes the relative root mse.  
    '''
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(true))
    error = num/den
    if root:
        error = np.sqrt(error)
    return error

def needed_regularity_Matern(k,d):
  """
    Parameters
    ----------
    k : int
        Level of desired regularity in C^k.
    d : int
        Dimension of the domain of the DE.

    Returns
    -------
    nu
        Nice regularity class in Matern kernel(nu = s - d/2).
  """
  s = np.floor(d/2 + k) + 1
  nu = s - d/2
  t = nu - 0.5
  if t.is_integer():
    print('Needed regularity in Matern class is {}/2'.format(int(2*nu)))
    return nu
  else:
    print('Needed regularity in Matern class is {}/2'.format(int((2*(nu+0.5)))))
    return nu + 1/2