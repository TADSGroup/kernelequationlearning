import jax.numpy as jnp

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