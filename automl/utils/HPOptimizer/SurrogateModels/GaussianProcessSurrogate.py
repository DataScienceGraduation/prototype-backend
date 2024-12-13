# GaussianProcessSurrogate Class
from .BaseSurrogateModel import BaseSurrogateModel
from typing import Tuple
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class GaussianProcessSurrogate(BaseSurrogateModel):
    def __init__(self):
        self.model = GaussianProcessRegressor(kernel=C(1.0) * RBF(length_scale=1.0), n_restarts_optimizer=10)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        return self.model.predict(X, return_std=return_std)
