from .BaseSurrogateModel import BaseSurrogateModel
from typing import Tuple
import numpy as np

class GaussianProcessSurrogate(BaseSurrogateModel):
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        pass