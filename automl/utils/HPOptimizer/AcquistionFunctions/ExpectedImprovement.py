# ExpectedImprovement Class
from .BaseAcquistionFunction import BaseAcquistionFunction
from typing import Any
import numpy as np
from scipy.stats import norm

class ExpectedImprovement(BaseAcquistionFunction):
    def evaluate(self, X: np.ndarray, surrogate_model: Any) -> np.ndarray:
        """
        Evaluate the Expected Improvement acquisition function at the given input points using the surrogate model.
        """
        y_pred, sigma = surrogate_model.predict(X, return_std=True)
        best_y = np.max(y_pred)
        
        with np.errstate(divide='warn'):
            improvement = y_pred - best_y
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0  # If sigma is zero, EI is zero.

        return ei
