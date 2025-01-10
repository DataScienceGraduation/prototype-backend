# ExpectedImprovement Class
from .BaseAcquistionFunction import BaseAcquistionFunction
from typing import Any
import numpy as np
from scipy.stats import norm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ExpectedImprovement(BaseAcquistionFunction):
    def evaluate(self, X: np.ndarray, surrogate_model: Any, best_observed: np.float64) -> np.ndarray:
        """
        Evaluate the Expected Improvement acquisition function at the given input points using the surrogate model.
        """
        logging.info("Evaluating Expected Improvement acquisition function")
        logging.info(f"type of X: {type(X)}")
        # convert X to a pandas DataFrame, X is a list of dictionaries
        X = np.array([list(x.values()) for x in X])

        logging.info(f'X: {X}')

        y_pred, sigma = surrogate_model.predict(X, return_std=True)
        best_y = max(y_pred)
        if best_observed > 0.1:
            best_y = best_observed
        
        with np.errstate(divide='warn'):
            improvement = y_pred - best_y
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0  # If sigma is zero, EI is zero.

        logging.info(f'Expected Improvement: {ei}')

        return ei
