from .BaseAcquistionFunction import BaseAcquistionFunction
from typing import Any
import numpy as np


class ExpectedImprovement(BaseAcquistionFunction):
    def evaluate(self, X: np.ndarray, surrogate_model: Any) -> np.ndarray:
        """
        Evaluate the Expected Improvement acquisition function at the given input points using the surrogate model.

        Parameters
        ----------
        X : np.ndarray
            The input points to evaluate, of shape (n_samples, n_features).
        surrogate_model : Any
            The surrogate model used to provide predictions and uncertainty estimates.

        Returns
        -------
        np.ndarray
            The evaluated values of the Expected Improvement acquisition function at the given points.
        """
        pass