from abc import ABC, abstractmethod
import numpy as np
from typing import Any

class BaseAcquistionFunction(ABC):
    """
    Abstract base class for all acquisition functions used in Bayesian optimization.
    """

    @abstractmethod
    def evaluate(self, X: np.ndarray, surrogate_model: Any) -> np.ndarray:
        """
        Evaluate the acquisition function at the given input points using the surrogate model.

        Parameters
        ----------
        X : np.ndarray
            The input points to evaluate, of shape (n_samples, n_features).
        surrogate_model : Any
            The surrogate model used to provide predictions and uncertainty estimates.

        Returns
        -------
        np.ndarray
            The evaluated values of the acquisition function at the given points.
        """
        pass
