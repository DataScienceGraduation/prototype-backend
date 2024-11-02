from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class SurrogateModel(ABC):
    """
    Abstract base class for all surrogate models used in Bayesian optimization.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the surrogate model to the given training data.

        Parameters
        ----------
        X : np.ndarray
            The training data features, of shape (n_samples, n_features).
        y : np.ndarray
            The target values, of shape (n_samples,).
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the surrogate model, with optional standard deviation.

        Parameters
        ----------
        X : np.ndarray
            The input data for making predictions, of shape (n_samples, n_features).
        return_std : bool, optional
            Whether to return the standard deviation of the predictions (default is False).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The predicted mean and, if return_std is True, the standard deviation of the predictions.
        """
        pass
