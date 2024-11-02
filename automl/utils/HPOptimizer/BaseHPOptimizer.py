from abc import ABC, abstractmethod
# create an enum for the different types of tasks
from enum import Enum
from pandas import DataFrame
from sklearn.base import ClassifierMixin
from typing import Tuple

from ..Enums import Task, Metric

class BaseHPOptimizer:
    """
    Base class for all hyperparameter optimizers

    Parameters
    ----------
    task: Task
        The type of task to perform
    time_budget: int
        The time budget for the optimization in seconds
    metric: Metric
        The metric to optimize
    fast_mode: bool
        Whether to use a faster mode for optimization
    verbose: bool
        Whether to print out information during optimization
    """
    def __init__(self, task: Task, time_budget: int, metric: Metric = Metric.ACCURACY, fast_mode: bool = False, verbose: bool = False):
        self.task = task
        self.time_budget = time_budget
        self.metric = metric
        self.fast_mode = fast_mode
        self.verbose = verbose
        self.optimal_hyperparameters = {}
        self.optimal_model = None

    @abstractmethod
    def fit(self, df: DataFrame, target_variable: str, fast_mode: bool = False) -> None: 
        pass

    def getOptimalModel(self) -> ClassifierMixin:
        return self.optimal_model
    
    def getMetric(self) -> Metric:
        return self.metric
    
    def getMetricValue(self) -> float:
        return self.metric_value
    