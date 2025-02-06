from .BaseHPOptimizer import BaseHPOptimizer as BHPO
from ..Enums import Task, Metric
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import time
from pandas import DataFrame
from typing import Tuple
from sklearn.base import BaseEstimator


class RandomSearchHPOptimizer(BHPO):
    """
    Random search hyperparameter optimizer

    Parameters
    ----------
    task: Task
        The type of task to perform (classification or regression)
    time_budget: int
        The time budget for the optimization in seconds
    metric: Metric
        The metric to optimize
    fast_mode: bool
        Whether to use a faster mode for optimization
    verbose: bool
        Whether to print out information during optimization
    """
    METRIC_FUNCTIONS = {
        Metric.ACCURACY: accuracy_score,
        Metric.MSE: mean_squared_error,
        Metric.R2: r2_score,
        Metric.MAE: mean_absolute_error,
    }
    
    METRIC_HIGHER_BETTER = {
        Metric.ACCURACY: True,
        Metric.MSE: False,
        Metric.R2: True,
        Metric.MAE: False,
    }

    def __init__(self, task: Task, time_budget: int, metric: Metric = Metric.ACCURACY, fast_mode: bool = False, verbose: bool = False):
        super().__init__(task, time_budget, metric, fast_mode, verbose)
        self.higher_is_better = self.METRIC_HIGHER_BETTER[metric]
    
    def trainModel(self, X_train: DataFrame, X_test: DataFrame, y_test: DataFrame, y_train: DataFrame, model: BaseEstimator, target_variable: str, params: dict) -> Tuple[BaseEstimator, float, dict]:
        # Train the model
        param_grid = {k: [np.random.choice(v)] for k, v in params.items()}
        for k in param_grid.keys():
            if isinstance(param_grid[k][0], np.float64):
                param_grid[k] = float(param_grid[k][0])
            else:
                param_grid[k] = int(param_grid[k][0])
        model.set_params(**param_grid)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        metric_func = self.METRIC_FUNCTIONS[self.metric]
        score = metric_func(y_test, y_pred)

        return model, score, param_grid

    def fit(self, df: DataFrame, target_variable: str):
        # Determine models and parameters based on task
        if self.task == Task.CLASSIFICATION:
            models = {
                'Random Forest': RandomForestClassifier(random_state=42),
                'XGBoost': xgb.XGBClassifier(),
            }
            # Add Logistic Regression if binary
            if len(df[target_variable].unique()) == 2:
                models['Logistic Regression'] = LogisticRegression()
            model_params = {
                'Random Forest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'XGBoost': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, 20],
                    'learning_rate': np.arange(0.001, 0.01, 0.001)
                },
                'Logistic Regression': {
                    'C': np.arange(0.1, 10, 0.1)
                }
            }
        elif self.task == Task.REGRESSION:
            models = {
                'Random Forest': RandomForestRegressor(random_state=42),
                'XGBoost': xgb.XGBRegressor(),
                'Ridge': Ridge(),
                'Lasso': Lasso(),
                'Linear Regression': LinearRegression(),
            }
            model_params = {
                'Random Forest': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'XGBoost': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, 20],
                    'learning_rate': np.arange(0.001, 0.01, 0.001)
                },
                'Ridge': {
                    'alpha': np.arange(0.1, 10, 0.1)
                },
                'Lasso': {
                    'alpha': np.arange(0.1, 10, 0.1)
                },
                'Linear Regression': {}
            }

        # Prepare data
        X = df.drop(target_variable, axis=1)
        y = df[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        best_model = None
        best_score = -np.inf if self.higher_is_better else np.inf
        best_params = None

        start_time = time.time()
        while time.time() - start_time < self.time_budget:
            for model_name, model in models.items():
                print(f"Training {model_name}")
                # Train the model
                model, score, params = self.trainModel(
                    X_train, X_test, y_test, y_train, model, target_variable, model_params[model_name]
                )
                # Update the best model
                if (self.higher_is_better and score > best_score) or (not self.higher_is_better and score < best_score):
                    best_model = model
                    best_score = score
                    best_params = params
                if self.verbose:
                    print(f"Model: {model_name}, Score: {score:.4f}, Params: {params}")
            print(f"Time Elapsed: {time.time() - start_time:.2f}s")
            print(f"Best Model: {best_model.__class__.__name__} with {self.metric.name} Score: {best_score:.4f} and Parameters: {best_params}")

        self.optimal_model = best_model
        self.metric_value = best_score
        print(f"Optimization Complete. Best Model: {best_model.__class__.__name__} with {self.metric.name} Score: {best_score:.4f}")