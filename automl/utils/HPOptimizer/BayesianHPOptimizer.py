from .BaseHPOptimizer import BaseHPOptimizer
from .AcquistionFunctions import ExpectedImprovement
from .SurrogateModels import GaussianProcessSurrogate
import numpy as np


class BayesianHPOptimizer(BaseHPOptimizer):
    def __init__(self, task, time_budget, metric, fast_mode, verbose, surrogate_model = GaussianProcessSurrogate, acquisition_function = ExpectedImprovement):
        super().__init__(task, time_budget, metric, fast_mode, verbose)
        self.surrogate_model = surrogate_model
        self.acquisition_function = acquisition_function
        self.state_space = {
            'model': ['RandomForest', 'XGBoost', 'LogisticRegression'],
            'learning_rate': np.arange(0.01, 0.1, 0.01),
            'n_estimators': np.arange(100, 300, 100),
            'max_depth': np.arange(5, 20, 5),
            'min_samples_split': np.arange(2, 10, 2),
            'C': np.arange(0.1, 10, 0.1),
            'gamma': np.arange(0.1, 10, 0.1)
        }

    def _generate_candidate_points(self, X: np.ndarray, n_points: int = 30):
        """
        Generate candidate points within the bounds of the input data space.
        """
        bounds = [(X[:, i].min(), X[:, i].max()) for i in range(X.shape[1])]
        candidate_points = np.random.uniform(low=bounds[0][0], high=bounds[0][1], size=(n_points, X.shape[1]))
        return candidate_points
    
    def _objective_function(self, params: np.ndarray):
        """
        Objective function to be optimized by the Bayesian optimization process.
        """
        try:
            model = self.trainModel(params)
            score = self.evaluateModel(model)
            return score
        except Exception as e:
            print(f"Error: {e}")
        return
        

    def fit(self, df, target_variable, fast_mode=False):
        X = df.drop(target_variable, axis=1)
        y = df[target_variable]

        initial_points = 5 if fast_mode else 20
        self.surrogate_model = self.surrogate_model()

        X_sample = X.sample(n=initial_points)
        y_sample = y[X_sample.index]

        self.surrogate_model.fit(X_sample, y_sample)

        n_iter = 5 if fast_mode else 20

        for i in range(n_iter):
            candidate_points = self._generate_candidate_points(X_sample)
            
            acquisition_values = self.acquisition_function.evaluate(candidate_points, self.surrogate_model)
            next_point = candidate_points[np.argmax(acquisition_values)]

            next_value = self._objective_function(next_point)

            X_sample = np.vstack([X_sample, next_point])
            y_sample = np.append(y_sample, y.loc[next_point])

            self.surrogate_model.fit(X_sample, y_sample)

            if next_value > self.metric_value:
                self.metric_value = next_value
                self.optimal_hyperparameters = next_point
                self.optimal_model = self.surrogate_model

