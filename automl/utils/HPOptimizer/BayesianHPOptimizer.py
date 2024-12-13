from .BaseHPOptimizer import BaseHPOptimizer
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

class BayesianHPOptimizer(BaseHPOptimizer):
    def __init__(self, task, time_budget, metric, fast_mode, verbose, surrogate_model, acquisition_function):
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
        
        # Set up a LabelEncoder for the 'model' parameter
        self.model_encoder = LabelEncoder()
        self.model_encoder.fit(self.state_space['model'])

    def _generate_candidate_points(self, n_points: int = 30) -> list:
        """
        Generate candidate points within the bounds of the input data space, encoding categorical values as needed.
        """
        candidates = []
        for _ in range(n_points):
            candidate = {param: np.random.choice(values) for param, values in self.state_space.items()}
            # Encode the model name numerically
            candidate['model'] = self.model_encoder.transform([candidate['model']])[0]
            candidates.append(candidate)
        return candidates  # Return a list of dictionaries


    def _objective_function(self, params: dict, X: np.ndarray, y: np.ndarray) -> float:
        """
        Objective function to be optimized by the Bayesian optimization process.
        """
        # Decode model type for initialization
        print(params['model'])
        params['model'] = self.model_encoder.inverse_transform(params['model'])[0]
        model = self._initialize_model(params)
        score = cross_val_score(model, X, y, cv=5, scoring=self.metric)
        return score.mean()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Perform Bayesian Optimization to fit the best model based on the defined objective.
        """
        history = []

        # Initial random samples
        for _ in range(5):
            params = self._generate_candidate_points(1)[0]
            print(params)
            score = self._objective_function(params, X, y)
            history.append((params, score))

        # Main Bayesian Optimization loop
        for _ in range(self.time_budget):
            X_history = np.array([list(h[0].values()) for h in history])
            y_history = np.array([h[1] for h in history])
            self.surrogate_model.fit(X_history, y_history)

            candidates = self._generate_candidate_points(30)
            acquisition_values = self.acquisition_function.evaluate(candidates, self.surrogate_model)
            best_candidate = candidates[np.argmax(acquisition_values)]
            score = self._objective_function(dict(zip(self.state_space.keys(), best_candidate)), X, y)
            history.append((dict(zip(self.state_space.keys(), best_candidate)), score))

        # Return the best parameters found
        best_params = max(history, key=lambda item: item[1])[0]
        # Decode the model type for the best model before initializing
        best_params['model'] = self.model_encoder.inverse_transform([best_params['model']])[0]
        return self._initialize_model(best_params)
