from .BaseHPOptimizer import BaseHPOptimizer
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging
import time

logging.basicConfig(
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s'  
)

class BayesianHPOptimizer(BaseHPOptimizer):
    def __init__(self, task, time_budget, metric, fast_mode, verbose, surrogate_model, acquisition_function):
        super().__init__(task, time_budget, metric, fast_mode, verbose)
        self.surrogate_model = surrogate_model
        self.acquisition_function = acquisition_function
        self.state_space = {
            'model': ['RandomForest', 'XGBoost', 'LogisticRegression'],
            'learning_rate': np.arange(0.001, 0.1, 0.001),
            'n_estimators': np.arange(100, 300, 10),
            'max_depth': np.arange(5, 30, 1),
            'min_samples_split': np.arange(2, 10, 2),
            'C': np.arange(0.1, 10, 0.1),
            'gamma': np.arange(0.1, 10, 0.1)
        }

        self.best_score = 0.0
        
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
    
    def _initialize_model(self, params: dict):
        """
        Initialize a model based on the provided parameters.
        """
        model = None
        if params['model'] == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], min_samples_split=params['min_samples_split'])
        elif params['model'] == 'XGBoost':
            from xgboost import XGBClassifier
            model = XGBClassifier(learning_rate=params['learning_rate'], n_estimators=params['n_estimators'], max_depth=params['max_depth'], gamma=params['gamma'])
        elif params['model'] == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(C=params['C'])
        else:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(C=5)
        return model


    def _objective_function(self, params: dict, X: np.ndarray, y: np.ndarray) -> float:
        """
        Objective function to be optimized by the Bayesian optimization process.
        """
        # Decode model type for initialization
        logging.info(f'params: {params}')

        params['model'] = self.model_encoder.inverse_transform([params['model']])[0]
        model = self._initialize_model(params)
        scores = cross_val_score(model, X, y, cv=5, scoring=self.metric.name.lower())
        score = scores.mean()
        return score

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Perform Bayesian Optimization to fit the best model based on the defined objective.
        """
        history = []
        scores = []

        # Initial random samples
        for _ in range(5):
            params = self._generate_candidate_points(1)[0]
            score = self._objective_function(params, X, y)
            history.append((params, score))
        
        logging.info(f"history: {history}")

        # Main Bayesian Optimization loop
        currentTime = time.time()
        while time.time() - currentTime < self.time_budget:
            X_history = np.array([list(h[0].values()) for h in history])
            y_history = np.array([h[1] for h in history])
            X_history[:, 0] = self.model_encoder.transform(X_history[:, 0])
            X_history = X_history.astype(np.float64)
            self.surrogate_model.fit(X_history, y_history)
            candidates = self._generate_candidate_points(30)
            acquisition_values = self.acquisition_function.evaluate(candidates, self.surrogate_model, self.best_score)
            best_candidate = candidates[np.argmax(acquisition_values)]
            params = dict()
            for key, value in best_candidate.items():
                params[key] = value
            score = self._objective_function(params, X, y)
            scores.append(score)
            self.best_score = max(score, self.best_score)
            params = dict()
            for key, value in best_candidate.items():
                params[key] = value
            params['model'] = self.model_encoder.inverse_transform([params['model']])[0]
            history.append([params, score])
            logging.info(f"Best Score: {self.best_score}")
            logging.info(f"Available Time: {self.time_budget - (time.time() - currentTime)}")

        logging.info(f"history: {history}")
        history.pop()
        best_params = max(history, key=lambda item: item[1])[0]
        logging.info(f"Best Params found {best_params}")

        model = self._initialize_model(best_params)

        model.fit(X, y)

        logging.info(f"Best Score: {self.best_score}")


        self.optimal_model = model

        return model
