import BaseHPOptimizer as BHPO
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import time


class RandomSearchHPOptimizer(BHPO):
    """
    Random search hyperparameter optimizer

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
    def __init__(self, task: BHPO.Task, time_budget: int, metric: BHPO.Metric = BHPO.Metric.ACCURACY, fast_mode: bool = False, verbose: bool = False):
        super().__init__(task, time_budget, metric, fast_mode, verbose)
    
    def trainModel(X_train: BHPO.DataFrame, X_test: BHPO.DataFrame, y_test: BHPO.DataFrame, y_train: BHPO.DataFrame, model: BHPO.ClassifierMixin, target_variable: str, params: dict) -> BHPO.Tuple[BHPO.ClassifierMixin, float]:
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

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        return model, accuracy


    def fit(self, df: BHPO.DataFrame, target_variable: str, fast_mode: bool = False):
        timeout = 10 if fast_mode else 300  # Timeout in seconds
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(),
            #'SVM (Linear)': svm.SVC(kernel='linear', probability=True),
            #'SVM (RBF)': svm.SVC(kernel='rbf', probability=True)
        }

        # Add Logistic Regression if binary classification
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
            'SVM (Linear)': {
                'C': np.arange(0.1, 10, 0.1)
            },
            'SVM (RBF)': {
                'C': np.arange(0.1, 10, 0.1),
                'gamma': np.arange(0.1, 10, 0.1)
            },
            'Logistic Regression': {
                'C': np.arange(0.1, 10, 0.1)
            }
        }
    
        # Prepare the data
        X = df.drop(target_variable, axis=1)
        y = df[target_variable]

        # Fast mode uses 50% of the data for speed
        test_size = 0.5 if fast_mode else 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        best_model = None
        best_accuracy_score = 0
        best_params = None

        currentTime = time.time()
        while time.time() - currentTime < timeout:
            for model_name, model in models.items():
                print(f"Training {model_name}")
                if model_name not in model_params:
                    raise ValueError(f"Parameters for {model_name} not found")
                
                # Train the model
                model, accuracy = self.trainModel(X_train, y_train, X_test, y_test, model, target_variable, model_params[model_name])
                # Update the best model
                if accuracy > best_accuracy_score:
                    best_model = model
                    best_accuracy_score = accuracy
                    best_params = model_params[model_name]
            print(f"Time Elapsed: {time.time() - currentTime:.2f}s")
            print(f"Best Model: {best_model.__class__.__name__} with Accuracy Score: {best_accuracy_score:.4f} and Parameters: {best_params}")


        print(f"Best Model: {best_model.__class__.__name__} with Accuracy Score: {best_accuracy_score:.4f} and Parameters: {best_params}")
        self.optimal_model = best_model
        self.metric_value = best_accuracy_score
        

    def getOptimalModel(self) -> BHPO.ClassifierMixin:
        return self.optimal_model
    
    def getMetric(self) -> BHPO.Metric:
        return self.metric
    
    def getMetricValue(self) -> float:
        return self.metric_value