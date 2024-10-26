from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, target_variable=None):
        self.target_variable = target_variable
        self.scalers = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(include=['int64', 'float64']).columns:
            if col == self.target_variable:
                continue
            self.scalers[col] = StandardScaler()
            self.scalers[col].fit(X[col].values.reshape(-1, 1))
        return self