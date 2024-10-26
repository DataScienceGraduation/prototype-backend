from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, target_variable=None):
        self.target_variable = target_variable
        self.label_encoders = {}
        self.target_encoder = None

    def fit(self, X, y=None):
        for col in X.select_dtypes(include=['object', 'category', 'string']).columns:
            self.label_encoders[col] = LabelEncoder()
            self.label_encoders[col].fit(X[col])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.label_encoders.keys():
            X_transformed[col] = self.label_encoders[col].transform(X_transformed[col])
        return X_transformed