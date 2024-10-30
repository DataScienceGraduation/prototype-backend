from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from .RemoveHighlyCorrelated import RemoveHighlyCorrelated
from .CustomStandardScaler import CustomStandardScaler


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, target_variable=None):
        self.target_variable = target_variable
        self.pipeline = Pipeline([
            ('remove_highly_correlated', RemoveHighlyCorrelated(target_variable=self.target_variable)),
            ('standard_scaler', CustomStandardScaler(target_variable=self.target_variable)),
        ])

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self
    
    def transform(self, X):
        return self.pipeline.transform(X)