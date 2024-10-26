from sklearn.base import BaseEstimator, TransformerMixin

class DropSingleValueColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Identify columns with a single unique value
        self.columns_to_drop_ = [col for col in X.columns if X[col].nunique() == 1]
        return self
    
    def transform(self, X):
        # Drop the identified columns
        return X.drop(columns=self.columns_to_drop_, errors='ignore')