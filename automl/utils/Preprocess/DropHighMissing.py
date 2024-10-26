from sklearn.base import BaseEstimator, TransformerMixin

class DropHighMissing(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.4):
        self.threshold = threshold
    
    def fit(self, X, y=None):
        self.columns_to_drop_ = X.columns[X.isnull().mean() > self.threshold].tolist()
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_, errors='ignore')