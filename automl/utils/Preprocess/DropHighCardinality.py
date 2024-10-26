from sklearn.base import BaseEstimator, TransformerMixin

class DropHighCardinality(BaseEstimator, TransformerMixin):
    def __init__(self, cardinality_threshold=0.9):
        self.cardinality_threshold = cardinality_threshold
    
    def fit(self, X, y=None):
        self.columns_to_drop_ = []
        for col in X.select_dtypes(include=['object', 'category']).columns:
            if X[col].nunique() / X.shape[0] >= self.cardinality_threshold:
                self.columns_to_drop_.append(col)
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_, errors='ignore')