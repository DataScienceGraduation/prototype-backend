import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class RemoveHighlyCorrelated(BaseEstimator, TransformerMixin):
    def __init__(self, target_variable=None, correlation_threshold=0.8):
        self.correlation_threshold = correlation_threshold
        self.target_variable = target_variable
    
    def fit(self, X, y=None):
        # Combine X and y for correlation calculation
        df = X.copy()

        # Find numerical columns
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

        # Correlation matrix
        corr_matrix = df[numerical_columns].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        self.columns_to_drop_ = []

        # Find highly correlated pairs and drop one of them
        for col in upper_tri.columns:
            for row in upper_tri.index:
                if upper_tri.loc[row, col] > self.correlation_threshold:
                    # Drop the column with lower correlation to the target variable
                    if abs(df[col].corr(df[self.target_variable])) < abs(df[row].corr(df[self.target_variable])):
                        self.columns_to_drop_.append(col)
                    else:
                        self.columns_to_drop_.append(row)

        # Deduplicate and ensure we don't drop the target variable itself
        self.columns_to_drop_ = list(set(self.columns_to_drop_))
        if self.target_variable in self.columns_to_drop_:
            self.columns_to_drop_.remove(self.target_variable)

        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_, errors='ignore')