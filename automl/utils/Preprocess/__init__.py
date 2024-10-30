from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from .RemoveDuplicates import RemoveDuplicates
from .DropHighMissing import DropHighMissing
from .DropSingleValueColumns import DropSingleValueColumns
from .DropHighCardinality import DropHighCardinality
from .DropNullRows import DropNullRows
from .CustomLabelEncoder import CustomLabelEncoder
from .RemoveOutliers import RemoveOutliers


class Preprocess(BaseEstimator, TransformerMixin):
    def __init__(self, target_variable=None):
        self.target_variable = target_variable
        self.pipeline = Pipeline([
            ('remove_duplicates', RemoveDuplicates()),
            ('drop_high_missing', DropHighMissing()),
            ('drop_single_value_columns', DropSingleValueColumns()),
            ('drop_high_cardinality', DropHighCardinality()),
            ('drop_null_rows', DropNullRows()),
            ('label_encoding', CustomLabelEncoder(target_variable=self.target_variable)),
            ('remove_outliers', RemoveOutliers())
        ])  

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)