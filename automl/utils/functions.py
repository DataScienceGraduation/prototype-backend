import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import joblib
import xgboost as xgb
from pandas import DataFrame
from typing import Tuple

class RemoveDuplicates(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.unique_indices_ = X.drop_duplicates().index
        return self
    
    def transform(self, X):
        return X.loc[self.unique_indices_]

class DropHighMissing(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.4):
        self.threshold = threshold
    
    def fit(self, X, y=None):
        self.columns_to_drop_ = X.columns[X.isnull().mean() > self.threshold].tolist()
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_, errors='ignore')

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

class RemoveHighlyCorrelated(BaseEstimator, TransformerMixin):
    def __init__(self, target_variable=None, correlation_threshold=0.8):
        self.correlation_threshold = correlation_threshold
        self.target_variable = target_variable
    
    def fit(self, X, y=None):
        # Since this transformer needs both X and y, we ensure y is provided
        if y is None:
            y = X[self.target_variable]

        # Combine X and y for correlation calculation
        df = X.copy()
        df[self.target_variable] = y

        # Find numerical columns
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns

        # Correlation matrix
        corr_matrix = df[numerical_columns].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        self.columns_to_drop_ = []

        # Find highly correlated pairs and drop the one with the lower correlation to target
        for col in upper_tri.columns:
            for row in upper_tri.index:
                if upper_tri.loc[row, col] > self.correlation_threshold:
                    corr_with_target_col = abs(df[col].corr(df[self.target_variable]))
                    corr_with_target_row = abs(df[row].corr(df[self.target_variable]))
                    if corr_with_target_col < corr_with_target_row:
                        self.columns_to_drop_.append(col)
                    else:
                        self.columns_to_drop_.append(row)

        # Deduplicate and ensure we don't drop the target variable itself
        self.columns_to_drop_ = list(set(self.columns_to_drop_))
        if self.target_variable in self.columns_to_drop_:
            self.columns_to_drop_.remove(self.target_variable)

        return self
    
    def transform(self, X):
        # Drop the columns identified in fit
        return X.drop(columns=self.columns_to_drop_, errors='ignore')

class RemoveOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0.05):
        self.contamination = contamination
    
    def fit(self, X, y=None):
        numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
        self.iso_forest_ = IsolationForest(contamination=self.contamination, random_state=42)
        self.iso_forest_.fit(X[numerical_cols])
        return self
    
    def transform(self, X):
        numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
        outliers = self.iso_forest_.predict(X[numerical_cols])
        return X[outliers != -1]

class DropNullRows(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.dropna()


class ConvertToDataFrame(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return pd.DataFrame(X, columns=X.columns, index=X.index)


class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, target_variable=None):
        self.target_variable = target_variable
        self.label_encoders = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(include=['object', 'category']).columns:
            if col == self.target_variable:
                continue
            self.label_encoders[col] = LabelEncoder()
            self.label_encoders[col].fit(X[col])
        return self

    def transform(self, X):
        for col in X.select_dtypes(include=['object', 'category']).columns:
            if col == self.target_variable:
                continue
            X[col] = self.label_encoders[col].transform(X[col])
        return X

    def inverse_transform(self, X):
        for col in X.select_dtypes(include=['object', 'category']).columns:
            if col == self.target_variable:
                continue
            X[col] = self.label_encoders[col].inverse_transform(X[col])
        return X

    
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

    def transform(self, X):
        Xt = X.copy()
        for col in X.select_dtypes(include=['int64', 'float64']).columns:
            if col == self.target_variable:
                continue
            X[col] = self.scalers[col].transform(X[col].values.reshape(-1, 1))
        return pd.DataFrame(X, columns=X.columns, index=Xt.index)
    
    def inverse_transform(self, X):
        for col in X.select_dtypes(include=['int64', 'float64']).columns:
            if col == self.target_variable:
                continue
            X[col] = self.scalers[col].inverse_transform(X[col].values.reshape(-1, 1))
        return X


class Preprocess(BaseEstimator, TransformerMixin):
    def __init__(self, target_variable=None):
        self.target_variable = target_variable

        self.sl = Pipeline([
            ('remove_duplicates', RemoveDuplicates()),
            ('drop_high_missing', DropHighMissing()),
            ('drop_high_cardinality', DropHighCardinality()),
            ('remove_outliers', RemoveOutliers()),
            ('drop_null_rows', DropNullRows())
        ])

    def fit(self, X, y=None):
        self.sl.fit(X)
        return self

    def transform(self, X):
        return self.sl.transform(X)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, target_variable=None):
        self.target_variable = target_variable

        self.sl = Pipeline([
            ('remove_highly_correlated', RemoveHighlyCorrelated(target_variable=self.target_variable)),
            ('standard_scaler', CustomStandardScaler(target_variable=self.target_variable)),
        ])

    def fit(self, X, y=None):
        self.sl.fit(X)
        return self
    
    def transform(self, X):
        transformed_array = self.sl.transform(X)

        return pd.DataFrame(transformed_array, columns=X.columns, index=X.index)


def createPipeline(df: DataFrame, target_variable: str) -> Pipeline:
    sl = Pipeline([
        ('feature_engineer', FeatureEngineer(target_variable))
    ])


    sl.fit(df)

    df = sl.transform(df)

    print(df.head(5))

    return sl

def selectBestModel(df: DataFrame, target_variable: str, fast_mode: bool = False) -> Tuple[ClassifierMixin, float]:
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(),
        'SVM (Linear)': svm.SVC(kernel='linear', probability=True),
        'SVM (RBF)': svm.SVC(kernel='rbf', probability=True)
    }

    # Add Logistic Regression if binary classification
    if len(df[target_variable].unique()) == 2:
        models['Logistic Regression'] = LogisticRegression()

    # Parameter grids for GridSearchCV and RandomizedSearchCV
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200] if fast_mode else [50, 100, 200, 500],
            'max_depth': [None, 10, 20, 30]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'SVM (Linear)': {
            'C': [0.1, 1, 10]
        },
        'SVM (RBF)': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto']
        },
        'Logistic Regression': {
            'C': [0.1, 1, 10],
            'max_iter': [100, 500, 1000]
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

    # Timeout for fast mode (10 seconds)
    timeout = 10 if fast_mode else None

    # Train and evaluate each model with hyperparameter optimization
    for model_name, model in models.items():
        print(f"Running {model_name} model...")
        
        if fast_mode:
            # RandomizedSearchCV with cv=2 for fast mode
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grids[model_name],
                n_iter=5,
                cv=2,
                random_state=42,
                n_jobs=-1
            )
        else:
            # GridSearchCV with cv=5 for full mode
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[model_name],
                cv=5,
                n_jobs=-1
            )

        try:
            search.fit(X_train, y_train)
        except TimeoutError:
            print(f"{model_name} exceeded the time limit of {timeout} seconds in fast mode.")
            continue

        # Best estimator
        best_estimator = search.best_estimator_

        # Predict on the test set
        y_pred = best_estimator.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print(f"{model_name} Accuracy: {accuracy:.4f} (Best Params: {search.best_params_})")

        # Update best model based on accuracy score
        if accuracy > best_accuracy_score:
            best_accuracy_score = accuracy
            best_model = best_estimator

    print(f"Best Model: {best_model.__class__.__name__} with Accuracy Score: {best_accuracy_score:.4f}")
    return best_model, best_accuracy_score
