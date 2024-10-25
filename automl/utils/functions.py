import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pandas import DataFrame
from typing import Tuple
import time

class RemoveDuplicates(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop_duplicates()

class DropHighMissing(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.4):
        self.threshold = threshold
    
    def fit(self, X, y=None):
        self.columns_to_drop_ = X.columns[X.isnull().mean() > self.threshold].tolist()
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_, errors='ignore')

class DropSingleValueColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Identify columns with a single unique value
        self.columns_to_drop_ = [col for col in X.columns if X[col].nunique() == 1]
        return self
    
    def transform(self, X):
        # Drop the identified columns
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
        X_transformed = X.copy()
        for col in self.scalers.keys():
            X_transformed[col] = self.scalers[col].transform(X_transformed[col].values.reshape(-1, 1))
        return X_transformed
    
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

def createPipeline(df: DataFrame, target_variable: str) -> Pipeline:
    print("Creating the pipeline")
    pipeline = Pipeline([
        ('preprocess', Preprocess(target_variable=target_variable)),
        ('feature_engineer', FeatureEngineer(target_variable=target_variable)),
    ])

    # Fit the pipeline on the dataset
    pipeline.fit(df, df[target_variable])

    # Apply the transformations
    transformed_df = pipeline.transform(df)

    return pipeline

def trainModel(df: DataFrame, model: ClassifierMixin, target_variable: str, params: dict) -> Tuple[ClassifierMixin, float]:
    X = df.drop(target_variable, axis=1)
    y = df[target_variable]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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


def selectBestModel(df: DataFrame, target_variable: str, fast_mode: bool = False) -> Tuple[ClassifierMixin, float]:
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
            model, accuracy = trainModel(df, model, target_variable, model_params[model_name])
            # Update the best model
            if accuracy > best_accuracy_score:
                best_model = model
                best_accuracy_score = accuracy
                best_params = model_params[model_name]
        print(f"Time Elapsed: {time.time() - currentTime:.2f}s")
        print(f"Best Model: {best_model.__class__.__name__} with Accuracy Score: {best_accuracy_score:.4f} and Parameters: {best_params}")


    print(f"Best Model: {best_model.__class__.__name__} with Accuracy Score: {best_accuracy_score:.4f} and Parameters: {best_params}")
    return best_model, best_accuracy_score

