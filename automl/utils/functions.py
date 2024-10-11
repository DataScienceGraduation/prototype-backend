import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer, IterativeImputer

# Preprocess function
def preprocess(df: DataFrame) -> DataFrame:    

    # Step 1: Remove duplicates
    try:
        df = df.drop_duplicates()
    except Exception as e:
        print(f"Error in removing duplicates: {e}")
    
    # Step 2: Identify constant columns
    try:
        constant_columns = [col for col in df.columns if df[col].nunique() == 1]
        df.drop(columns=constant_columns, inplace=True)
    except Exception as e:
        print(f"Error in identifying constant columns: {e}")

    # Step 3: Drop columns with too many missing values
    try:
        threshold = 0.4
        df.dropna(axis=1, thresh=int((1 - threshold) * len(df)), inplace=True)
    except Exception as e:
        print(f"Error in dropping columns with missing values: {e}")

    # Step 4: Impute missing values for numerical columns
    try:
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col].fillna(df[col].median(), inplace=True)  # Impute with median
    except Exception as e:
        print(f"Error in imputing missing values for numerical columns: {e}")

    # Step 5: Impute missing values for categorical columns
    try:
        for col in df.select_dtypes(include=['object']).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)  # Impute with mode
    except Exception as e:
        print(f"Error in imputing missing values for categorical columns: {e}")

    # Step 6: Handle high cardinality categorical columns
    try:
        high_cardinality_cols = [col for col in df.select_dtypes(include=['object']).columns 
                                 if df[col].nunique() / len(df) >= 0.9]
        
        df.drop(columns=high_cardinality_cols, inplace=True)
    except Exception as e:
        print(f"Error in handling high cardinality categorical columns: {e}")

    # Step 7: Scale numerical features
    try:
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    except Exception as e:
        print(f"Error in scaling numerical features: {e}")

    # Step 8: Remove outliers using Isolation Forest
    try:
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        iso_forest = IsolationForest(contamination=0.05)
        outliers = iso_forest.fit_predict(df[numerical_cols])
        df = df[outliers != -1]
    except Exception as e:
        print(f"Error in removing outliers using Isolation Forest: {e}")

    # Final Step: Remove any remaining null values
    try:
        df.dropna(inplace=True)
    except Exception as e:
        print(f"Error in removing remaining null values: {e}")

    df_cleaned = df
    return df_cleaned

# Feature Engineering function
def featureEngineer(df: DataFrame, target_variable: str) -> DataFrame:
    df_features = df
    return df_features

# Model Selection function
def selectBestModel(df: DataFrame, target_variable_name: str) -> ClassifierMixin:
    model = RandomForestClassifier()
    return model