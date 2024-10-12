import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import LabelEncoder

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

    # Step 9: Label encode categorical columns
    try:
        categorical_cols = df.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col])
    except Exception as e:
        print(f"Error in label encoding categorical columns: {e}")

    # Final Step: Remove any remaining null values
    try:
        df.dropna(inplace=True)
    except Exception as e:
        print(f"Error in removing remaining null values: {e}")


    df_cleaned = df
    return df_cleaned


# Feature Engineering function
def featureEngineer(df: DataFrame, target_variable: str) -> DataFrame:
    # 1. Removing highly correlated features
    try:
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        corr_matrix = df[numerical_columns].corr().abs()

        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        high_corr_pairs = [(col1, col2) for col1 in upper_tri.columns for col2 in upper_tri.index if
                           upper_tri.loc[col2, col1] > 0.80]
        columns_to_drop = []

        for col1, col2 in high_corr_pairs:
            # Get correlation with the target variable
            corr_with_target_col1 = df[col1].corr(df[target_variable])
            corr_with_target_col2 = df[col2].corr(df[target_variable])

            # Drop the column with lower correlation to the target variable
            if abs(corr_with_target_col1) < abs(corr_with_target_col2):
                columns_to_drop.append(col1)
            else:
                columns_to_drop.append(col2)
        df.drop(columns_to_drop, axis=1, inplace=True)
    except Exception as e:
        print(f"Error in creating the correlation matrix: {e}")

    # 2. Applying PCA
    try:
        pca = PCA()
        pca.fit(df.drop(target_variable, axis=1))

        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        thresholds = [0.90, 0.95, 0.99]
        best_components = []

        for threshold in thresholds:
            n_components = np.argmax(cumulative_variance >= threshold) + 1
            best_components.append(n_components)

        pca = PCA(n_components=best_components[0])
        df_transformed = pca.fit_transform(df.drop(target_variable, axis=1))

        # Convert the transformed array back to a DataFrame if needed
        df_pca = pd.DataFrame(df_transformed, columns=[f'PC{i + 1}' for i in range(best_components[0])])

        # Optionally concatenate the target variable back to the PCA DataFrame if needed
        df_final = pd.concat([df_pca, df[target_variable].reset_index(drop=True)], axis=1)
        return df_final

    except Exception as e:
       print(f"Error Applying PCA: {e}")

    return df


# Model Selection function
def selectBestModel(df: DataFrame, target_variable_name: str) -> ClassifierMixin:
    model = RandomForestClassifier()
    return model