from django.test import TestCase
import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import xgboost as xgb
from .utils.functions import featureEngineer, selectBestModel, preprocess
from sklearn.experimental import enable_iterative_imputer
import pandas as pd


# Test for Feature Engineering
@pytest.fixture
def test_feature_engineering():
    # Create the DataFrame using your provided data
    data = {
        'feature_1': [-2.122893, -0.233728, 1.966536, 0.466917, -0.613779],
        'feature_2': [-1.692462, -0.384591, 1.815277, 0.426717, -0.577104],
        'feature_3': [-0.494863, 0.337907, 0.494017, 0.831855, 0.916246],
        'feature_4': [0.010582, 4.643309, 1.148911, 1.031489, 0.015330],
        'target': [0, 1, 1, 1, 0]
    }

    df = pd.DataFrame(data)

    # Apply feature engineering to the DataFrame
    result = featureEngineer(df, 'target')

    # Check that the number of features after removing highly correlated features and applying PCA is less
    assert result.shape[1] < df.shape[
        1], "The number of features after feature engineering should be less than the original number of features."

    # Check if the target variable is still present
    assert 'target' in result.columns, "The target variable should be retained in the final DataFrame."

    # Validate that PCA reduced the dimensionality as expected
    assert len(
        result.columns) - 1 <= 4, "The number of features after PCA should not exceed the number of original features minus the target variable."


# Test for Model Selection
def test_selectBestModel():
    # Create sample binary classification data
    data = {
        'feature_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature_2': [5, 4, 3, 2, 1, 0, 1, 2, 3, 4],
        'feature_3': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'feature_4': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        'target': [0, 1, 1, 0, 1, 1, 0, 1, 0, 1]
    }

    df = pd.DataFrame(data)

    # Run the model selection function
    best_model = selectBestModel(df, 'target')

    # Check that the best model is not None
    assert best_model is not None, "The best model should not be None"

    # Check that the best model is one of the expected types
    assert isinstance(best_model, (RandomForestClassifier, xgb.XGBClassifier, svm.SVC, LogisticRegression)), \
        "The best model should be one of the defined classifiers"

    # Optionally, you can check if the accuracy score is reasonable (greater than 0)
    assert best_model.score(df.drop('target', axis=1), df['target']) > 0, \
        "The accuracy score of the best model should be greater than 0"


def sample_data():
    return pd.DataFrame({
        'A': [1, 2, 2, 3, None],
        'B': [None, None, 'foo', 'foo', 'bar'],
        'C': [1, 1, 1, 1, 1],
        'D': [1.0, 2.0, None, 4.0, 5.0]
    })


def test_remove_duplicates(sample_data):
    processed_data = preprocess(sample_data)
    assert processed_data.shape[0] == 4  # Should have removed one duplicate


def test_constant_columns(sample_data):
    processed_data = preprocess(sample_data)
    assert 'C' not in processed_data.columns  # Constant column 'C' should be removed


def test_missing_value_imputation(sample_data):
    processed_data = preprocess(sample_data)
    assert processed_data['D'].isnull().sum() == 0  # No missing values in 'D'


def test_high_cardinality_removal(sample_data):
    processed_data = preprocess(sample_data)
    assert 'B' not in processed_data.columns  # High cardinality column 'B' should be removed


def test_scaling_numerical_features(sample_data):
    processed_data = preprocess(sample_data)
    assert (processed_data[['A', 'D']] >= 0).all().all()  # Check if values are scaled properly


def test_label_encoding(sample_data):
    processed_data = preprocess(sample_data)
    assert processed_data['B'].dtype == 'int64'  # 'B' should be label encoded


def test_no_remaining_nulls(sample_data):
    processed_data = preprocess(sample_data)
    assert processed_data.isnull().sum().sum() == 0  # No remaining null values in the final DataFrame


if __name__ == "__main__":
    pytest.main()
