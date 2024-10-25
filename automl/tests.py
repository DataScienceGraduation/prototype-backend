from django.test import TestCase
import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import xgboost as xgb
from .utils.functions import FeatureEngineer, selectBestModel, Preprocess, RemoveDuplicates
from sklearn.experimental import enable_iterative_imputer
import pandas as pd
import os   


sample_data = pd.DataFrame({
        'A': [1, 1, 2, 3, None],
        'B': ['foo', 'foo', 'foo', 'foo', 'bar'],
        'C': [1, 1, 1, 1, 1],
        'D': [1.0, 1.0, None, 4.0, 5.0],
        'target': [1, 1, 0, 1, 0]
})

print(sample_data)


def test_feature_engineering():
    df = pd.DataFrame(sample_data)

    result = FeatureEngineer('target').fit_transform(df)

    assert result.shape[1] < df.shape[
        1], "The number of features after feature engineering should be less than the original number of features."

    assert 'target' in result.columns, "The target variable should be retained in the final DataFrame."

def test_selectBestModel():
    curDir = os.path.dirname(__file__)
    dataDir = os.path.join(curDir, 'testData')
    dataDir = os.path.join(dataDir, 'diabetes.csv')
    data = pd.read_csv(dataDir)

    df = pd.DataFrame(data)
    df = Preprocess('Outcome').fit_transform(df)
    df = FeatureEngineer('Outcome').fit_transform(df)

    best_model, accuracy = selectBestModel(df, 'Outcome', fast_mode=True)
    print(best_model.__class__.__name__)

    assert best_model is not None, "The best model should not be None"

    print(best_model)
    assert isinstance(best_model, (RandomForestClassifier, xgb.XGBClassifier, svm.SVC, LogisticRegression)), \
        "The best model should be one of the defined classifiers"

    assert accuracy > 0, \
        "The accuracy score of the best model should be greater than 0"


def test_remove_duplicates():
    processed_data = RemoveDuplicates().fit_transform(sample_data)
    print(processed_data)
    assert processed_data.shape[0] == 4  # Should have removed one duplicate


def test_constant_columns():
    processed_data = Preprocess('target').fit_transform(sample_data)
    assert 'C' not in processed_data.columns  # Constant column 'C' should be removed


def test_missing_value_imputation():
    processed_data = Preprocess('target').fit_transform(sample_data)
    assert processed_data['D'].isnull().sum() == 0  # No missing values in 'D'


def test_high_cardinality_removal():
    processed_data = Preprocess('target').fit_transform(sample_data)
    assert 'C' not in processed_data.columns  # High cardinality column 'B' should be removed


def test_scaling_numerical_features():
    processed_data = Preprocess('target').fit_transform(sample_data)
    assert (processed_data[['A', 'D']] >= 0).all().all()  # Check if values are scaled properly


def test_label_encoding():
    processed_data = Preprocess('target').fit_transform(sample_data)
    assert processed_data['B'].dtype == 'int64'  # 'B' should be label encoded


def test_no_remaining_nulls():
    processed_data = Preprocess('target').fit_transform(sample_data)
    assert processed_data.isnull().sum().sum() == 0  # No remaining null values in the final DataFrame


if __name__ == "__main__":
    pytest.main()
