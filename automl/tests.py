from django.test import TestCase
from utils.functions import preprocess
import pandas as pd
# Create your tests here.

@pytest.fixture
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