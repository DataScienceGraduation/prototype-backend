# tests/test_tasks.py
import pytest
from automlapp.models import ModelEntry
from automlapp.tasks import train_model_task
from unittest import mock
import pandas as pd
import os
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import io

@pytest.mark.django_db
def test_train_model_task(mocker):
    # Mock the file read and data preprocessing functions
    # Ensure both 'X' and 'y' have the same number of samples (4 rows in this case)
    # sample dataframe
    curDir = os.path.dirname(__file__)
    curDir = os.path.join(curDir, 'diabetes.csv')
    df = pd.read_csv(curDir)
    
    # Create a test ModelEntry object
    entry = ModelEntry.objects.create(
        name="Test Model",
        description="A test model",
        task="classification",
        target_variable="Outcome",
        list_of_features=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        status="Data Loaded",
        model_name="",
        evaluation_metric="accuracy",
        evaluation_metric_value=0.0
    )
    
    # Save the dataframe to the storage
    file_path = f'data/{entry.id}.csv'
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    default_storage.save(file_path, ContentFile(csv_buffer.getvalue().encode('utf-8')))
    
    mocker.patch('pandas.read_csv', return_value=df)
    
    # Mock the pipeline.transform method    
    mock_pipeline = mocker.MagicMock()
    mock_pipeline.transform.return_value = df
    mocker.patch('automl.functions.createPipeline', return_value=mock_pipeline)


    # Create a mock for XGBClassifier, RandomForestClassifier, etc.
    mock_xgb = mocker.MagicMock()
    mock_xgb.__class__.__name__ = 'XGBClassifier'
    mocker.patch('xgboost.XGBClassifier', return_value=mock_xgb)

    # Mock RandomSearchOptimizer
    mock_hpo = mocker.MagicMock()
    mock_hpo.metric_value = 0.95  # Simulate accuracy metric value
    mock_model = mocker.MagicMock()  # Mock the model object
    
    # Mock methods in RandomSearchOptimizer
    # Mock RandomSearchHPOptimizer and its methods accurately
    mock_hpo = mocker.MagicMock()
    mock_hpo.metric_value = 0.95  # Simulated accuracy metric value
    mock_model = mocker.MagicMock()
    mock_model.__class__.__name__ = "XGBClassifier"

    # Mock methods and return values
    mocker.patch('automl.optimizers.RandomSearchOptimizer', return_value=mock_hpo)
    mock_hpo.get_optimal_model.return_value = mock_model
    mock_hpo.fit.return_value = None
    mock_hpo.get_metric_value.return_value = 0.95

    mocker.patch('joblib.dump', return_value=None)
        
    
    # Run the Celery task
    train_model_task(entry.id)
    
    # Verify that the ModelEntry object has been updated
    entry.refresh_from_db()
    assert entry.status == 'Done'
    assert entry.evaluation_metric_value == 0.95
    assert entry.model_name == 'XGBClassifier'
    
    # Ensure the model file has been saved
    model_path = f'models/{entry.id}.pkl'
    assert default_storage.exists(model_path)
    
    # Clean up
    default_storage.delete(model_path)
    default_storage.delete(file_path)