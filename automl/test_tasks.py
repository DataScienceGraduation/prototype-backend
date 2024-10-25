# tests/test_tasks.py
import pytest
from automl.models import ModelEntry
from automl.tasks import train_model_task
import pandas as pd
import os

@pytest.mark.django_db
def test_train_model_task(mocker):
    # Mock the file read and data preprocessing functions
    # Ensure both 'X' and 'y' have the same number of samples (4 rows in this case)
    # sample dataframe
    df = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [1, 2, 3, 4],
        'target': [4, 4, 4, 4]
    })
    mocker.patch('pandas.read_csv', return_value=df)
    
    # Mock the pipeline.transform method    
    mock_pipeline = mocker.MagicMock()
    mock_pipeline.transform.return_value = df
    mocker.patch('automl.utils.functions.createPipeline', return_value=mock_pipeline)


    # Create a mock for XGBClassifier
    mock_xgb_classifier = mocker.patch('xgboost.XGBClassifier', autospec=True)

    mocker.patch('automl.utils.functions.selectBestModel', return_value=(mock_xgb_classifier, 0.95))
    
    # Create a test ModelEntry object
    entry = ModelEntry.objects.create(
        name="Test Model",
        description="A test model",
        task="classification",
        target_variable="target",
        list_of_features=['feature1', 'target'],
        status="Data Loaded",
        model_name="",
        evaluation_metric="accuracy",
        evaluation_metric_value=0.0
    )
    
    # Run the Celery task
    train_model_task(entry.id)
    
    # Verify that the ModelEntry object has been updated
    entry.refresh_from_db()
    assert entry.status == 'Done'
    assert entry.evaluation_metric_value == 0.95
    assert entry.model_name == 'XGBClassifier'
    
    # Ensure the model file has been saved
    assert os.path.exists(f'models/{entry.id}.pkl')
    
    # Clean up
    os.remove(f'models/{entry.id}.pkl')
