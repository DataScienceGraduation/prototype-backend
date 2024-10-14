# tests/test_views.py
import pytest
from django.urls import reverse
from django.test import Client
from django.core.files.uploadedfile import SimpleUploadedFile
from automl.models import ModelEntry
import pandas as pd
import os

@pytest.mark.django_db
def test_load_data_view():
    client = Client()
    
    # Create a sample CSV file for testing
    sample_data = "feature1,feature2,target\n1,2,A\n3,4,B\n5,6,A"
    sample_file = SimpleUploadedFile("test.csv", sample_data.encode('utf-8'), content_type="text/csv")
    
    # Send POST request to loadData endpoint with the sample file
    response = client.post(reverse('loadData'), {'file': sample_file})
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Assert that the response contains the correct keys
    assert data['success'] is True
    assert 'data' in data
    assert 'id' in data
    
    # Verify that a new ModelEntry object has been created
    entry = ModelEntry.objects.get(id=data['id'])
    assert entry is not None
    
    # Clean up the CSV file saved during the test
    os.remove(f'data/{entry.id}.csv')

# tests/test_views.py
@pytest.mark.django_db
def test_train_model_view(mocker):
    client = Client()
    
    # Create a sample ModelEntry object for testing
    entry = ModelEntry.objects.create(
        name="",
        description="",
        task="",
        target_variable="",
        list_of_features=['feature1', 'feature2'],
        status="Data Loaded",
        model_name="",
        evaluation_metric="",
        evaluation_metric_value=0
    )
    
    # Mock the Celery task to avoid actually running it during the test
    mocker.patch('automl.tasks.train_model_task.delay')

    # Send POST request to trainModel endpoint
    response = client.post(reverse('trainModel'), {
        'id': entry.id,
        'name': 'Test Model',
        'description': 'A test model',
        'task': 'classification',
        'target_variable': 'target'
    })
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Assert the response indicates success
    assert data['success'] is True
    
    # Verify that the ModelEntry has been updated
    entry.refresh_from_db()
    assert entry.name == 'Test Model'
    assert entry.description == 'A test model'
    assert entry.status == 'Model Training'
