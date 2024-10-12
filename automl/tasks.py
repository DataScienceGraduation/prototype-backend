from celery import shared_task
from .models import ModelEntry
from .utils.functions import preprocess, featureEngineer, selectBestModel
import joblib
import os

import pandas as pd

@shared_task
def train_model_task(entry_id):
    try:
        entry = ModelEntry.objects.get(id=entry_id)
        # Simulate long model training process
        entry.status = 'Preprocessing Data'
        entry.save()
        df = pd.read_csv(f'data/{entry.id}.csv')
        df = preprocess(df, entry.target_variable)
        entry.status = 'Feature Engineering'
        entry.save()
        df = featureEngineer(df, entry.target_variable)
        entry.status = 'Model Selection and Training'
        entry.save()
        model, accuracy = selectBestModel(df, entry.target_variable)
        entry.model_name = model.__class__.__name__
        entry.evaluation_metric = 'Accuracy'
        entry.evaluation_metric_value = accuracy
        entry.status = 'Done'
        entry.save()
        os.remove(f'data/{entry.id}.csv')
        joblib.dump(model, f'models/{entry.id}.pkl')
    except Exception as e:
        print(f"Error during model training: {e}")
        entry.status = 'Training Failed'
        entry.save()