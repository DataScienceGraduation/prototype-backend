from celery import shared_task
from .models import ModelEntry
from .utils.HPOptimizer.RandomSearchHPOptimizer import RandomSearchHPOptimizer
from .utils.HPOptimizer.BaseHPOptimizer import Task, Metric
from .utils.functions import createPipeline
import joblib
import os

import pandas as pd

@shared_task
def train_model_task(entry_id):
    try:
        entry = ModelEntry.objects.get(id=entry_id)
        # Simulate long model training process
        entry.status = 'Loading Data'
        entry.save()
        df = pd.read_csv(f'data/{entry.id}.csv')
        entry.status = 'Preprocessing Data'
        entry.save()
        pl = createPipeline(df, entry.target_variable)
        df = pl.transform(df)
        entry.status = 'Model Selection and Training'
        entry.save()
        entry.status = df
        entry.save()
        hpo = RandomSearchHPOptimizer(task=Task.CLASSIFICATION, time_budget=600, metric=Metric.ACCURACY)
        hpo.fit(df, entry.target_variable)
        accuracy = hpo.metric_value
        model = hpo.getOptimalModel()
        entry.status = 'Saving Model'
        entry.model_name = model.__class__.__name__
        entry.evaluation_metric = 'Accuracy'
        entry.evaluation_metric_value = accuracy
        entry.status = 'Done'
        entry.save()
        joblib.dump(model, f'models/{entry.id}.pkl')
        joblib.dump(pl, f'pipelines/{entry.id}.pkl')
    except Exception as e:
        print(f"Error during model training: {df}, {e}, {entry.status}")
        entry.status = 'failed'
        entry.save()