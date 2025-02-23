from celery import shared_task
from .models import ModelEntry
from automl.optimizers import RandomSearchOptimizer
from automl.enums import Task, Metric
from automl.functions import createPipeline
import joblib
import os
import pandas as pd

@shared_task
def train_model_task(entry_id):
    try:
        entry = ModelEntry.objects.get(id=entry_id)
        entry.status = 'Loading Data'
        entry.save()
        df = pd.read_csv(f'data/{entry.id}.csv')
        entry.status = 'Preprocessing Data'
        entry.save()
        pl = createPipeline(df, entry.target_variable)
        df = pl.transform(df)
        entry.status = 'Model Selection and Training'
        entry.save()
        hpo = RandomSearchOptimizer(task=Task.parse(entry.task), time_budget=100)
        hpo.fit(df.drop(entry.target_variable, axis=1), df[entry.target_variable])
        accuracy = hpo.get_metric_value()
        model = hpo.get_optimal_model()
        entry.status = 'Saving Model'
        entry.model_name = model.__class__.__name__
        if entry.task == 'Classification':
            entry.evaluation_metric = Metric.ACCURACY
            entry.evaluation_metric_value = accuracy
        else:
            entry.evaluation_metric = Metric.RMSE
            entry.evaluation_metric_value = accuracy
        entry.status = 'Done'
        entry.save()
        joblib.dump(model, f'models/{entry.id}.pkl')
        joblib.dump(pl, f'pipelines/{entry.id}.pkl')
    except Exception as e:
        print(f"Error during model training: {df}, {e}, {entry.status}")
        entry.status = 'failed'
        entry.save()