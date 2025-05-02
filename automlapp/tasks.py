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
        
        # Handle clustering differently since it doesn't need a target variable
        if entry.task == 'Clustering':
            pl = createPipeline(df, None)
            df = pl.transform(df)
            X = df  # For clustering, use all features
        else:
            pl = createPipeline(df, entry.target_variable)
            df = pl.transform(df)
            X = df.drop(entry.target_variable, axis=1)
            y = df[entry.target_variable]
            
        entry.status = 'Model Selection and Training'
        entry.save()
        hpo = RandomSearchOptimizer(task=Task.parse(entry.task), time_budget=300)
        
        if entry.task == 'Clustering':
            hpo.fit(X, None)  # For clustering, no target variable needed
        else:
            hpo.fit(X, y)
            
        accuracy = hpo.get_metric_value()
        model = hpo.get_optimal_model()
        entry.status = 'Saving Model'
        entry.model_name = model.__class__.__name__
        if entry.task == 'Classification':
            entry.evaluation_metric = Metric.ACCURACY
            entry.evaluation_metric_value = accuracy
        elif entry.task == 'Regression':
            entry.evaluation_metric = Metric.RMSE
            entry.evaluation_metric_value = accuracy
        elif entry.task == 'Clustering':
            entry.evaluation_metric = Metric.SILHOUETTE
            entry.evaluation_metric_value = accuracy
        elif entry.task == 'TimeSeries':
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