from celery import shared_task
from .models import ModelEntry
from automl.optimizers import RandomSearchOptimizer
from automl.enums import Task, Metric
from automl.functions import createPipeline
import joblib
import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@shared_task
def train_model_task(entry_id):
    """
    Celery task to train a machine learning model using RandomSearchOptimizer.

    Args:
        entry_id: ID of the ModelEntry record to process.

    Raises:
        Exception: Propagates any errors to Celery for handling.
    """
    try:
        logger.info(f"Starting training for ModelEntry ID: {entry_id}")
        entry = ModelEntry.objects.get(id=entry_id)
        
        entry.status = 'Loading Data'
        entry.save()
        logger.info(f"Loading data for entry ID: {entry_id}")
        df = pd.read_csv(f'data/{entry.id}.csv')

        entry.status = 'Preprocessing Data'
        entry.save()
        logger.info(f"Preprocessing data for entry ID: {entry_id}")
        
        # Handle time series specific preprocessing
        if entry.task == 'TimeSeries':
            if not entry.datetime_column:
                raise ValueError("Datetime column is required for time series tasks")
            if entry.datetime_column not in df.columns:
                raise ValueError(f"Datetime column '{entry.datetime_column}' not found in the dataset")
            try:
                df[entry.datetime_column] = pd.to_datetime(df[entry.datetime_column], format=entry.date_format).dt.strftime('%d/%m/%y')
            except Exception as e:
                raise ValueError(f"Error converting datetime column: {str(e)}")
            
        pipeline = createPipeline(df, entry.target_variable, task=entry.task)
        df_transformed = pipeline.transform(df)

        entry.status = 'Model Selection and Training'
        entry.save()
        logger.info(f"Starting model selection and training for entry ID: {entry_id}")
        
        try:
            task_enum = Task.parse(entry.task)
        except Exception as e:
            raise ValueError(f"Error parsing task type: {str(e)}")
            
        optimizer = RandomSearchOptimizer(task=task_enum, time_budget=300)

        if entry.task.lower() == 'clustering':
            optimizer.fit(df_transformed, None)
        elif entry.task == 'TimeSeries':
            optimizer.fit(None, df_transformed[entry.target_variable])
        else:
            optimizer.fit(
                df_transformed.drop(entry.target_variable, axis=1),
                df_transformed[entry.target_variable]
            )
        entry.status = 'Saving Model'
        entry.save()
        logger.info(f"Saving model for entry ID: {entry_id}")
        model = optimizer.get_optimal_model()
        metric_value = optimizer.get_metric_value()
        entry.model_name = model.__class__.__name__

        # Set evaluation metrics based on task type
        metric_map = {
            'Classification': Metric.ACCURACY,
            'Regression': Metric.RMSE,
            'Clustering': Metric.SILHOUETTE,
            'TimeSeries': Metric.RMSE
        }
        entry.evaluation_metric = metric_map.get(entry.task, Metric.ACCURACY)
        entry.evaluation_metric_value = abs(metric_value) if entry.task == 'TimeSeries' else metric_value
        logger.info(f"Model metrics - Task: {entry.task}, Metric: {entry.evaluation_metric}, Value: {entry.evaluation_metric_value}")

        os.makedirs('models', exist_ok=True)
        os.makedirs('pipelines', exist_ok=True)
        joblib.dump(model, f'models/{entry.id}.pkl')
        joblib.dump(pipeline, f'pipelines/{entry.id}.pkl')
        logger.info(f"Model and pipeline saved for entry ID: {entry_id}")

        entry.status = 'Done'
        entry.save()
        logger.info(f"Training completed successfully for entry ID: {entry_id}")

    except Exception as e:
        entry.status = 'Failed'
        entry.save()
        logger.error(f"Error during model training for entry ID: {entry_id}: {str(e)}")
        raise