from celery import shared_task
from .models import ModelEntry
from automl.optimizers import RandomSearchOptimizer
from automl.enums import Task, Metric
from automl.functions import createPipeline
import joblib
import os
from django.conf import settings
import pandas as pd
import logging
import numpy as np
from automl.config import CLUSTERING_CONFIG
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_clustering_score(labels, features):
    """
    Calculate a custom clustering score combining Silhouette and Davies-Bouldin metrics.
    Both metrics are scaled using MinMax scaling, and Davies-Bouldin is inverted.
    """
    # Calculate raw metrics
    silhouette = silhouette_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)
    
    # Invert Davies-Bouldin (since lower is better)
    davies_bouldin = 1 / (1 + davies_bouldin)  # Adding 1 to avoid division by zero
    
    # Scale both metrics to [0,1] range
    scaler = MinMaxScaler()
    metrics = np.array([[silhouette], [davies_bouldin]])
    scaled_metrics = scaler.fit_transform(metrics)
    
    # Calculate weighted score
    final_score = 0.65 * scaled_metrics[0][0] + 0.35 * scaled_metrics[1][0]
    return final_score, silhouette, davies_bouldin

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
        
        file_path = os.path.join(settings.DATA_DIR, f'{entry.id}.csv')
        df = pd.read_csv(file_path)

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
            
        optimizer = RandomSearchOptimizer(task=task_enum, time_budget=100)
        metric_value = None  # Initialize metric_value

        if entry.task.lower() == 'clustering':
            max_clusters = int(np.sqrt(len(df_transformed)))
            CLUSTERING_CONFIG["models"]["KMeans"]["n_clusters"] = list(np.arange(2, max_clusters + 1, 1))
            optimizer.fit(df_transformed, None)
            
            # Get the optimal model and calculate custom score
            model = optimizer.get_optimal_model()
            labels = model.predict(df_transformed)
            custom_score, silhouette, davies_bouldin = calculate_clustering_score(labels, df_transformed)
            metric_value = custom_score
            
            # Log individual metric values
            logger.info(f"Clustering metrics - Silhouette: {silhouette:.4f}, Davies-Bouldin: {davies_bouldin:.4f}, Custom Score: {custom_score:.4f}")
            
        elif entry.task == 'TimeSeries':
            optimizer.fit(None, df_transformed[entry.target_variable])
            model = optimizer.get_optimal_model()
            metric_value = optimizer.get_metric_value()
        else:
            optimizer.fit(
                df_transformed.drop(entry.target_variable, axis=1),
                df_transformed[entry.target_variable]
            )
            model = optimizer.get_optimal_model()
            metric_value = optimizer.get_metric_value()

        entry.status = 'Saving Model'
        entry.save()
        logger.info(f"Saving model for entry ID: {entry_id}")
        entry.model_name = model.__class__.__name__

        # Set evaluation metrics based on task type
        metric_map = {
            'Classification': Metric.ACCURACY,
            'Regression': Metric.RMSE,
            #'Clustering': Metric.SILHOUETTE,
            'Clustering': 'CustomClusteringScore',  # Changed from Metric.SILHOUETTE
            'TimeSeries': Metric.RMSE
        }
        entry.evaluation_metric = metric_map.get(entry.task, Metric.ACCURACY)
        entry.evaluation_metric_value = abs(metric_value) if entry.task == 'TimeSeries' else metric_value
        logger.info(f"Model metrics - Task: {entry.task}, Metric: {entry.evaluation_metric}, Value: {entry.evaluation_metric_value}")

        model_path = os.path.join(settings.MODELS_DIR, f'{entry.id}.pkl')
        pipeline_path = os.path.join(settings.PIPELINES_DIR, f'{entry.id}.pkl')
        joblib.dump(model, model_path)
        joblib.dump(pipeline, pipeline_path)
        logger.info(f"Model and pipeline saved for entry ID: {entry_id}")

        entry.status = 'Done'
        entry.save()
        logger.info(f"Training completed successfully for entry ID: {entry_id}")

    except Exception as e:
        entry.status = 'Failed'
        entry.save()
        logger.error(f"Error during model training for entry ID: {entry_id}: {str(e)}")
        raise