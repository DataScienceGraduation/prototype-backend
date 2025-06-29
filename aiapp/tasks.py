from .models import Dashboard
from celery import shared_task
import pandas as pd
from .agents.orchestrator import DashboardPipeline
from django.core.files.storage import default_storage
import os
from automlapp.models import ModelEntry
import logging
from django.db import transaction
from django.conf import settings

logger = logging.getLogger(__name__)

@shared_task
def suggest_charts_task(model_id):
    dataset_path = f"data/{model_id}.csv"
    if not default_storage.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    try:
        model_entry = ModelEntry.objects.get(id=model_id)
        with transaction.atomic():
            dashboard, created = Dashboard.objects.get_or_create(
                model_entry=model_entry,
                defaults={
                    "charts": [],
                    "title": model_entry.name,
                    "description": model_entry.description,
                    "status": "generating"
                }
            )
            if not created:
                dashboard.status = "generating"
                dashboard.save(update_fields=["status"])
        logger.info(f"Dashboard status set to 'generating' for model {model_id}")
        with default_storage.open(dataset_path) as f:
            df = pd.read_csv(f)
        pipeline = DashboardPipeline()
        charts = pipeline.run(df, model_description=model_entry.description)
        dashboard.charts = charts
        dashboard.status = "completed"
        dashboard.title = model_entry.name
        dashboard.description = model_entry.description
        dashboard.save(update_fields=["charts", "status", "title", "description"])
        logger.info(f"Dashboard status set to 'completed' for model {model_id}")
        return charts
    except ModelEntry.DoesNotExist:
        logger.error(f"ModelEntry with id {model_id} does not exist.")
        raise
    except Exception as e:
        logger.error(f"Error saving dashboard for model {model_id}: {e}")
        try:
            dashboard = Dashboard.objects.get(model_entry_id=model_id)
            dashboard.status = "failed"
            dashboard.save(update_fields=["status"])
            logger.info(f"Dashboard status set to 'failed' for model {model_id}")
        except Exception as ex:
            logger.error(f"Could not update dashboard status to 'failed': {ex}")
        raise