from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

# Set default Django settings module for 'celery'
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'automl_backend.settings')

# Create the Celery app instance for aiapp
app = Celery('aiapp')

# Load task modules from all registered Django app configs
app.config_from_object('django.conf:settings', namespace='CELERY')

# Autodiscover tasks in aiapp
app.autodiscover_tasks(['aiapp'])
