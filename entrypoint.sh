#!/bin/sh

# Apply database migrations
echo "Applying database migrations..."
python manage.py migrate

echo "Starting server"
gunicorn automl_backend.wsgi:application --bind 0.0.0.0:8000
