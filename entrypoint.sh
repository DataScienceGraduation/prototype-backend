#!/bin/sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Apply database migrations
echo "Applying database migrations..."
python manage.py migrate

echo "Starting server"
gunicorn automl_backend.wsgi:application --bind 0.0.0.0:8000
