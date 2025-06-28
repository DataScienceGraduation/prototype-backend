#!/bin/sh

# Apply database migrations
echo "Apply database migrations"
python manage.py migrate

# Start server
echo "Starting server"
gunicorn automl_backend.wsgi:application --bind 0.0.0.0:8000
