from django.urls import path
from . import views

urlpatterns = [
    # Reporting endpoints
    path('generate/', views.generate_report, name='generate_report'),
    path('get/', views.get_report, name='get_report'),
    path('list/', views.list_reports, name='list_reports'),
    path('export-chart-csv/', views.export_chart_csv, name='export_chart_csv'),
    path('chart-stats/', views.get_chart_stats, name='get_chart_stats'),

    # Celery task endpoints
    path('task-status/', views.get_task_status, name='get_task_status'),
    path('report-by-task/', views.get_report_by_task, name='get_report_by_task'),

    # Dashboard endpoints
    path('start-suggest-charts/', views.start_suggest_charts, name='start_suggest_charts'),
    path('get-suggest-charts-result/', views.get_suggest_charts_result, name='get_suggest_charts_result'),
    path('get-dashboard-by-model/', views.get_dashboard_by_model, name='get_dashboard_by_model'),
    
    # Data profiling endpoints
    path('dataprofile/', views.dataprofile, name='dataprofile'),
    path('delete-dataprofile/', views.delete_dataprofile, name='delete_dataprofile'),

    path('get-latest-report-for-model/', views.get_latest_report_for_model, name='get_latest_report_for_model'),
]