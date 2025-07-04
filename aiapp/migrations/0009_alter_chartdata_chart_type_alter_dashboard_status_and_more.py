# Generated by Django 5.1.2 on 2025-07-03 15:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('aiapp', '0008_dataprofile'),
    ]

    operations = [
        migrations.AlterField(
            model_name='chartdata',
            name='chart_type',
            field=models.CharField(choices=[('pie', 'Pie Chart'), ('line', 'Line Chart'), ('bar', 'Bar Chart'), ('scatter', 'Scatter Plot'), ('histogram', 'Histogram'), ('box', 'Box Plot'), ('correlation', 'Correlation Matrix')], max_length=255),
        ),
        migrations.AlterField(
            model_name='dashboard',
            name='status',
            field=models.CharField(default='pending', help_text='Dashboard generation status: pending, generating, completed, failed', max_length=255),
        ),
        migrations.AlterField(
            model_name='datainsight',
            name='insight_type',
            field=models.CharField(choices=[('feature_importance', 'Feature Importance'), ('data_distribution', 'Data Distribution'), ('correlation_analysis', 'Correlation Analysis'), ('missing_values', 'Missing Values Analysis'), ('outlier_detection', 'Outlier Detection'), ('class_balance', 'Class Balance'), ('performance_metrics', 'Performance Metrics')], max_length=255),
        ),
        migrations.AlterField(
            model_name='report',
            name='current_step',
            field=models.CharField(blank=True, help_text='Current processing step', max_length=255),
        ),
        migrations.AlterField(
            model_name='report',
            name='report_type',
            field=models.CharField(default='analysis', max_length=255),
        ),
        migrations.AlterField(
            model_name='report',
            name='status',
            field=models.CharField(default='pending', max_length=255),
        ),
    ]
