from django.db import models
from automlapp.models import ModelEntry
import json
import logging

logger = logging.getLogger(__name__)

class Report(models.Model):
    """
    Model to store generated reports for trained models
    """
    id = models.AutoField(primary_key=True)
    model_entry = models.ForeignKey(ModelEntry, on_delete=models.CASCADE, related_name='reports')
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # AI-generated insights
    ai_insights = models.TextField(blank=True, help_text="AI-generated insights from Gemini")

    # Report metadata
    report_type = models.CharField(max_length=50, default='analysis')  # analysis, performance, data_quality
    status = models.CharField(max_length=20, default='pending')  # pending, generating, completed, failed

    # Progress tracking for Celery tasks
    task_id = models.CharField(max_length=255, blank=True, null=True, help_text="Celery task ID")
    progress_percentage = models.IntegerField(default=0, help_text="Progress percentage (0-100)")
    current_step = models.CharField(max_length=200, blank=True, help_text="Current processing step")

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Report for {self.model_entry.name} - {self.title}"


class ChartData(models.Model):
    """
    Model to store chart data and configurations
    """
    CHART_TYPES = [
        ('pie', 'Pie Chart'),
        ('line', 'Line Chart'),
        ('bar', 'Bar Chart'),
        ('scatter', 'Scatter Plot'),
        ('histogram', 'Histogram'),
        ('box', 'Box Plot'),
        ('correlation', 'Correlation Matrix'),
    ]

    id = models.AutoField(primary_key=True)
    report = models.ForeignKey(Report, on_delete=models.CASCADE, related_name='charts')
    chart_type = models.CharField(max_length=20, choices=CHART_TYPES)
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)

    # Chart configuration and data
    chart_config = models.JSONField(default=dict, help_text="Chart configuration (colors, layout, etc.)")
    chart_data = models.JSONField(default=dict, help_text="Chart data (labels, values, etc.)")

    # LLM-generated content
    chart_code = models.TextField(blank=True, help_text="LLM-generated Python code for the chart")
    llm_reasoning = models.TextField(blank=True, help_text="LLM's reasoning for choosing and generating this chart")

    # Generated chart files
    chart_image_base64 = models.TextField(blank=True, help_text="Base64 encoded chart image")
    chart_html = models.TextField(blank=True, help_text="Interactive HTML chart")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['id']

    def __str__(self):
        return f"{self.chart_type.title()} Chart - {self.title}"

    def get_chart_data_as_dict(self):
        """Helper method to get chart data as dictionary"""
        try:
            if isinstance(self.chart_data, str):
                return json.loads(self.chart_data)
            elif isinstance(self.chart_data, dict):
                return self.chart_data
            else:
                return {}
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Error parsing chart data for chart {self.id}: {e}")
            return {'error': 'Invalid chart data format'}

    def set_chart_data_from_dict(self, data_dict):
        """Helper method to set chart data from dictionary"""
        self.chart_data = data_dict


class DataInsight(models.Model):
    """
    Model to store specific data insights and statistics
    """
    INSIGHT_TYPES = [
        ('feature_importance', 'Feature Importance'),
        ('data_distribution', 'Data Distribution'),
        ('correlation_analysis', 'Correlation Analysis'),
        ('missing_values', 'Missing Values Analysis'),
        ('outlier_detection', 'Outlier Detection'),
        ('class_balance', 'Class Balance'),
        ('performance_metrics', 'Performance Metrics'),
    ]

    id = models.AutoField(primary_key=True)
    report = models.ForeignKey(Report, on_delete=models.CASCADE, related_name='insights')
    insight_type = models.CharField(max_length=30, choices=INSIGHT_TYPES)
    title = models.CharField(max_length=255)
    description = models.TextField()

    # Insight data
    insight_data = models.JSONField(default=dict, help_text="Structured insight data")
    priority = models.IntegerField(default=1, help_text="Priority level (1=high, 5=low)")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['priority', '-created_at']

    def __str__(self):
        return f"{self.insight_type.replace('_', ' ').title()} - {self.title}"


class ReportTemplate(models.Model):
    """
    Model to store report templates for different model types
    """
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    description = models.TextField()
    model_task_type = models.CharField(max_length=50)  # Classification, Regression, TimeSeries, Clustering

    # Template configuration
    template_config = models.JSONField(default=dict, help_text="Template configuration")
    chart_types = models.JSONField(default=list, help_text="List of chart types to generate")
    insight_types = models.JSONField(default=list, help_text="List of insight types to generate")

    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['model_task_type', 'name']

    def __str__(self):
        return f"{self.name} ({self.model_task_type})"


class Dashboard(models.Model):
    """
    Stores dashboard chart suggestions for a single model. One dashboard per ModelEntry.
    """
    model_entry = models.OneToOneField('automlapp.ModelEntry', on_delete=models.CASCADE, related_name='dashboard', primary_key=True)
    title = models.CharField(max_length=255, blank=True, default="")
    description = models.TextField(blank=True, default="")
    charts = models.JSONField(default=list, help_text="Chart suggestions for this model")
    status = models.CharField(max_length=20, default="pending", help_text="Dashboard generation status: pending, generating, completed, failed")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Dashboard for {self.model_entry.name}"


class DataProfile(models.Model):
    """
    Model to store generated data profiles for models
    """
    id = models.AutoField(primary_key=True)
    model_entry = models.ForeignKey('automlapp.ModelEntry', on_delete=models.CASCADE, related_name='data_profiles')
    html_content = models.TextField(help_text="Generated HTML profile content")
    dataset_shape = models.CharField(max_length=50, help_text="Dataset shape (rows, columns)")
    columns = models.TextField(help_text="JSON string of column names")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        unique_together = ['model_entry']

    def __str__(self):
        return f"Data Profile for {self.model_entry.name}"