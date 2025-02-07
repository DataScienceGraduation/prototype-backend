from django.db import models

# Create your models here.
class ModelEntry(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=255)
    description = models.TextField()
    task = models.CharField(max_length=255)
    target_variable = models.CharField(max_length=255)
    list_of_features = models.TextField(blank=True)
    status = models.CharField(max_length=255)
    model_name = models.CharField(max_length=255)
    evaluation_metric = models.CharField(max_length=255)
    evaluation_metric_value = models.FloatField()