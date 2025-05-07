from django.db import models
from django.contrib.auth.models import AbstractUser

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
    datetime_column = models.CharField(max_length=255, blank=True, null=True)
    date_format = models.CharField(max_length=255, blank=True, null=True)

class User(AbstractUser):
    models = models.ManyToManyField(ModelEntry)