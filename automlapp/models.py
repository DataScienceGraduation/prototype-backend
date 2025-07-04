from django.db import models
from django.contrib.auth.models import AbstractUser
import secrets

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

class APIKey(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='api_keys')
    key = models.CharField(max_length=64, unique=True, editable=False)
    name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        if not self.key:
            self.key = secrets.token_hex(32)
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.name} ({self.user.username})"