from rest_framework import serializers
from .models import APIKey

class APIKeySerializer(serializers.ModelSerializer):
    class Meta:
        model = APIKey
        fields = ['id', 'name', 'key', 'created_at']
        read_only_fields = ['key', 'created_at']
