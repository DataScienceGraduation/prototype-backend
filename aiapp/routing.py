from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r"ws/dashboard/(?P<model_id>\w+)/$", consumers.DashboardConsumer.as_asgi()),
    re_path(r"ws/report/(?P<task_id>[0-9a-f\-]+)/$", consumers.ReportConsumer.as_asgi()),
]