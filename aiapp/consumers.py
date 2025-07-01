import json
from channels.generic.websocket import AsyncWebsocketConsumer

class DashboardConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.model_id = self.scope["url_route"]["kwargs"]["model_id"]
        self.group_name = f"dashboard_{self.model_id}"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def dashboard_ready(self, event):
        await self.send(text_data=json.dumps({
            "type": "dashboard_ready",
            "model_id": self.model_id,
        }))

class ReportConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.task_id = self.scope["url_route"]["kwargs"]["task_id"]
        self.group_name = f"report_{self.task_id}"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def report_ready(self, event):
        await self.send(text_data=json.dumps({
            "type": "report_ready",
            "task_id": self.task_id,
            "report_id": event.get("report_id"),
        }))