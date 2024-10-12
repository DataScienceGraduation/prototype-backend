from django.contrib import admin
from django.urls import path
from automl.views import loadData, trainModel

urlpatterns = [
    path('admin/', admin.site.urls),
    path('loadData/', loadData, name='loadData'), 
    path('trainModel/', trainModel, name='trainModel'),  
]
