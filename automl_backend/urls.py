from django.contrib import admin
from django.urls import path
from automl.views import loadData, trainModel, getAllModels, infer, getModel
urlpatterns = [
    path('admin/', admin.site.urls),
    path('loadData/', loadData, name='loadData'), 
    path('trainModel/', trainModel, name='trainModel'),  
    path('getAllModels/', getAllModels, name='getAllModels'),
    path('infer/', infer, name='infer'),
    path('getModel/', getModel, name='getModel')
]
