from django.contrib import admin
from django.urls import path
from automlapp.views import loadData, trainModel, getAllModels, infer, getModel, login, register
urlpatterns = [
    path('admin/', admin.site.urls),
    path('loadData/', loadData, name='loadData'), 
    path('trainModel/', trainModel, name='trainModel'),  
    path('getAllModels/', getAllModels, name='getAllModels'),
    path('infer/', infer, name='infer'),
    path('getModel/', getModel, name='getModel'),
    path('login/', login, name='login'),
    path('register/', register, name='register')
]
