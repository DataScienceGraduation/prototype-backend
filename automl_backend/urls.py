from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from automlapp.views import loadData, trainModel, getAllModels, infer, getModel, getModelDataset, login, register, is_valid_token, batchPredict
from automlapp.db_views import DBConnectionViewSet

router = DefaultRouter()
router.register(r'db_connection', DBConnectionViewSet, basename='db_connection')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('loadData/', loadData, name='loadData'),
    path('trainModel/', trainModel, name='trainModel'),
    path('getAllModels/', getAllModels, name='getAllModels'),
    path('infer/', infer, name='infer'),
    path('getModel/', getModel, name='getModel'),
    path('getModelDataset/', getModelDataset, name='getModelDataset'),
    path('login/', login, name='login'),
    path('register/', register, name='register'),
    path('isAuthenticated/', is_valid_token, name='isAuthenticated'),
    path('aiapp/', include('aiapp.urls')),
    path('automlapp/batch_predict/',batchPredict , name='batchPredict'),
    path('api/', include(router.urls)),
]
