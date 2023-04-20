from django.urls import path
from . import views

urlpatterns = [
    path('register-user/', views.RegisterUser, name='register_user'),
    path('<int:id>/register-signature/', views.RegisterSignature, name='register_signature'),
    path('scan/', views.ScanSignature, name='scan_signature'),
    path('retrain/', views.RetrainModel, name='retrain_model'),
    path('', views.index, name='index'),
]
