from django.urls import path
from main.views import predict

urlpatterns = [
    path('predict/', predict),
]