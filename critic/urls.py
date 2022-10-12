from django.urls import path

from . import views

urlpatterns = [
  path('', views.authenticate, name='authenticate'),
  path('authenticate', views.authenticate, name='authenticate'),
  path('storage_data', views.storage_data, name='storage_data'),
  path('analysis/<str:spotify_username>', views.analyse, name='analyse'),
]