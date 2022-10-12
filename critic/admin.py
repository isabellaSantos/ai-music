from django.contrib import admin
from .models import TrackPrediction

class TrackPredictionAdmin(admin.ModelAdmin):
  ...

admin.site.register(TrackPrediction, TrackPredictionAdmin)