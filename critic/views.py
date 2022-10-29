from django.http import HttpResponse
from django.http import Http404
from django.shortcuts import render

import os
import spotipy
import spotipy.util as util

from .models import SpotifyUser, SpotifyAPIUser, SpotifyAPI, PredictionHelper

def authenticate(request):
  client_id = os.environ.get('SPOTIPY_CLIENT_ID')
  redirect_uri = os.environ.get('SPOTIPY_REDIRECT_URI')
  scope = os.environ.get('SPOTIPY_SCOPE')
  spotify_uri = 'https://accounts.spotify.com/authorize?response_type=code&client_id=' + client_id + '&scope=' + scope + '&redirect_uri=' + redirect_uri
  return render(request, 'critic/pages/authenticate.html', {'spotify_uri': spotify_uri})

def storage_data(request):
  token = request.GET.get('code')
  if token:
    spotify = SpotifyAPI.authenticate(request, token)
    user = SpotifyAPIUser.storage_data(spotify, token)
    return render(request, 'critic/pages/storage_data.html', {'spotify_user': user})
  else:
    return render(request, 'critic/pages/error.html', {})

  return render(request, 'critic/pages/storage_data.html', {'spotify_user': user})

def analyse(request, spotify_username):
  try:
    user = SpotifyUser.objects.filter(username=spotify_username).latest('id')
  except SpotifyUser.DoesNotExist:
    return render(request, 'critic/pages/error.html', {})

  spotify = SpotifyAPI.authenticate(request, user.spotify_token)
  top_tracks = SpotifyAPI.top_tracks(spotify)
  top_artists = SpotifyAPI.top_artists(spotify)

  prediction = PredictionHelper.predict_user_tracks(user)
  title = "Uhm, você é o tipo %s quando se trata de música" % '-'.join(prediction['titles'])
  return render(request, 'critic/pages/analysis.html', {'title': title,'texts':prediction['texts'], 'top_tracks': top_tracks, 'top_artists': top_artists})
