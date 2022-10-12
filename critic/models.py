from django.db import models
from django.conf import settings
from jsonfield import JSONField

import spotipy
import random
import pickle
import os
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

class SpotifyUser(models.Model):
  username = models.CharField(max_length=250)
  spotify_token = models.CharField(max_length=250)
  created_at = models.DateTimeField(auto_now_add=True)

  def __str__(self):
    return self.username

class SpotifyTrack(models.Model):
  spotify_user = models.ForeignKey(SpotifyUser, on_delete=models.CASCADE)
  spotify_id = models.CharField(max_length=250)
  spotify_data = JSONField()
  classification = models.IntegerField(default=0)

  def __str__(self):
    return self.spotify_id

class TrackPrediction(models.Model):
  classification_id = models.IntegerField(default=0, unique=True)
  name = models.CharField(max_length=250)
  title = models.CharField(max_length=250)
  description = models.CharField(max_length=250)

  def __str__(self):
    return self.name

class SpotifyAPI:
  def authenticate(token):
    auth_manager = spotipy.oauth2.SpotifyOAuth(client_id=os.environ.get('SPOTIPY_CLIENT_ID'),
                                               client_secret=os.environ.get('SPOTIPY_CLIENT_SECERET'),
                                               redirect_uri=os.environ.get('SPOTIPY_REDIRECT_URI'),
                                               scope=os.environ.get('SPOTIPY_SCOPE'))
    auth_manager.get_access_token(token)
    spotify = spotipy.Spotify(auth_manager=auth_manager)
    return spotify

  def top_tracks(spotify):
    track_list = spotify.current_user_top_tracks(limit=5, time_range='short_term')
    top_tracks = []
    for track in track_list['items']:
      title = '%s, de %s' % (track['name'], track['artists'][0]['name'])
      top_tracks.append(title)
    return top_tracks

  def top_artists(spotify):
    artist_list = spotify.current_user_top_artists(limit=5, time_range='long_term')
    top_artists = []
    for artist in artist_list['items']:
      top_artists.append(artist['name'])
    return top_artists

class AllowedGenre:
  def get_list():
    file_path = os.path.join(settings.BASE_DIR / 'critic' / 'lib', os.path.basename('allowed_genres.json'))
    json_data = open(file_path)
    data = json.load(json_data)
    return data['genres']

class SpotifyAPIUser:
  def get_genre_from_artist(genres):
    allowed_genres = AllowedGenre.get_list()
    genre = ''
    for item in allowed_genres:
      if item in genres:
        genre = item
        break
    if genre == '':
      genre = 'Other'

    return genre

  def format_spotify_data(spotify_track, features, artists):
    spotify_data = {
      'song_name': spotify_track['name'],
      'song_id': spotify_track['id'],
      'artist_name': spotify_track['artists'][0]['name'],
      'album_name': spotify_track['album']['name'],
      'release_date': spotify_track['album']['release_date'],
    }
    audio_feature = list(filter(lambda f: f['id'] == spotify_track['id'], features))[0]
    artist = list(filter(lambda f: f['id'] == spotify_track['artists'][0]['id'], artists))[0]

    spotify_data['artist_genre'] = SpotifyAPIUser.get_genre_from_artist(artist['genres'])
    spotify_data['acousticness'] = audio_feature['acousticness']
    spotify_data['danceability'] = audio_feature['danceability']
    spotify_data['duration_ms'] = audio_feature['duration_ms']
    spotify_data['energy'] = audio_feature['energy']
    spotify_data['instrumentalness'] = audio_feature['instrumentalness']
    spotify_data['liveness'] = audio_feature['liveness']
    spotify_data['loudness'] = audio_feature['loudness']
    spotify_data['speechiness'] = audio_feature['speechiness']
    spotify_data['tempo'] = audio_feature['tempo']
    spotify_data['valence'] = audio_feature['valence']

    return spotify_data

  def storage_data(token):
    spotify = SpotifyAPI.authenticate(token)

    spotify_user = spotify.current_user()
    user = SpotifyUser(username=spotify_user['id'], spotify_token=token)
    user.save()

    top_tracks_array = spotify.current_user_top_tracks(limit=50, time_range='medium_term')
    track_ids = []
    artist_ids = []
    for track in top_tracks_array['items']:
      track_ids.append(track['id'])
      artist_ids.append(track['artists'][0]['id'])
    features = spotify.audio_features(track_ids)
    artists = spotify.artists(artist_ids)

    for spotify_track in top_tracks_array['items']:
      spotify_data = SpotifyAPIUser.format_spotify_data(spotify_track, features, artists['artists'])
      track = SpotifyTrack(spotify_user=user, spotify_id=spotify_track['id'], spotify_data=spotify_data)
      track.save()

    return user

class PredictionHelper():

  def format_prediction(prediction_array):
    prediction = dict(sorted(prediction_array.items(), key=lambda item: item[1], reverse=True))

    prediction_titles = []
    prediction_texts = []
    tot = 0
    for i, idx in enumerate(prediction):
      qtd = prediction[str(idx)]
      pred_item = TrackPrediction.objects.get(classification_id=idx)

      if i < 4:
        prediction_titles.append(pred_item.title)

      if tot < 40:
        tot += qtd
        qtd_per = qtd * 100 / 50
        prediction_texts.append({'description':pred_item.description, 'percentage':qtd_per})
    
    return {'titles': prediction_titles, 'texts': prediction_texts}

  def format_dataframe(tracks):
    data = {
      'artist_genre': [],
      'release_date': [],
      'acousticness': [],
      'danceability': [],
      'duration_ms': [],
      'energy': [],
      'instrumentalness': [],
      'liveness': [],
      'loudness': [],
      'speechiness': [],
      'tempo': [],
      'valence': []
    }
    for track in tracks:
      data['artist_genre'].append(track.spotify_data['artist_genre'])
      data['release_date'].append(track.spotify_data['release_date'][0 : 4])
      data['acousticness'].append(track.spotify_data['acousticness'])
      data['danceability'].append(track.spotify_data['danceability'])
      data['duration_ms'].append(track.spotify_data['duration_ms'])
      data['energy'].append(track.spotify_data['energy'])
      data['instrumentalness'].append(track.spotify_data['instrumentalness'])
      data['liveness'].append(track.spotify_data['liveness'])
      data['loudness'].append(track.spotify_data['loudness'])
      data['speechiness'].append(track.spotify_data['speechiness'])
      data['tempo'].append(track.spotify_data['tempo'])
      data['valence'].append(track.spotify_data['valence'])

    df = pd.DataFrame.from_dict(data)
    X = df[['release_date', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']]

    le = preprocessing.LabelEncoder()
    le.fit(AllowedGenre.get_list())
    le_genres = le.transform(df['artist_genre'])
    X['artist_genre'] = le_genres
    
    pickle_folder = settings.BASE_DIR / 'critic' / 'lib'
    file_path = os.path.join(pickle_folder, os.path.basename('scaler.sav'))
    file = open(file_path,'rb')
    scaler = pickle.load(file)
    X = scaler.transform(X)

    return X
    
  def predict_user_tracks(user):
    user_tracks = SpotifyTrack.objects.filter(spotify_user_id=user.id)
    data_df = PredictionHelper.format_dataframe(user_tracks)

    pickle_folder = settings.BASE_DIR / 'critic' / 'lib'
    file_path = os.path.join(pickle_folder, os.path.basename('random_forest_model.sav'))
    file = open(file_path,'rb')
    classifier = pickle.load(file)
    classifiers = classifier.predict(data_df)

    for i, c in enumerate(classifiers):
      classification = classifiers[i]
      track = user_tracks[i]
      track.classification = classification
      track.save()

    prediction = {}
    for c in classifiers:
      if str(c) in prediction:
        prediction[str(c)] += 1
      else:
        prediction[str(c)] = 1
    return PredictionHelper.format_prediction(prediction)
