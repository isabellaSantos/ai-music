from django.core.management.base import BaseCommand, CommandError
from critic.models import TrackPrediction

class Command(BaseCommand):
  help = 'Add the necessary sample data for the project'

  def handle(self, *args, **options):
    predictions = [
      {'id': 1, 'name': 'kpop dinossauro', 'title': 'kpoper-old-school', 'description': 'kpop das antigas, de SHINEe para trás'},
      {'id': 2, 'name': 'kpop gg', 'title': 'gg-stan', 'description': 'conceitos cute, girl crush, colorido, sexy, retro. Enfim, se for GG, você tá dentro'},
      {'id': 3, 'name': 'kpop bg', 'title': 'bg-stan', 'description': 'boy group convicta'},
      {'id': 4, 'name': 'divas pop', 'title': 'briga-no-twitter', 'description': 'divas pop'},
      {'id': 5, 'name': 'pop 80s / 90s', 'title': 'pop-erudito', 'description': 'pop 80s / 90s'},
      {'id': 6, 'name': 'pop indie', 'title': 'pitchfork', 'description': 'pop sem pretensão de charts, mas com muita aclamação da crítica'},
      {'id': 7, 'name': 'indie 00s', 'title': 'indie', 'description': 'indie 00s'},
      {'id': 8, 'name': 'emo', 'title': 'emo-com-orgulho', 'description': 'emo'},
      {'id': 9, 'name': 'rock classico', 'title': '🤘', 'description': 'rock classico'},
      {'id': 10, 'name': 'rock 80s / 90s', 'title': 'rockeirx-cabeludx', 'description': 'rock 80s / 90s'},
      {'id': 11, 'name': 'hip-hop 00s', 'title': 'hip-hop', 'description': 'hip-hop 00s'},
      {'id': 12, 'name': 'rap', 'title': 'rap-raiz', 'description': 'rap'},
      {'id': 13, 'name': 'trap', 'title': 'trap', 'description': 'trap'},
      {'id': 14, 'name': 'pop br', 'title': 'anitter', 'description': 'pop br'},
      {'id': 15, 'name': 'sofrencia sertanejo', 'title': 'só-sofrencia', 'description': 'sofrencia'},
      {'id': 16, 'name': 'piseiro', 'title': 'joao-gomes', 'description': 'piseiro'},
      {'id': 17, 'name': 'musica classica', 'title': 'bethoven', 'description': 'musica classica'},
      {'id': 18, 'name': 'eletronica', 'title': 'tuts-tuts', 'description': 'eletronica'},
      {'id': 19, 'name': 'tik tok', 'title': 'tiktoker', 'description': 'tiktok gen z'},
      {'id': 20, 'name': 'charts', 'title': 'bb100', 'description': 'escutar apenas músicas que estão nos topos das paradas e desconhece algo lançado a menos de 2 anos atrás'},
      {'id': 21, 'name': 'boomer', 'title': 'boomer', 'description': 'música dos anos 80, porque naquela época que as coisas eram boas'},
      {'id': 22, 'name': 'outros', 'title': '-', 'description': '-'},
    ]

    for prediction in predictions:
      track_prediction = TrackPrediction(
        classification_id = prediction['id'],
        name = prediction['name'],
        title = prediction['title'],
        description = prediction['description'],
      )
      track_prediction.save()

      self.stdout.write(self.style.SUCCESS('Successfully created prediction "%s"' % track_prediction.name))
