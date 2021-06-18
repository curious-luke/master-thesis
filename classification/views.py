from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage
from pathlib import Path
import pandas as pd
import tensorflow_hub as tf_hub

# Import from Django project
from .models import Similar_Tweet
from classification.predict_tweet import hate_prediction, is_hate_speech
from classification.similarity_use import similarity_calc
#from classification.similarity_spacy import similarity_calc

### Load Universal Sentence Encoder
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
print('Loading USE module ...')
USEmodel = tf_hub.load(module_url)
print ("Module %s loaded!" % module_url)


### Functions to call the HTML pages

def home(request):
    return render(request, 'classification/home.html')


def about(request):
    return render(request, 'classification/about.html')


def predict(request):
    # Input
    inp_txt = str(request.POST.get('tweet_text'))
    input_text = [inp_txt]
    language = request.POST.get('text_language')
    #language = bool(request.POST.get('text_language'))


    # Classification
    pred_class, pred_prob = hate_prediction(input_text, language)
    prob = round(float(pred_prob)*100, 2)
    pred = is_hate_speech(pred_class)

    # Similarity Calculation USE
    ranking = similarity_calc(input_text, language, USEmodel)

    # Similarity Calculation SpaCy
    #ranking = similarity_calc(inp_txt, language)

    tweet_obj = []
    for index, row in ranking.iterrows():
        tweet_text = row['text']
        tweet_id = row['text_id']
        cos_sim = round(float(row['cos_sim'])*100, 2)
        given_label = is_hate_speech(row['task_1_int'])
        pred_label = is_hate_speech(row['pred_label'])
        pred_proba = row['pred_proba']

        tweet = Similar_Tweet(tweet_text, tweet_id, cos_sim, given_label, pred_label, pred_proba)
        tweet_obj.append(tweet)

    # Save variables in context
    context = {
        "input_text": inp_txt,
        "pred": pred,
        "prob": prob,
        "sim_tweets": tweet_obj
    }

    return render(request, 'classification/predict.html', context)

