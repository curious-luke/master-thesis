import json
import numpy as np
import tensorflow as tf
import keras.models
from tensorflow.keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences


# Prediction with trained model
def hate_prediction(input_text, language):

    ### Load Tokenizer
    with open('model/tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)

    ### Select filepath for language model
    if language == True:
        filep = 'model/saved_model_en.h5' #english model
    else:
        filep = 'model/saved_model_de.h5' #german model

    ### Load model and padding for input text
    model = keras.models.load_model(filep, compile = True)
    ps = pad_sequences(tokenizer.texts_to_sequences(input_text), maxlen=180 , padding='post')

    ### Predict class for input text
    prediction = model.predict_classes(ps)
    #prediction = model(ps)
    prob = model.predict_proba(ps)

    return prediction, prob


def is_hate_speech(pred_class):
    if pred_class == 1:
        pred = "Hate Speech"
    else:
        pred = "No Hate Speech"
    return pred
