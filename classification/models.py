from django.db import models

# Create your models here.

class Similar_Tweet(models.Model):

    def __init__(self, text, tweet_id, cos_sim, given_label, pred_label, pred_proba):
        self.text = text
        self.tweet_id = tweet_id
        self.cos_sim = cos_sim
        self.given_label = given_label
        self.pred_label = pred_label
        self.pred_proba = pred_proba
