from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='classification-home'),
    path('about/', views.about, name='classification-about'),
    #path('result/', views.result, name='classification-result'),
    path('predict/', views.predict, name='classification-predict')
]
