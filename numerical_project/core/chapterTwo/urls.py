from django.urls import path
from . import views

urlpatterns = [
    path('', views.chapter_view, name='chapter_two'),
]
