from django.urls import path
from . import views

app_name = 'chapterOne'

urlpatterns = [
    path('', views.chapter_one_view, name='chapter_one_main'),
]