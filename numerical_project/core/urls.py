from django.urls import path
from .chapterOne import views as views1
from .chapterTwo import views as views2
from .chapterThree import views as views3

urlpatterns = [
    path('chapterOne/', views1.chapter_one_view, name='capitulo1'),
    path('chapterTwo/', views2.chapter_two_view, name='capitulo2'),
    path('chapterThree/', views3.chapter_three_view, name='capitulo3'),
]