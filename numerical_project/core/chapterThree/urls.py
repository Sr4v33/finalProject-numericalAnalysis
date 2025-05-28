from django.urls import path
from . import views

app_name = 'chapterThree'

urlpatterns = [
    path('', views.interpolation_view, name='chapter_three_main'),
    path('graph/', views.graph_view, name='graph_ch3'),
    path('compare/', views.compare_methods_view, name='compare_methods_ch3'),
    path('download-pdf/', views.download_pdf_report_view, name='download_pdf_report'),
]





