from django.urls import path
from . import views

app_name = 'chapterOne'

urlpatterns = [
    path('', views.chapter_one_view, name='chapter_one_main'),
    path('compare/', views.compare_methods_view, name='compare_methods'),
    path('download-pdf/', views.download_pdf_report_view, name='download_pdf_report'),
    path('graph/', views.graph_function_view, name='graph_function'),
]