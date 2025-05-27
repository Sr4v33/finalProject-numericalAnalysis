from django.urls import path
from . import views

app_name = 'chapterTwo'

urlpatterns = [
    path('', views.chapter_two_view, name='chapter_two_main'),
    path('compare/', views.compare_methods_ch2_view, name='compare_methods_ch2'),
    path('download-pdf/', views.download_pdf_report_ch2_view, name='download_pdf_report_ch2'),
]