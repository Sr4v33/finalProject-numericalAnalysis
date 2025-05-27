from django.shortcuts import render

def index_view(request):
    context = {
        'page_title': 'Proyecto de Análisis Numérico',
        'chapters': [
            {'name': 'Capítulo 1: Solución de Ecuaciones no Lineales', 'url_name': 'chapterOne:chapter_one_main'},
            {'name': 'Capítulo 2: Sistemas de Ecuaciones Lineales', 'url_name': 'chapterTwo:chapter_two_main'},
            {'name': 'Capítulo 3: Interpolación', 'url_name': 'chapterThree:chapter_three_main'}
        ]
    }
    return render(request, 'core/index.html', context)